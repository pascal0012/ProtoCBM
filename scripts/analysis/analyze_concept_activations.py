"""
Compare concept activation patterns across ProtoCBM, CBM, and Oracle-ProtoCBM models.

Produces three types of plots:
1. Concept group detail: For a chosen concept group (e.g. has_wing_color::),
   violin plots of sigmoid scores per option, one row per model.
2. Global bar plot: Mean predicted probability per concept across all test samples,
   comparing models side-by-side.
3. Per body-part bar plot: Aggregate concept activations by body part per model.

Usage:
    python analyze_concept_activations.py --config configs/protocbm/analysis/concept_analysis.yaml
"""

import os
import sys
import argparse
from argparse import Namespace
from copy import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from cub.config import BASE_DIR
from cub.dataset import CUBDataset
from utils_protocbm.train_utils import gather_args, create_model, prepare_model
from utils_protocbm.eval_utils import get_eval_transform_for_model
from utils_protocbm.index_translation import get_attribute_names
from utils_protocbm.mappings import CBM_SELECTED_CUB_ATTRIBUTE_IDS


# ── Attribute helpers ──────────────────────────────────────────────────────────

def get_cbm_attribute_names():
    """Return list of 112 CBM-selected attribute names."""
    attr_file = os.path.join(BASE_DIR, "data/CUB_200_2011/attributes/attributes.txt")
    with open(attr_file, "r") as f:
        attributes = f.readlines()
    used = [attributes[m] for m in CBM_SELECTED_CUB_ATTRIBUTE_IDS]
    return [a.strip().split(" ", 1)[1] for a in used]


def group_attributes(attr_names):
    """Group attribute indices by their prefix (e.g. 'has_wing_color')."""
    groups = {}
    for idx, name in enumerate(attr_names):
        prefix = name.split("::")[0] if "::" in name else name
        groups.setdefault(prefix, []).append((idx, name))
    return groups


BODY_PART_KEYWORDS = {
    "bill": ["bill_shape", "bill_color", "bill_length"],
    "wing": ["wing_color", "wing_shape", "wing_pattern"],
    "head": ["head_pattern", "crown_color", "forehead_color", "nape_color",
             "eye_color"],
    "breast": ["breast_color", "breast_pattern"],
    "belly": ["belly_color", "belly_pattern"],
    "back": ["back_color", "back_pattern"],
    "tail": ["tail_shape", "tail_pattern", "upper_tail_color",
             "under_tail_color"],
    "leg": ["leg_color"],
    "throat": ["throat_color"],
    "upperparts": ["upperparts_color"],
    "underparts": ["underparts_color"],
    "primary_color": ["primary_color"],
    "size": ["size"],
    "shape": ["shape"],
}


def assign_body_part(attr_name):
    """Map an attribute name to a body part category."""
    lower = attr_name.lower()
    for part, keywords in BODY_PART_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return part
    return "other"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_from_spec(spec):
    """
    Load a model given a specification dict with keys:
      model_name, concept_mapper, mode, checkpoint, proto_n_vectors,
      (optional) concept_activation, use_sigmoid
    """
    args = Namespace(
        model_name=spec["model_name"],
        concept_mapper=spec["concept_mapper"],
        mode=spec["mode"],
        checkpoint=spec["checkpoint"],
        n_attributes=112,
        backbone="inception",
        backbone_pretrained=True,
        backbone_freeze=False,
        expand_dim=0,
        use_aux=False,
        proto_n_vectors=spec.get("proto_n_vectors", 1),
        concept_activation=spec.get("concept_activation", "none"),
        use_attr=True,
        no_img=False,
        bottleneck=False,
        dataset="cub",
        saliency_method="attention",
    )
    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
    model.eval()
    return model, device, args


def extract_scores(model, images, attr_labels, model_name):
    """
    Run model forward pass and return concept scores [B, 112] as raw logits.
    """
    outputs = model(images, attr_labels)

    if isinstance(outputs, tuple) and len(outputs) == 3:
        # ProtoMod: (class_logits, similarity_scores, attention_maps)
        _, sim_scores, _ = outputs
        return sim_scores
    elif isinstance(outputs, list) and len(outputs) > 1:
        # CBM: [class_logits, attr0, attr1, ..., attr111] or [attr0, ..., attr111]
        if outputs[0].shape[-1] == 1:
            attr_outputs = outputs
        else:
            attr_outputs = outputs[1:]
        return torch.cat(attr_outputs, dim=1)
    else:
        raise ValueError(f"Unexpected output format from {model_name}: {type(outputs)}")


# ── Data loading ──────────────────────────────────────────────────────────────

def get_test_loader(model, batch_size=64):
    """Create a DataLoader for the CUB test set."""
    args_for_transform = Namespace(
        model_name="protocbm",
        backbone="inception",
    )
    transform = get_eval_transform_for_model(model, args_for_transform)[0]
    pkl_path = os.path.join(BASE_DIR, "data/CUB_processed/class_attr_data_10/test.pkl")
    image_dir = os.path.join(BASE_DIR, "data/CUB_200_2011")
    dataset = CUBDataset([pkl_path], image_dir, transform, "cub")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader


# ── Collect scores from all models ───────────────────────────────────────────

MODEL_SPECS = {
    "ProtoCBM": {
        "model_name": "protocbm",
        "concept_mapper": "protomod",
        "mode": "XC",
        "checkpoint": "weights/protoCBM-models/independent/xc_sigmoid.pth",
        "proto_n_vectors": 1,
    },
    "CBM": {
        "model_name": "cbm",
        "concept_mapper": "cbm",
        "mode": "XCY",
        "checkpoint": "weights/baseline_CUB/concept-mapper-xc/best_model_1-xc.pth",
        "proto_n_vectors": 1,
        "concept_activation": "none",
    },
    "Oracle-ProtoCBM": {
        "model_name": "protocbm",
        "concept_mapper": "protomod",
        "mode": "XCY",
        "checkpoint": "outputs/retrain_run25_variance/seed_2/best_model_2.pth",
        "proto_n_vectors": 1,
    },
}


def collect_all_scores(model_specs, batch_size=64, max_batches=None):
    """
    Load each model and collect sigmoid probabilities + GT labels over the test set.

    Returns:
        scores_dict: {model_name: np.ndarray [N, 112]} of sigmoid probabilities
        gt_labels:   np.ndarray [N, 112] of ground truth
    """
    scores_dict = {}
    gt_labels = None

    for name, spec in model_specs.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {name}")
        print(f"{'='*60}")

        model, device, model_args = load_model_from_spec(spec)
        loader = get_test_loader(model, batch_size)

        all_scores = []
        all_gt = []

        with torch.no_grad():
            for batch_idx, (images, class_labels, attr_labels) in enumerate(
                tqdm(loader, desc=f"Scoring {name}")
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(device)
                attr_labels_tensor = torch.stack(attr_labels, dim=1).float().to(device)

                raw_scores = extract_scores(model, images, attr_labels_tensor, name)
                probs = torch.sigmoid(raw_scores)

                all_scores.append(probs.cpu().numpy())
                all_gt.append(attr_labels_tensor.cpu().numpy())

        scores_dict[name] = np.concatenate(all_scores, axis=0)
        if gt_labels is None:
            gt_labels = np.concatenate(all_gt, axis=0)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    return scores_dict, gt_labels


# ── Plot 1: Concept group detail ─────────────────────────────────────────────

def plot_concept_group_detail(scores_dict, gt_labels, attr_names, group_prefix, save_dir):
    """
    For a concept group (e.g. 'has_wing_color'), create a figure with
    3 rows (models) x K columns (options).
    Each subplot shows a violin/histogram of sigmoid scores for that concept option.
    """
    groups = group_attributes(attr_names)

    if group_prefix not in groups:
        print(f"Warning: group '{group_prefix}' not found. Available: {list(groups.keys())}")
        return

    members = groups[group_prefix]
    n_options = len(members)
    model_names = list(scores_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(
        n_models, n_options,
        figsize=(max(2.5 * n_options, 10), 3 * n_models),
        squeeze=False,
        sharey=True,
    )

    colors_pos = ["#2ecc71", "#3498db", "#e74c3c"]
    colors_neg = ["#a8e6cf", "#a8d8ea", "#f5b7b1"]

    for row, model_name in enumerate(model_names):
        for col, (attr_idx, attr_name) in enumerate(members):
            ax = axes[row, col]
            option_label = attr_name.split("::")[-1] if "::" in attr_name else attr_name

            probs = scores_dict[model_name][:, attr_idx]
            gt = gt_labels[:, attr_idx].astype(bool)

            # Split by GT label
            probs_pos = probs[gt]
            probs_neg = probs[~gt]

            # Overlaid histograms
            bins = np.linspace(0, 1, 30)
            if len(probs_neg) > 0:
                ax.hist(probs_neg, bins=bins, alpha=0.5, color=colors_neg[row],
                        label=f"GT=0 (n={len(probs_neg)})", density=True)
            if len(probs_pos) > 0:
                ax.hist(probs_pos, bins=bins, alpha=0.7, color=colors_pos[row],
                        label=f"GT=1 (n={len(probs_pos)})", density=True)

            ax.axvline(0.5, color="gray", ls="--", lw=0.8)

            if row == 0:
                ax.set_title(option_label, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model_name, fontsize=10, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("P(concept)", fontsize=8)

            ax.legend(fontsize=5, loc="upper center")
            ax.set_xlim(0, 1)

    fig.suptitle(
        f"Concept Activation Distributions: {group_prefix}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    safe_name = group_prefix.replace("::", "_").replace(" ", "_")
    path = os.path.join(save_dir, f"concept_group_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 2: Global bar plot ──────────────────────────────────────────────────

def plot_global_concept_bars(scores_dict, gt_labels, attr_names, save_dir):
    """
    Bar plot of mean predicted probability per concept.
    One row per model (stacked vertically), each with GT prevalence overlay.
    """
    n_attrs = len(attr_names)
    model_names = list(scores_dict.keys())
    n_models = len(model_names)

    gt_mean = gt_labels.mean(axis=0)
    model_means = {name: scores_dict[name].mean(axis=0) for name in model_names}

    # Sort by GT prevalence for readability
    sort_idx = np.argsort(gt_mean)[::-1]
    short_names = [attr_names[j].split("::")[-1] if "::" in attr_names[j] else attr_names[j]
                   for j in sort_idx]

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    x = np.arange(n_attrs)

    fig, axes = plt.subplots(n_models, 1, figsize=(28, 5 * n_models), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for row, name in enumerate(model_names):
        ax = axes[row]
        ax.bar(x - 0.2, gt_mean[sort_idx], 0.4,
               label="GT prevalence", color="#bdc3c7", alpha=0.7, edgecolor="none")
        ax.bar(x + 0.2, model_means[name][sort_idx], 0.4,
               label=name, color=colors[row], alpha=0.8, edgecolor="none")
        ax.set_ylabel("Mean Prob.", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(-1, n_attrs)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short_names, rotation=90, fontsize=5, ha="center")

    fig.suptitle("Mean Concept Activation per Attribute (sorted by GT prevalence)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "global_concept_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    # Also create a version grouped by concept prefix
    groups = group_attributes(attr_names)
    group_names_sorted = sorted(groups.keys())
    group_means_gt = []
    group_means_models = {name: [] for name in model_names}

    for gname in group_names_sorted:
        idxs = [idx for idx, _ in groups[gname]]
        group_means_gt.append(gt_mean[idxs].mean())
        for name in model_names:
            group_means_models[name].append(model_means[name][idxs].mean())

    x2 = np.arange(len(group_names_sorted))

    fig2, axes2 = plt.subplots(n_models, 1, figsize=(20, 4 * n_models), sharex=True, sharey=True)
    if n_models == 1:
        axes2 = [axes2]

    for row, name in enumerate(model_names):
        ax = axes2[row]
        ax.bar(x2 - 0.2, group_means_gt, 0.4,
               label="GT prevalence", color="#bdc3c7", alpha=0.7)
        ax.bar(x2 + 0.2, group_means_models[name], 0.4,
               label=name, color=colors[row], alpha=0.8)
        ax.set_ylabel("Mean Prob.", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")

    axes2[-1].set_xticks(x2)
    axes2[-1].set_xticklabels(group_names_sorted, rotation=45, fontsize=8, ha="right")

    fig2.suptitle("Mean Concept Activation per Concept Group",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    path2 = os.path.join(save_dir, "global_concept_bars_by_group.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close(fig2)


# ── Plot 3: Per body-part bar plot ───────────────────────────────────────────

def plot_body_part_bars(scores_dict, gt_labels, attr_names, save_dir):
    """
    Aggregate concept activations by body part.
    Two columns (activation + bias), one row per model (stacked vertically).
    """
    model_names = list(scores_dict.keys())
    n_models = len(model_names)

    # Assign attributes to body parts
    part_to_indices = {}
    for idx, name in enumerate(attr_names):
        part = assign_body_part(name)
        part_to_indices.setdefault(part, []).append(idx)

    part_names = sorted(part_to_indices.keys())
    n_parts = len(part_names)

    gt_mean = gt_labels.mean(axis=0)
    model_means = {name: scores_dict[name].mean(axis=0) for name in model_names}

    part_gt = [gt_mean[part_to_indices[p]].mean() for p in part_names]

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    x = np.arange(n_parts)

    fig, axes = plt.subplots(n_models, 2, figsize=(18, 4 * n_models), squeeze=False,
                             sharex=True)

    for row, name in enumerate(model_names):
        part_model = [model_means[name][part_to_indices[p]].mean() for p in part_names]
        part_bias = [model_means[name][part_to_indices[p]].mean() - gt_mean[part_to_indices[p]].mean()
                     for p in part_names]

        # Left column: activation
        ax_act = axes[row, 0]
        ax_act.bar(x - 0.2, part_gt, 0.4,
                   label="GT prevalence", color="#bdc3c7", alpha=0.7)
        ax_act.bar(x + 0.2, part_model, 0.4,
                   label=name, color=colors[row], alpha=0.8)
        ax_act.set_ylabel("Mean Prob.", fontsize=9)
        ax_act.set_ylim(0, 1)
        ax_act.legend(fontsize=8, loc="upper right")
        ax_act.set_title(f"{name} — Activation", fontsize=10, fontweight="bold", loc="left")

        # Right column: bias
        ax_bias = axes[row, 1]
        ax_bias.bar(x, part_bias, 0.6, color=colors[row], alpha=0.8, label=name)
        ax_bias.axhline(0, color="black", lw=0.5)
        ax_bias.set_ylabel("Pred - GT", fontsize=9)
        ax_bias.legend(fontsize=8, loc="upper right")
        ax_bias.set_title(f"{name} — Bias", fontsize=10, fontweight="bold", loc="left")

    for col in range(2):
        axes[-1, col].set_xticks(x)
        axes[-1, col].set_xticklabels(part_names, rotation=45, ha="right", fontsize=9)

    fig.suptitle("Body Part Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "body_part_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Concept Activation Analysis")
    parser.add_argument("--output_dir", type=str, default="outputs/concept_analysis",
                        help="Directory to save plots")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches for quick testing")
    parser.add_argument("--concept_groups", type=str, nargs="+",
                        default=["has_wing_color", "has_breast_color", "has_bill_shape",
                                 "has_back_color", "has_head_pattern"],
                        help="Concept group prefixes to plot detail views for")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    attr_names = get_cbm_attribute_names()
    print(f"Loaded {len(attr_names)} attribute names")

    # Print available concept groups
    groups = group_attributes(attr_names)
    print(f"\nAvailable concept groups ({len(groups)}):")
    for gname, members in sorted(groups.items()):
        print(f"  {gname}: {len(members)} options")

    # Collect scores from all models
    scores_dict, gt_labels = collect_all_scores(
        MODEL_SPECS, batch_size=args.batch_size, max_batches=args.max_batches
    )
    print(f"\nCollected scores: {list(scores_dict.keys())}")
    print(f"Scores shape per model: {next(iter(scores_dict.values())).shape}")
    print(f"GT shape: {gt_labels.shape}")

    # Save raw data for later use
    np.savez(
        os.path.join(args.output_dir, "concept_scores.npz"),
        gt_labels=gt_labels,
        attr_names=attr_names,
        **{f"scores_{name}": scores for name, scores in scores_dict.items()},
    )
    print(f"Saved raw scores to {args.output_dir}/concept_scores.npz")

    # Generate plots
    print("\n--- Generating concept group detail plots ---")
    for group_prefix in args.concept_groups:
        plot_concept_group_detail(scores_dict, gt_labels, attr_names, group_prefix, args.output_dir)

    print("\n--- Generating global bar plot ---")
    plot_global_concept_bars(scores_dict, gt_labels, attr_names, args.output_dir)

    print("\n--- Generating body part analysis ---")
    plot_body_part_bars(scores_dict, gt_labels, attr_names, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
