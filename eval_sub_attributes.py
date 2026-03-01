"""
Evaluate trained models on the SUB dataset for ATTRIBUTE prediction accuracy.
This is adapted from: https://github.com/ExplainableML/sub/blob/main/CBM_testing/test_ind_cbm_example.py

Config Options:
    use_majority_voting: bool (default: False)
        If True, applies majority voting to denoise the ground-truth attributes.
        For each class, if >50% of samples have an attribute, all samples get it;
        if <=50% have it, the attribute is removed for that class.

    save_majority_csv: bool (default: False)
        If True and use_majority_voting is True, saves the majority-voted
        attributes to a CSV file at {log_dir}/eval_sub/majority_voted_attributes.csv
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cub.dataset import SUBDataset
from cub.config import BASE_DIR, N_ATTRIBUTES_CBM
from utils_protocbm.train_utils import gather_args
from utils_protocbm.eval_utils import get_eval_transform_for_model, create_model_for_eval
from utils_protocbm.mappings import CBM_SELECTED_CUB_ATTRIBUTE_IDS
from torch.utils.data import DataLoader


def get_cbm_attribute_names():
    """
    Get the names of the 112 CBM-selected attributes.
    Returns list of attribute names indexed by CBM attribute index.
    """
    attr_file = os.path.join(BASE_DIR, "data/CUB_200_2011/attributes/attributes.txt")

    # Load all CUB attributes (1-indexed in file)
    idx_to_attr = {}
    with open(attr_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            idx = int(parts[0]) - 1  # Convert to 0-indexed
            name = parts[1]
            idx_to_attr[idx] = name

    # Get CBM-selected attributes in order
    cbm_attr_names = [
        idx_to_attr[cub_idx] for cub_idx in CBM_SELECTED_CUB_ATTRIBUTE_IDS
    ]
    return cbm_attr_names


def create_sub_to_cbm_mapping():
    """
    Create mapping from SUB attribute names to CBM attribute indices.

    SUB uses format: "has_breast_color--red"
    CUB uses format: "has_breast_color::red"

    Returns:
        sub_attr_to_cbm_idx: Dict mapping SUB attr name -> CBM index (0-111)
        cbm_attr_names: List of CBM attribute names
    """
    cbm_attr_names = get_cbm_attribute_names()

    # Create reverse lookup: CUB attr name -> CBM index
    cub_name_to_cbm_idx = {name: i for i, name in enumerate(cbm_attr_names)}

    # SUB uses '--' instead of '::'
    sub_attr_to_cbm_idx = {}
    for cub_name, cbm_idx in cub_name_to_cbm_idx.items():
        sub_name = cub_name.replace("::", "--")
        sub_attr_to_cbm_idx[sub_name] = cbm_idx

    return sub_attr_to_cbm_idx, cbm_attr_names


def get_overlapping_cbm_indices(sub_attr_to_cbm_idx, dataset_attr_names):
    """
    Identify the CBM attribute indices that overlap between SUB and CUB.

    These are the 17 SUB attributes that also exist in the 112 CBM-selected attributes.
    sub_attr_to_cbm_idx maps ALL 112 CBM names (in SUB format) to CBM indices,
    so we intersect with the actual SUB dataset attribute names to find the true overlap.

    Args:
        sub_attr_to_cbm_idx: Dict mapping CBM attr name (in SUB format) -> CBM index
        dataset_attr_names: List of actual SUB dataset attribute names

    Returns:
        overlapping_cbm_indices: Sorted list of CBM indices (0-111) that overlap
    """
    sub_names = set(dataset_attr_names)
    overlapping_cbm_indices = set()

    for sub_name, cbm_idx in sub_attr_to_cbm_idx.items():
        if sub_name in sub_names:
            overlapping_cbm_indices.add(cbm_idx)

    return sorted(overlapping_cbm_indices)


def normalize_sub_bird_name(name):
    """
    Normalize bird name for matching between SUB and CUB datasets.
    SUB uses names like "White_breasted_Nuthatch" or "White breasted Nuthatch"
    CUB folder names use "White_breasted_Nuthatch"
    """
    # Replace spaces with underscores and convert to lowercase for comparison
    return name.replace(" ", "_").lower()


def build_bird_class_attribute_cache(img_dir, use_majority_voting=False):
    """
    Build a cache mapping bird class names to their original CBM attribute labels.

    Loads per-image binary attribute labels from image_attribute_labels.txt
    (full CUB dataset, all 11,788 images) and aggregates them per class.

    When use_majority_voting is True, applies a >50% threshold to produce binary
    attributes per class. When False, keeps the raw proportions (0-1).

    Args:
        img_dir: Path to CUB images directory (for class name mapping)
        use_majority_voting: If True, apply majority voting to denoise attributes.
            For each class, if >50% of samples have an attribute, all samples get it;
            if <=50% have it, the attribute is removed for that class.

    Returns:
        bird_to_attrs: Dict mapping normalized bird name -> list of 112 attribute values
            (binary 0/1 if majority_voting, continuous 0-1 proportions otherwise)
    """
    # Build mapping from class index to bird name
    class_idx_to_bird_name = {}
    folders = os.listdir(img_dir)
    for folder in folders:
        parts = folder.split(".")
        if len(parts) >= 2:
            class_idx = int(parts[0]) - 1  # Convert to 0-indexed
            bird_name = parts[1]  # e.g., "White_breasted_Nuthatch"
            class_idx_to_bird_name[class_idx] = bird_name

    # Both modes load from image_attribute_labels.txt (full dataset)
    attr_labels_path = os.path.join(
        BASE_DIR, "data/CUB_200_2011/attributes/image_attribute_labels.txt"
    )
    class_labels_path = os.path.join(
        BASE_DIR, "data/CUB_200_2011/image_class_labels.txt"
    )
    print(f"Loading per-image attribute labels from: {attr_labels_path}")

    # Build image_id -> class_idx mapping (convert to 0-indexed)
    image_to_class = {}
    with open(class_labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            class_id = int(parts[1]) - 1  # Convert to 0-indexed
            image_to_class[image_id] = class_id

    # Aggregate: per class, count how many images have each attribute present
    n_attrs = 312
    class_attr_sums = defaultdict(lambda: [0] * n_attrs)
    class_image_ids = defaultdict(set)

    with open(attr_labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            attr_id = int(parts[1]) - 1  # Convert to 0-indexed
            is_present = int(parts[2])

            class_idx = image_to_class.get(image_id)
            if class_idx is None:
                continue

            class_image_ids[class_idx].add(image_id)
            class_attr_sums[class_idx][attr_id] += is_present

    if use_majority_voting:
        # Apply majority voting on full dataset: >50% -> 1, <=50% -> 0
        bird_to_attrs = {}
        for class_idx, bird_name in class_idx_to_bird_name.items():
            if class_idx not in class_image_ids:
                continue
            total = len(class_image_ids[class_idx])
            if total == 0:
                continue
            majority_attrs = [
                1 if class_attr_sums[class_idx][cub_idx] / total > 0.5 else 0
                for cub_idx in CBM_SELECTED_CUB_ATTRIBUTE_IDS
            ]
            normalized_name = normalize_sub_bird_name(bird_name)
            bird_to_attrs[normalized_name] = majority_attrs
    else:
        # Convert to CBM-selected attribute proportions (continuous 0-1)
        bird_to_attrs = {}
        for class_idx, bird_name in class_idx_to_bird_name.items():
            if class_idx not in class_image_ids:
                continue
            total = len(class_image_ids[class_idx])
            if total == 0:
                continue
            full_proportions = [
                class_attr_sums[class_idx][a] / total for a in range(n_attrs)
            ]
            cbm_attrs = [full_proportions[cub_idx] for cub_idx in CBM_SELECTED_CUB_ATTRIBUTE_IDS]
            normalized_name = normalize_sub_bird_name(bird_name)
            bird_to_attrs[normalized_name] = cbm_attrs

    return bird_to_attrs


def save_majority_voted_attributes_csv(bird_to_attrs, cbm_attr_names, output_path):
    """
    Save majority-voted attributes to a CSV file.

    Args:
        bird_to_attrs: Dict mapping normalized bird name -> list of binary attribute labels
        cbm_attr_names: List of CBM attribute names
        output_path: Path to save the CSV file
    """
    rows = []
    for bird_name, attrs in sorted(bird_to_attrs.items()):
        row = {"bird_class": bird_name}
        for i, attr_name in enumerate(cbm_attr_names):
            row[attr_name] = attrs[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved majority-voted attributes to: {output_path}")
    return df


def find_original_attribute_indices(
    bird_name, changed_attr_sub_name, cbm_attr_names, bird_to_attrs,
    use_majority_voting=False,
):
    """
    Find which original attributes were replaced by the changed attribute.

    For example, if a Cardinal normally has "has_breast_color::red" and SUB changed it to
    "has_breast_color::blue", this function returns the CBM indices for all originally
    active attributes of that type.

    When use_majority_voting=True, attributes are binary (0/1), so all non-zero
    candidates are returned. When False, attributes are continuous proportions
    (0-1), so the single attribute with the highest proportion in the group is
    returned (the dominant original attribute for that body part).

    Args:
        bird_name: Bird species name (e.g., "Cardinal")
        changed_attr_sub_name: SUB attribute name that was changed (e.g., "has_breast_color--blue")
        cbm_attr_names: List of CBM attribute names
        bird_to_attrs: Cached dict mapping normalized bird name -> attribute labels/proportions
        use_majority_voting: If True, attributes are binary; if False, continuous proportions.

    Returns:
        List of CBM indices for original attributes that were present before substitution,
        or empty list if none found
    """
    # Get the attribute type (e.g., "has_breast_color" from "has_breast_color--blue")
    attr_type = changed_attr_sub_name.split("--")[0]

    # Find all CBM attributes of the same type
    candidate_indices: list[int] = []
    for cbm_idx, cub_name in enumerate(cbm_attr_names):
        if cub_name.startswith(attr_type + "::"):
            candidate_indices.append(cbm_idx)

    if not candidate_indices:
        return []

    # Get the base attributes for this bird from cache (using normalized name)
    normalized_name = normalize_sub_bird_name(bird_name)
    base_attrs = bird_to_attrs.get(normalized_name)
    if base_attrs is None:
        return []

    if use_majority_voting:
        # Binary attributes: return ALL candidates that are active (non-zero)
        original_indices = [cbm_idx for cbm_idx in candidate_indices if base_attrs[cbm_idx]]
    else:
        # Continuous proportions: return the single attribute with the highest score
        best_idx = max(candidate_indices, key=lambda idx: base_attrs[idx])
        if base_attrs[best_idx] > 0:
            original_indices = [best_idx]
        else:
            original_indices = []

    return original_indices


def get_attribute_predictions(model, inputs, return_features=False, use_sigmoid=False):
    """
    Get attribute predictions from model output.

    Model output formats:
    - ProtoMod: (class_logits, similarity_scores, attention_maps) where sim_scores is [B, 112].
    - CBMMapper: [class_logits, attr1, attr2, ..., attr112] where each attr is [B, 1].

    For models trained with BCE loss, set use_sigmoid=True so that sigmoid is
    applied to raw logits before thresholding. For MSE-trained models (default),
    scores are used directly.

    Args:
        model: The trained model
        inputs: Tuple of input tensors (images, attr_labels)
        return_features: If True, return raw feature activations for old/new attribute comparison
        use_sigmoid: If True, apply sigmoid to model outputs before thresholding

    Returns:
        attr_preds: Tensor of shape [B, 112] with binary predictions
        attr_probs: Tensor of shape [B, 112] with probabilities/scores
        attr_features: (Optional) Tensor of shape [B, 112] with raw feature activations
    """
    outputs = model(*inputs)

    if isinstance(outputs, tuple) and len(outputs) == 3:
        # ProtoMod: (class_logits, similarity_scores, attention_maps)
        _, sim_scores, _ = outputs
        attr_features = sim_scores if return_features else None
        attr_probs = torch.sigmoid(sim_scores) if use_sigmoid else sim_scores

    elif isinstance(outputs, list) and len(outputs) > 1:
        # CBMMapper: [class_logits, attr1, attr2, ..., attrN]
        # CBM always outputs logits, so sigmoid is always applied.
        attr_outputs = outputs[1:]  # Skip class logits
        attr_logits = torch.cat(attr_outputs, dim=1)
        attr_probs = torch.sigmoid(attr_logits)
        attr_features = attr_logits if return_features else None
    else:
        raise ValueError(f"Unexpected model output format: {type(outputs)}")

    attr_preds = (attr_probs >= 0.5).float()

    if return_features:
        return attr_preds, attr_probs, attr_features
    return attr_preds, attr_probs


def eval_sub_attributes(args):
    """
    Evaluate model on SUB dataset for attribute prediction accuracy.

    Tests both:
    1. New attribute detection (should predict substituted attribute as PRESENT)
    2. Original attribute detection (should predict original attribute as ABSENT)
    """
    # Create the model and load weights
    model, device, _ = create_model_for_eval(args)
    model.eval()

    # Get transforms for the model
    transform = get_eval_transform_for_model(model, args)[0]

    # Load SUB dataset
    sub_data_dir = getattr(args, "sub_data_dir", os.path.join(BASE_DIR, "data/SUB"))
    if not os.path.isabs(sub_data_dir):
        sub_data_dir = os.path.join(BASE_DIR, sub_data_dir)
    sub_limit = getattr(args, "sub_limit", None) # Optionally limit number of samples

    print(f"Loading SUB dataset from {sub_data_dir}")
    dataset = SUBDataset(
        sub_data_dir, transform=transform, limit=sub_limit, only_cub_attributes=True
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Bird classes: {len(dataset.bird_names)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create SUB -> CBM attribute mapping
    sub_attr_to_cbm_idx, cbm_attr_names = create_sub_to_cbm_mapping()

    # Image dir for bird class name mapping
    if hasattr(args, "image_dir") and args.image_dir:
        img_dir = os.path.join(BASE_DIR, args.image_dir)
    else:
        img_dir = os.path.join(BASE_DIR, "data/CUB_200_2011/images")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(
            f"Image directory not found at {img_dir}. Check your image_dir config."
        )

    # Check for majority voting flag
    use_majority_voting = getattr(args, "use_majority_voting", False)
    save_majority_csv = getattr(args, "save_majority_csv", False)

    print("Building bird class attribute cache...")
    if use_majority_voting:
        print("Using MAJORITY VOTING to denoise attributes (>50% threshold)")
    else:
        print("Using CUB per-image attribute proportions (image_attribute_labels.txt)")

    bird_to_attrs = build_bird_class_attribute_cache(
        img_dir, use_majority_voting=use_majority_voting
    )
    print(f"Cached attributes for {len(bird_to_attrs)} bird classes")

    # Optionally save majority-voted attributes to CSV
    if use_majority_voting and save_majority_csv:
        csv_path = os.path.join(args.log_dir, "eval_sub", "majority_voted_attributes.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        save_majority_voted_attributes_csv(bird_to_attrs, cbm_attr_names, csv_path)

    # Tracking metrics
    total_samples = 0
    new_attr_correct = 0  # Correctly predicted the NEW attribute as PRESENT
    original_attr_correct = 0  # Correctly predicted the ORIGINAL attribute as ABSENT

    # Per-attribute tracking
    per_attr_new_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_new_total = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_original_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_original_total = torch.zeros(N_ATTRIBUTES_CBM)

    # Track feature differences between old and new attributes
    feature_diff_stats = {
        'l1_distances': [],  # L1 distance between old and new feature activations (all cases)
        'probs_distances': [],  # distances between probabilities
        # Breakdown by prediction scenarios
        'both_present': [],  # new=present, old=present
        'both_absent': [],  # new=absent, old=absent
    }

    # Per-sample activation scores for old vs new attributes
    sample_new_scores = []   # sigmoid activation for new attribute per sample
    sample_old_scores = []   # sigmoid activation for old attribute per sample (avg if multiple)
    old_higher_list = []     # 1 if old activation > new activation, 0 otherwise

    # Track unmatched items for debugging
    unmatched_birds = set()
    unmatched_attrs = set()

    with torch.no_grad():
        for images, bird_labels, attr_labels in tqdm(
            loader, desc="Evaluating SUB Attributes"
        ):
            images = images.to(device)
            attr_labels = attr_labels.float().to(device)

            batch_size = images.size(0)

            # Get attribute predictions and raw features
            use_sigmoid = getattr(args, "use_sigmoid", False)
            attr_preds, attr_probs, attr_features = get_attribute_predictions(
                model, (images, attr_labels), return_features=True, use_sigmoid=use_sigmoid
            )
            attr_preds = attr_preds.cpu()
            attr_probs = attr_probs.cpu()
            attr_features = attr_features.cpu()

            for i in range(batch_size):
                # Get SUB attribute info
                sub_attr_idx = int(attr_labels[i].item())               # index of attr (CUB)
                sub_attr_name = dataset.attr_names[sub_attr_idx]        # name of attr (SUB)
                bird_name = dataset.bird_names[bird_labels[i].item()]   # bird species name (SUB)

                # Map NEW attribute to CBM index
                new_cbm_idx = sub_attr_to_cbm_idx.get(sub_attr_name)    # convert CUB idx to CBM idx
                if new_cbm_idx is None:
                    unmatched_attrs.add(sub_attr_name)
                    continue

                # Find ORIGINAL attributes - skip sample if not found
                # This ensures both metrics are evaluated on the same samples
                original_cbm_indices = find_original_attribute_indices(
                    bird_name, sub_attr_name, cbm_attr_names, bird_to_attrs,
                    use_majority_voting=use_majority_voting,
                )

                if not original_cbm_indices:
                    unmatched_birds.add(bird_name)
                    continue  # Skip this sample entirely

                total_samples += 1
                per_attr_new_total[new_cbm_idx] += 1    # Count for attribute

                # Test 1: Is the NEW (substituted) attribute predicted as PRESENT?
                new_is_correct = attr_preds[i, new_cbm_idx] == 1
                if new_is_correct:
                    new_attr_correct += 1
                    per_attr_new_correct[new_cbm_idx] += 1

                # Test 2: Are ALL ORIGINAL attributes predicted as ABSENT?
                # Note: Birds can have multiple active attributes per body part
                # Check each original attribute - ALL must be predicted as absent
                all_originals_absent = True
                for original_cbm_idx in original_cbm_indices:
                    per_attr_original_total[original_cbm_idx] += 1
                    if attr_preds[i, original_cbm_idx] == 0:
                        per_attr_original_correct[original_cbm_idx] += 1
                    else:
                        all_originals_absent = False

                # Original is "correct" only if ALL original attrs are absent
                if all_originals_absent:
                    original_attr_correct += 1

                # Compute feature differences between old and new attributes
                # Apply sigmoid to get activations in [0, 1] range
                new_feature_raw = attr_features[i, new_cbm_idx]
                new_feature = torch.sigmoid(new_feature_raw)
                new_pred_present = attr_preds[i, new_cbm_idx] == 1
                new_feature_probs = attr_probs[i, new_cbm_idx]

                for original_cbm_idx in original_cbm_indices:
                    old_feature_raw = attr_features[i, original_cbm_idx]
                    old_feature = torch.sigmoid(old_feature_raw)
                    old_pred_present = attr_preds[i, original_cbm_idx] == 1
                    old_feature_probs = attr_probs[i, original_cbm_idx]

                    # L1 distance (for all cases) - now between sigmoid activations
                    l1_dist = torch.abs(new_feature - old_feature).item()
                    feature_diff_stats['l1_distances'].append(l1_dist)

                    # Distance between probs
                    feature_diff_stats['probs_distances'].append(new_feature_probs - old_feature_probs)

                    # Categorize by prediction scenario
                    if new_pred_present and old_pred_present:
                        # Both predicted as present
                        feature_diff_stats['both_present'].append(l1_dist)
                    elif not new_pred_present and not old_pred_present:
                        # Both predicted as absent
                        feature_diff_stats['both_absent'].append(l1_dist)

                # Store per-sample scores: new vs old (average over original indices)
                new_score = new_feature.item()
                old_scores = [
                    torch.sigmoid(attr_features[i, idx]).item()
                    for idx in original_cbm_indices
                ]
                old_score = np.mean(old_scores)
                sample_new_scores.append(new_score)
                sample_old_scores.append(old_score)
                old_higher_list.append(1 if old_score > new_score else 0)


    return {
        "total_samples": total_samples,
        "new_attr_correct": new_attr_correct,
        "original_attr_correct": original_attr_correct,
        "per_attr_new_correct": per_attr_new_correct,
        "per_attr_new_total": per_attr_new_total,
        "per_attr_original_correct": per_attr_original_correct,
        "per_attr_original_total": per_attr_original_total,
        "cbm_attr_names": cbm_attr_names,
        "unmatched_birds": unmatched_birds,
        "unmatched_attrs": unmatched_attrs,
        "use_majority_voting": use_majority_voting,
        "feature_diff_stats": feature_diff_stats,
        "sample_new_scores": sample_new_scores,
        "sample_old_scores": sample_old_scores,
        "old_higher_list": old_higher_list,
    }


def plot_old_vs_new_scores(results, out_dir):
    """
    Plot old vs new attribute activation scores and save the figure.

    Creates a scatter plot where each point is a sample:
      x-axis = new (substituted) attribute sigmoid activation
      y-axis = old (original) attribute sigmoid activation
    Points are colored by whether old > new (orange) or not (blue).
    A diagonal line marks where old == new.

    Also saves the old_higher_list to a text file.

    Args:
        results: Dict returned by eval_sub_attributes
        out_dir: Directory to save the plot and list
    """
    new_scores = np.array(results["sample_new_scores"])
    old_scores = np.array(results["sample_old_scores"])
    old_higher = np.array(results["old_higher_list"])

    if len(new_scores) == 0:
        print("No samples to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    mask_old = old_higher == 1
    mask_new = ~mask_old

    ax.scatter(
        new_scores[mask_new], old_scores[mask_new],
        c="tab:blue", alpha=0.4, s=12, label=f"New >= Old ({mask_new.sum()})",
    )
    ax.scatter(
        new_scores[mask_old], old_scores[mask_old],
        c="tab:orange", alpha=0.4, s=12, label=f"Old > New ({mask_old.sum()})",
    )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)

    ax.set_xlabel("New attribute activation (sigmoid)")
    ax.set_ylabel("Old attribute activation (sigmoid)")
    ax.set_title("Old vs New Attribute Activations per Sample")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    plot_path = os.path.join(out_dir, "old_vs_new_activations.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved activation scatter plot to: {plot_path}")

    # --- Save old_higher_list ---
    list_path = os.path.join(out_dir, "old_higher_list.txt")
    with open(list_path, "w") as f:
        f.write(" ".join(str(v) for v in old_higher.tolist()) + "\n")
    print(f"Saved old_higher_list ({old_higher.sum()}/{len(old_higher)} old>new) to: {list_path}")

    # --- Print summary ---
    pct_old = 100 * old_higher.sum() / len(old_higher) if len(old_higher) > 0 else 0
    print(f"Old higher than new: {old_higher.sum()}/{len(old_higher)} ({pct_old:.1f}%)")


def print_results(results):
    """Print formatted evaluation results."""
    total = results["total_samples"]
    new_correct = results["new_attr_correct"]
    original_correct = results["original_attr_correct"]
    unmatched_birds = results["unmatched_birds"]
    unmatched_attrs = results["unmatched_attrs"]
    use_majority_voting = results.get("use_majority_voting", False)
    feature_diff_stats = results.get("feature_diff_stats", {})

    print("\n" + "=" * 70)
    print("SUB BENCHMARK RESULTS")
    print("=" * 70)

    # Show attribute mode
    if use_majority_voting:
        print("\nAttribute Mode: MAJORITY VOTING (denoised)")
        print("  - Attributes with >50% presence in class are set to 1")
        print("  - Attributes with <=50% presence in class are set to 0")
    else:
        print("\nAttribute Mode: CUB PER-IMAGE PROPORTIONS (image_attribute_labels.txt)")
        print("  - Aggregates per-image binary labels across all images per class")
        print("  - Original attribute = highest proportion in body-part group")

    print(f"\nTotal samples evaluated: {total}")
    print("(Only samples where both new and original attributes are in CBM-112)")

    # Warnings for unmatched items
    if unmatched_attrs:
        print(
            f"\nNOTE: {len(unmatched_attrs)} SUB attributes could not be mapped to CBM (skipped):"
        )
        for attr in sorted(unmatched_attrs)[:5]:  # Show first 5
            print(f"   - {attr}")
        if len(unmatched_attrs) > 5:
            print(f"   ... and {len(unmatched_attrs) - 5} more")

    if unmatched_birds:
        print(
            f"\nNOTE: {len(unmatched_birds)} bird classes had no original attribute in CBM-112 (skipped):"
        )
        for bird in sorted(unmatched_birds)[:5]:  # Show first 5
            print(f"   - {bird}")
        if len(unmatched_birds) > 5:
            print(f"   ... and {len(unmatched_birds) - 5} more")

    # Main metrics
    print("\n" + "-" * 70)
    print("MAIN METRICS")
    print("-" * 70)

    new_acc = 100 * new_correct / total if total > 0 else 0
    print("\n1. New Attribute Detection (should predict substituted attr as PRESENT):")
    print(f"   Accuracy: {new_acc:.2f}% ({new_correct}/{total})")

    original_present = total - original_correct
    original_present_rate = 100 * original_present / total if total > 0 else 0
    print("\n2. Original Attribute Hallucination (predicts removed attr as PRESENT):")
    print(f"   Rate: {original_present_rate:.2f}% ({original_present}/{total})")

    # Ratio of new detection vs old hallucination
    print("\n3. New Detection / Old Hallucination Ratio:")
    if original_present > 0:
        ratio = new_correct / original_present
        print(f"   Ratio: {ratio:.2f} ({new_correct} new detected / {original_present} old hallucinated)")
    else:
        print(f"   Ratio: inf ({new_correct} new detected / 0 old hallucinated)")
    print("   Interpretation: >1 means model detects new attrs more than it hallucinates old ones")

    # Main metrics based on higher scores: If the attribute score of the new metric is higher than that of the old, we classify it
    # as new predicted correctly, otherwise, we classify it as original detected.
    if feature_diff_stats and len(feature_diff_stats.get('probs_distances', [])) > 0:
        tmp_total = len(feature_diff_stats['probs_distances'])
        diff_based_new_predicted = 0
        diff_based_original_present = 0
        for diff in feature_diff_stats['probs_distances']:
            if diff > 0:
                diff_based_new_predicted += 1
            else:
                diff_based_original_present += 1

        diff_based_new_acc = 100 * diff_based_new_predicted / tmp_total if tmp_total > 0 else 0
        print("\n1. Difference-based New Attribute Detection (should predict substituted attr as PRESENT):")
        print(f"   Accuracy: {diff_based_new_acc:.2f}% ({diff_based_new_predicted}/{tmp_total})")

        diff_based_original_predicted = 100 * diff_based_original_present / tmp_total if tmp_total > 0 else 0
        print("\n2. Difference-based Original Attribute Hallucination (predicts removed attr as PRESENT):")
        print(f"   Rate: {diff_based_original_predicted:.2f}% ({diff_based_original_present}/{tmp_total})")

    # Feature difference statistics
    if feature_diff_stats and len(feature_diff_stats.get('l1_distances', [])) > 0:
        print("\n" + "-" * 70)
        print("FEATURE DIFFERENCE ANALYSIS (Old vs New Attributes)")
        print("-" * 70)

        l1_distances = feature_diff_stats['l1_distances']

        print(f"\n   Total comparisons: {len(l1_distances)}")
        print("\n   Overall L1 Distance (|sigmoid(old) - sigmoid(new)|):")
        print(f"      Mean:   {np.mean(l1_distances):.4f}")
        print(f"      Median: {np.median(l1_distances):.4f}")
        print(f"      Std:    {np.std(l1_distances):.4f}")
        print(f"      Min:    {np.min(l1_distances):.4f}")
        print(f"      Max:    {np.max(l1_distances):.4f}")
        print("      Note: Distances computed between sigmoid-activated features [0, 1]")

        # Breakdown by prediction scenario
        print("\n   Feature Distance by Prediction Scenario:")
        print(f"   {'Scenario':<30} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10}")
        print("   " + "-" * 70)

        scenarios = [
            ('both_present', 'Both present (new=1, old=1)'),
            ('both_absent', 'Both absent (new=0, old=0)'),
        ]

        for key, label in scenarios:
            distances = feature_diff_stats[key]
            if len(distances) > 0:
                print(f"   {label:<30} {len(distances):<10} {np.mean(distances):<10.4f} "
                      f"{np.median(distances):<10.4f} {np.std(distances):<10.4f}")
            else:
                print(f"   {label:<30} {0:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

        print("\n   Interpretation:")
        print("   - Higher distances = larger changes in model's internal representations")
        print("   - 'Both present/absent': Model predicts both attributes the same way")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = gather_args()

    # Print everything into separate file
    out_folder_path = os.path.join(args.log_dir, "eval_sub")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    path_to_output_txt = os.path.join(args.out_dir_part_seg, "sub_attribute_eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")

    print("\n" + "=" * 70)
    print("SUB Dataset - Attribute Prediction Evaluation")
    print("=" * 70 + "\n")

    results = eval_sub_attributes(args)

    # Plot old vs new activation scores and save old_higher_list
    # (done before redirecting stdout so print messages go to console)
    plot_old_vs_new_scores(results, out_folder_path)

    sys.stdout = open(path_to_output_txt, 'a')
    print_results(results)
