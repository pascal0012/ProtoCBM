"""
Evaluate trained models on the SUB dataset for ATTRIBUTE prediction accuracy.
This is adapted from: https://github.com/ExplainableML/sub/blob/main/CBM_testing/test_ind_cbm_example.py

The SUB benchmark tests whether concept-based models learn actual visual concepts
or just memorize class-attribute correlations. It does this by showing images where
a bird's attribute has been visually substituted (e.g., a Cardinal with blue breast
instead of red).

The script evaluates:
1. NEW attribute detection: Does the model see the substituted attribute as PRESENT?
   (High accuracy = model adapts to visual evidence)
2. ORIGINAL attribute detection: Does the model see the original attribute as ABSENT?
   (High accuracy = model doesn't hallucinate based on class priors)

A model that truly learns visual concepts should score high on both metrics.
A model that memorizes class-attribute correlations will score low on both.

Config Options:
    use_majority_voting: bool (default: False)
        If True, applies majority voting to denoise the ground-truth attributes.
        For each class, if >50% of samples have an attribute, all samples get it;
        if <=50% have it, the attribute is removed for that class.
        This reduces noise from inconsistent attribute annotations.

    save_majority_csv: bool (default: False)
        If True and use_majority_voting is True, saves the majority-voted
        attributes to a CSV file at {log_dir}/eval_sub/majority_voted_attributes.csv
"""

import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cub.dataset import SUBDataset
from cub.config import BASE_DIR, N_ATTRIBUTES_CBM
from utils_protocbm.train_utils import (
    gather_args,
    prepare_model,
    create_model,
)
from utils_protocbm.eval_utils import get_eval_transform_for_model
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


def normalize_bird_name(name):
    """
    Normalize bird name for matching between SUB and CUB datasets.
    SUB uses names like "White_breasted_Nuthatch" or "White breasted Nuthatch"
    CUB folder names use "White_breasted_Nuthatch"
    """
    # Replace spaces with underscores and convert to lowercase for comparison
    return name.replace(" ", "_").lower()


def build_bird_class_attribute_cache(val_pkl_path, img_dir, use_majority_voting=False):
    """
    Build a cache mapping bird class names to their original CBM attribute labels.

    This is done once at the start to avoid repeated file system lookups.

    Args:
        val_pkl_path: Path to validation pkl file with attribute labels
        img_dir: Path to CUB images directory (for class name mapping)
        use_majority_voting: If True, apply majority voting to denoise attributes.
            For each class, if >50% of samples have an attribute, all samples get it;
            if <=50% have it, the attribute is removed for that class.

    Returns:
        bird_to_attrs: Dict mapping normalized bird name -> list of 112 binary attribute labels
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

    # Load validation data
    data = pickle.load(open(val_pkl_path, "rb"))

    if use_majority_voting:
        # Aggregate attributes per class for majority voting
        class_attr_counts = defaultdict(lambda: {"total": 0, "attr_sums": None})

        for d in data:
            class_idx = d["class_label"]
            attrs = d["attribute_label"]

            if class_attr_counts[class_idx]["attr_sums"] is None:
                class_attr_counts[class_idx]["attr_sums"] = [0] * len(attrs)

            class_attr_counts[class_idx]["total"] += 1
            for i, attr_val in enumerate(attrs):
                class_attr_counts[class_idx]["attr_sums"][i] += attr_val

        # Apply majority voting: >50% -> 1, <=50% -> 0
        bird_to_attrs = {}
        for class_idx, counts in class_attr_counts.items():
            if class_idx in class_idx_to_bird_name:
                bird_name = class_idx_to_bird_name[class_idx]
                normalized_name = normalize_bird_name(bird_name)

                total = counts["total"]
                majority_attrs = [
                    1 if attr_sum / total > 0.5 else 0
                    for attr_sum in counts["attr_sums"]
                ]
                bird_to_attrs[normalized_name] = majority_attrs
    else:
        # Original behavior: use first sample's attributes for each class
        bird_to_attrs = {}
        for d in data:
            class_idx = d["class_label"]
            if class_idx in class_idx_to_bird_name:
                bird_name = class_idx_to_bird_name[class_idx]
                normalized_name = normalize_bird_name(bird_name)
                if normalized_name not in bird_to_attrs:
                    bird_to_attrs[normalized_name] = d["attribute_label"]

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
    bird_name, changed_attr_sub_name, cbm_attr_names, bird_to_attrs
):
    """
    Find which original attributes were replaced by the changed attribute.

    For example, if a Cardinal normally has "has_breast_color::red" and SUB changed it to
    "has_breast_color::blue", this function returns the CBM indices for all originally
    active attributes of that type.

    Note: Birds can have MULTIPLE active attributes per body part (e.g., a bird's back
    might be both brown and buff). This function returns ALL such attributes.

    Args:
        bird_name: Bird species name (e.g., "Cardinal")
        changed_attr_sub_name: SUB attribute name that was changed (e.g., "has_breast_color--blue")
        cbm_attr_names: List of CBM attribute names
        bird_to_attrs: Cached dict mapping normalized bird name -> attribute labels

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
    normalized_name = normalize_bird_name(bird_name)
    base_attrs = bird_to_attrs.get(normalized_name)
    if base_attrs is None:
        return []

    # Find ALL candidate attributes that are present in the base bird
    original_indices = [cbm_idx for cbm_idx in candidate_indices if base_attrs[cbm_idx]]

    return original_indices


def get_attribute_predictions(model, inputs, device, return_features=False):
    """
    Get attribute predictions from model output.

    ProtoCBM models output:
    - CBMMapper: [class_logits, attr1, attr2, ..., attr112] where each attr is [B, 1]
    - ProtoMod: (class_logits, similarity_scores, attention_maps) where sim_scores is [B, 112]

    Args:
        model: The trained model
        inputs: Tuple of input tensors (images, attr_labels)
        device: Device to run on
        return_features: If True, return raw feature activations for old/new attribute comparison

    Returns:
        attr_preds: Tensor of shape [B, 112] with binary predictions
        attr_probs: Tensor of shape [B, 112] with probabilities
        attr_features: (Optional) Tensor of shape [B, 112] with raw feature activations
    """
    outputs = model(*inputs)

    if isinstance(outputs, tuple) and len(outputs) == 3:
        # ProtoMod: (class_logits, similarity_scores, attention_maps)
        _, sim_scores, _ = outputs
        attr_probs = torch.sigmoid(sim_scores)
        attr_features = sim_scores if return_features else None

    elif isinstance(outputs, list) and len(outputs) > 1:
        # CBMMapper: [class_logits, attr1, attr2, ..., attrN]
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
    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
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

    # Paths for getting base attributes
    # Derive val.pkl path from split_dir (which points to test.pkl in the processed dir)
    if hasattr(args, "split_dir") and args.split_dir:
        split_dir_base = os.path.dirname(args.split_dir)
        val_pkl_path = os.path.join(BASE_DIR, split_dir_base, "val.pkl")
    else:
        # Fallback: assume data_dir contains val.pkl (for training configs)
        #! Not the case for our implementation but kept for compatibility
        val_pkl_path = os.path.join(BASE_DIR, args.data_dir, "val.pkl")

    # Image dir for bird class name mapping
    if hasattr(args, "image_dir") and args.image_dir:
        img_dir = os.path.join(BASE_DIR, args.image_dir)
    else:
        img_dir = os.path.join(BASE_DIR, "data/CUB_200_2011/images")

    # Build cache of bird class -> original attributes
    print(f"Loading validation attributes from: {val_pkl_path}")
    print(f"Loading bird class names from: {img_dir}")

    if not os.path.exists(val_pkl_path):
        raise FileNotFoundError(
            f"val.pkl not found at {val_pkl_path}. Check your split_dir or data_dir config."
        )
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
        print("Using ORIGINAL per-sample attributes (no denoising)")

    bird_to_attrs = build_bird_class_attribute_cache(
        val_pkl_path, img_dir, use_majority_voting=use_majority_voting
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

    # Track samples where model gets both right (adapts) vs both wrong (memorizes)
    both_correct = 0    # Adapts: sees new as present AND original as absent
    both_wrong = 0      # Memorizes: sees new as absent AND original as present

    # Track feature differences between old and new attributes
    feature_diff_stats = {
        'l1_distances': [],  # L1 distance between old and new feature activations (all cases)
        # Breakdown by prediction scenarios
        'adapts': [],  # new=present, old=absent (model adapts)
        'memorizes': [],  # new=absent, old=present (model memorizes)
        'both_present': [],  # new=present, old=present
        'both_absent': [],  # new=absent, old=absent
    }

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
            attr_preds, attr_probs, attr_features = get_attribute_predictions(
                model, (images, attr_labels), device, return_features=True
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
                    bird_name, sub_attr_name, cbm_attr_names, bird_to_attrs
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
                any_original_present = False
                for original_cbm_idx in original_cbm_indices:
                    per_attr_original_total[original_cbm_idx] += 1
                    if attr_preds[i, original_cbm_idx] == 0:
                        per_attr_original_correct[original_cbm_idx] += 1
                    else:
                        all_originals_absent = False
                        any_original_present = True

                # Original is "correct" only if ALL original attrs are absent
                if all_originals_absent:
                    original_attr_correct += 1

                # Track adaptation vs memorization
                if new_is_correct and all_originals_absent:
                    both_correct += 1  # Model adapts to visual evidence
                elif not new_is_correct and any_original_present:
                    both_wrong += 1  # Model memorizes class-attribute correlations

                # Compute feature differences between old and new attributes
                # Apply sigmoid to get activations in [0, 1] range
                new_feature_raw = attr_features[i, new_cbm_idx]
                new_feature = torch.sigmoid(new_feature_raw)
                new_pred_present = attr_preds[i, new_cbm_idx] == 1

                for original_cbm_idx in original_cbm_indices:
                    old_feature_raw = attr_features[i, original_cbm_idx]
                    old_feature = torch.sigmoid(old_feature_raw)
                    old_pred_present = attr_preds[i, original_cbm_idx] == 1

                    # L1 distance (for all cases) - now between sigmoid activations
                    l1_dist = torch.abs(new_feature - old_feature).item()
                    feature_diff_stats['l1_distances'].append(l1_dist)

                    # Categorize by prediction scenario
                    if new_pred_present and not old_pred_present:
                        # Model adapts: new=present, old=absent
                        feature_diff_stats['adapts'].append(l1_dist)
                    elif not new_pred_present and old_pred_present:
                        # Model memorizes: new=absent, old=present
                        feature_diff_stats['memorizes'].append(l1_dist)
                    elif new_pred_present and old_pred_present:
                        # Both predicted as present
                        feature_diff_stats['both_present'].append(l1_dist)
                    else:
                        # Both predicted as absent
                        feature_diff_stats['both_absent'].append(l1_dist)


    return {
        "total_samples": total_samples,
        "new_attr_correct": new_attr_correct,
        "original_attr_correct": original_attr_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "per_attr_new_correct": per_attr_new_correct,
        "per_attr_new_total": per_attr_new_total,
        "per_attr_original_correct": per_attr_original_correct,
        "per_attr_original_total": per_attr_original_total,
        "cbm_attr_names": cbm_attr_names,
        "unmatched_birds": unmatched_birds,
        "unmatched_attrs": unmatched_attrs,
        "use_majority_voting": use_majority_voting,
        "feature_diff_stats": feature_diff_stats,
        # Components for visualization
        "model": model,
        "dataset": dataset,
        "device": device,
        "bird_to_attrs": bird_to_attrs,
        "sub_attr_to_cbm_idx": sub_attr_to_cbm_idx,
    }


def visualize_predictions_grid(
    model,
    dataset,
    device,
    cbm_attr_names,
    bird_to_attrs,
    sub_attr_to_cbm_idx,
    output_dir,
    n_images_per_grid=10,
    n_grids=10,
    n_cols=5,
    seed=42,
):
    """
    Create multiple grid visualizations showing model predictions for old vs new attributes.

    Args:
        model: The trained model
        dataset: SUBDataset instance
        device: torch device
        cbm_attr_names: List of CBM attribute names
        bird_to_attrs: Dict mapping bird name -> original attributes
        sub_attr_to_cbm_idx: Dict mapping SUB attr name -> CBM index
        output_dir: Directory to save the visualizations
        n_images_per_grid: Number of images per grid
        n_grids: Number of grids to generate
        n_cols: Number of columns in each grid
        seed: Random seed for reproducibility
    """
    import random

    model.eval()

    # Collect ALL valid samples first
    print("Collecting valid samples for visualization...")
    all_valid_samples = []
    for idx in range(len(dataset)):
        sample = dataset.dataset[idx]
        sub_attr_idx = sample["attr_label"]
        sub_attr_name = dataset.attr_names[sub_attr_idx]
        bird_label = sample["bird_label"]
        bird_name = dataset.bird_names[bird_label]

        # Check if we can map this attribute
        new_cbm_idx = sub_attr_to_cbm_idx.get(sub_attr_name)
        if new_cbm_idx is None:
            continue

        # Find original attributes
        original_cbm_indices = find_original_attribute_indices(
            bird_name, sub_attr_name, cbm_attr_names, bird_to_attrs
        )
        if not original_cbm_indices:
            continue

        all_valid_samples.append(
            {
                "idx": idx,
                "image": sample["image"],
                "bird_name": bird_name,
                "new_attr_name": sub_attr_name.replace("--", "::"),
                "new_cbm_idx": new_cbm_idx,
                "original_cbm_indices": original_cbm_indices,
                "original_attr_names": [
                    cbm_attr_names[i] for i in original_cbm_indices
                ],
            }
        )

    if not all_valid_samples:
        print("No valid samples found for visualization")
        return

    print(f"Found {len(all_valid_samples)} valid samples")

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(all_valid_samples)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate multiple grids
    total_needed = n_images_per_grid * n_grids
    if len(all_valid_samples) < total_needed:
        print(f"Warning: Only {len(all_valid_samples)} samples available, "
              f"but {total_needed} requested. Some grids may have fewer images.")

    for grid_idx in range(n_grids):
        start_idx = grid_idx * n_images_per_grid
        end_idx = min(start_idx + n_images_per_grid, len(all_valid_samples))

        if start_idx >= len(all_valid_samples):
            print(f"No more samples for grid {grid_idx + 1}, stopping.")
            break

        samples = all_valid_samples[start_idx:end_idx]
        n_images = len(samples)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else list(axes)
        else:
            axes = axes.flatten()

        # Hide unused subplots
        for ax in axes[n_images:]:
            ax.axis("off")

        with torch.no_grad():
            for i, sample in enumerate(samples):
                ax = axes[i]

                # Get image and transform
                img = sample["image"]
                if not isinstance(img, np.ndarray):
                    img_display = np.array(img)
                else:
                    img_display = img

                # Get model prediction with features
                img_tensor, _, attr_label_idx = dataset[sample["idx"]]
                img_tensor = img_tensor.unsqueeze(0).to(device)
                # Create dummy attribute labels tensor (not actually used by the model for prediction)
                attr_labels_vis = torch.zeros(1, 112).to(device)
                attr_preds, attr_probs, attr_features = get_attribute_predictions(
                    model, (img_tensor, attr_labels_vis), device, return_features=True
                )
                attr_probs = attr_probs.cpu().squeeze(0)
                attr_features = attr_features.cpu().squeeze(0)

                # Get predictions for new and original attributes
                new_cbm_idx = sample["new_cbm_idx"]
                new_prob = attr_probs[new_cbm_idx].item()
                new_pred = "Present" if new_prob >= 0.5 else "Absent"
                new_feature_raw = attr_features[new_cbm_idx].item()
                new_feature = 1 / (1 + np.exp(-new_feature_raw))  # Apply sigmoid

                # For original, show the first one (typically there's only one)
                orig_cbm_idx = sample["original_cbm_indices"][0]
                orig_prob = attr_probs[orig_cbm_idx].item()
                orig_pred = "Present" if orig_prob >= 0.5 else "Absent"
                orig_feature_raw = attr_features[orig_cbm_idx].item()
                orig_feature = 1 / (1 + np.exp(-orig_feature_raw))  # Apply sigmoid

                # Compute feature difference (between sigmoid activations)
                feature_diff = abs(new_feature - orig_feature)

                # Extract attribute type and values (e.g., "breast_color" and "red" from "has_breast_color::red")
                attr_parts = sample["new_attr_name"].split("::")
                attr_type = attr_parts[0].replace("has_", "").replace("_", " ")
                new_attr_short = attr_parts[-1]
                orig_attr_short = sample["original_attr_names"][0].split("::")[-1]

                # Display image
                ax.imshow(img_display)
                ax.axis("off")

                # Title: bird class + attribute type + value change
                bird_name_display = sample["bird_name"].replace("_", " ")
                ax.set_title(
                    f"{bird_name_display}\n{attr_type}: {orig_attr_short} → {new_attr_short}",
                    fontsize=10,
                    fontweight="bold",
                )

                # Legend showing predictions and feature difference
                legend_text = (
                    f"Old ({orig_attr_short}): {orig_pred} ({orig_prob:.2f})\n"
                    f"New ({new_attr_short}): {new_pred} ({new_prob:.2f})\n"
                    f"Feature Δ: {feature_diff:.2f}"
                )

                # Color code: green if model adapts (new=present, old=absent), red if memorizes
                if new_prob >= 0.5 and orig_prob < 0.5:
                    box_color = "lightgreen"  # Adapts
                elif new_prob < 0.5 and orig_prob >= 0.5:
                    box_color = "lightcoral"  # Memorizes
                else:
                    box_color = "lightyellow"  # Mixed

                ax.text(
                    0.5,
                    -0.05,
                    legend_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="center",
                    bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
                )

        plt.tight_layout(h_pad=3.0)
        output_path = os.path.join(output_dir, f"predictions_grid_{grid_idx + 1:02d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved grid {grid_idx + 1}/{n_grids}: {output_path}")


def print_results(results):
    """Print formatted evaluation results."""
    total = results["total_samples"]
    new_correct = results["new_attr_correct"]
    original_correct = results["original_attr_correct"]
    both_correct = results["both_correct"]
    both_wrong = results["both_wrong"]
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
        print("\nAttribute Mode: ORIGINAL (per-sample, potentially noisy)")

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

    # Adaptation vs Memorization analysis
    print("\n" + "-" * 70)
    print("ADAPTATION vs MEMORIZATION ANALYSIS")
    print("-" * 70)

    if total > 0:
        adapt_rate = 100 * both_correct / total
        memorize_rate = 100 * both_wrong / total
        mixed_rate = 100 * (total - both_correct - both_wrong) / total

        print(
            f"\n   Adapts (new=1, original=0):    {adapt_rate:5.2f}% ({both_correct}/{total})"
        )
        print(
            f"   Memorizes (new=0, original=1): {memorize_rate:5.2f}% ({both_wrong}/{total})"
        )
        print(
            f"   Mixed results:                 {mixed_rate:5.2f}% ({total - both_correct - both_wrong}/{total})"
        )

        print("\n   Interpretation:")
        print("   - 'Adapts': Model correctly recognizes the visual change")
        print("   - 'Memorizes': Model ignores visual evidence, relies on class priors")
        print("   - 'Mixed': Model partially adapts (one metric correct, one wrong)")

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
            ('adapts', 'Adapts (new=1, old=0)'),
            ('memorizes', 'Memorizes (new=0, old=1)'),
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
        print("   - 'Adapts': Model correctly changes its activation for the new attribute")
        print("   - 'Memorizes': Model keeps old attribute active despite visual evidence")
        print("   - 'Both present/absent': Model predicts both attributes the same way")

    # Per-attribute breakdown
    print("\n" + "-" * 70)
    print("PER-ATTRIBUTE BREAKDOWN")
    print("-" * 70)

    cbm_attr_names = results["cbm_attr_names"]
    per_new_correct = results["per_attr_new_correct"]
    per_new_total = results["per_attr_new_total"]
    per_original_correct = results["per_attr_original_correct"]
    per_original_total = results["per_attr_original_total"]

    print(f"\n{'Attribute':<45} {'New Det.':<15} {'Orig. Present':<15}")
    print("-" * 75)

    for cbm_idx in range(N_ATTRIBUTES_CBM):
        if per_new_total[cbm_idx] > 0:
            new_acc = 100 * per_new_correct[cbm_idx] / per_new_total[cbm_idx]
            new_str = f"{new_acc:5.1f}% ({int(per_new_correct[cbm_idx]):4d}/{int(per_new_total[cbm_idx]):4d})"

            if per_original_total[cbm_idx] > 0:
                # Show complement: how often original is predicted as PRESENT (hallucinated)
                orig_present = per_original_total[cbm_idx] - per_original_correct[cbm_idx]
                orig_present_rate = 100 * orig_present / per_original_total[cbm_idx]
                orig_str = f"{orig_present_rate:5.1f}% ({int(orig_present):4d}/{int(per_original_total[cbm_idx]):4d})"
            else:
                orig_str = "N/A"

            print(f"{cbm_attr_names[cbm_idx]:<45} {new_str:<15} {orig_str:<15}")

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

    # Generate visualization grids
    n_vis_images = getattr(args, "n_vis_images", 10)
    n_vis_grids = getattr(args, "n_vis_grids", 10)
    vis_dir = os.path.join(out_folder_path, "prediction_grids")
    visualize_predictions_grid(
        model=results["model"],
        dataset=results["dataset"],
        device=results["device"],
        cbm_attr_names=results["cbm_attr_names"],
        bird_to_attrs=results["bird_to_attrs"],
        sub_attr_to_cbm_idx=results["sub_attr_to_cbm_idx"],
        output_dir=vis_dir,
        n_images_per_grid=n_vis_images,
        n_grids=n_vis_grids,
    )

    sys.stdout = open(path_to_output_txt, 'a')

    print_results(results)
