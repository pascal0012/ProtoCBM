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
"""

import os
import pickle
import sys

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


def build_bird_class_attribute_cache(val_pkl_path, img_dir):
    """
    Build a cache mapping bird class names to their original CBM attribute labels.

    This is done once at the start to avoid repeated file system lookups.

    Args:
        val_pkl_path: Path to validation pkl file with attribute labels
        img_dir: Path to CUB images directory (for class name mapping)

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

    # Load validation data and build bird -> attributes mapping
    data = pickle.load(open(val_pkl_path, "rb"))

    bird_to_attrs = {}
    for d in data:
        class_idx = d["class_label"]
        if class_idx in class_idx_to_bird_name:
            bird_name = class_idx_to_bird_name[class_idx]
            # Store with normalized key for reliable matching
            normalized_name = normalize_bird_name(bird_name)
            if normalized_name not in bird_to_attrs:
                bird_to_attrs[normalized_name] = d["attribute_label"]

    return bird_to_attrs


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


def get_attribute_predictions(model, inputs, device):
    """
    Get attribute predictions from model output.

    ProtoCBM models output:
    - CBMMapper: [class_logits, attr1, attr2, ..., attr112] where each attr is [B, 1]
    - ProtoMod: (class_logits, similarity_scores, attention_maps) where sim_scores is [B, 112]

    Returns:
        attr_preds: Tensor of shape [B, 112] with binary predictions
        attr_probs: Tensor of shape [B, 112] with probabilities
    """
    outputs = model(inputs)

    if isinstance(outputs, tuple) and len(outputs) == 3:
        # ProtoMod: (class_logits, similarity_scores, attention_maps)
        _, sim_scores, _ = outputs
        attr_probs = torch.sigmoid(sim_scores)

    elif isinstance(outputs, list) and len(outputs) > 1:
        # CBMMapper: [class_logits, attr1, attr2, ..., attrN]
        attr_outputs = outputs[1:]  # Skip class logits
        attr_probs = torch.sigmoid(torch.cat(attr_outputs, dim=1))
    else:
        raise ValueError(f"Unexpected model output format: {type(outputs)}")

    attr_preds = (attr_probs >= 0.5).float()
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

    print("Building bird class attribute cache...")
    bird_to_attrs = build_bird_class_attribute_cache(val_pkl_path, img_dir)
    print(f"Cached attributes for {len(bird_to_attrs)} bird classes")

    # Tracking metrics
    total_samples = 0
    new_attr_correct = 0  # Correctly predicted the NEW attribute as PRESENT
    original_attr_correct = 0  # Correctly predicted the ORIGINAL attribute as ABSENT
    original_attr_total = 0  # Total samples where we could find the original attribute

    # Per-attribute tracking
    per_attr_new_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_new_total = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_original_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_original_total = torch.zeros(N_ATTRIBUTES_CBM)

    # Track samples where model gets both right (adapts) vs both wrong (memorizes)
    both_correct = 0    # Adapts: sees new as present AND original as absent
    both_wrong = 0      # Memorizes: sees new as absent AND original as present

    # Track unmatched items for debugging
    unmatched_birds = set()
    unmatched_attrs = set()

    with torch.no_grad():
        for inputs, bird_labels, attr_labels in tqdm(
            loader, desc="Evaluating SUB Attributes"
        ):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Get attribute predictions
            attr_preds, _ = get_attribute_predictions(model, inputs, device)
            attr_preds = attr_preds.cpu()

            for i in range(batch_size):
                # Get SUB attribute info
                sub_attr_idx = attr_labels[i].item()                    # index of attr (CUB)
                sub_attr_name = dataset.attr_names[sub_attr_idx]        # name of attr (SUB)
                bird_name = dataset.bird_names[bird_labels[i].item()]   # bird species name (SUB)

                # Map NEW attribute to CBM index
                new_cbm_idx = sub_attr_to_cbm_idx.get(sub_attr_name)    # convert CUB idx to CBM idx
                if new_cbm_idx is None:
                    unmatched_attrs.add(sub_attr_name)
                    continue

                total_samples += 1
                per_attr_new_total[new_cbm_idx] += 1    # Count for attribute

                # Test 1: Is the NEW (substituted) attribute predicted as PRESENT?
                new_is_correct = attr_preds[i, new_cbm_idx] == 1
                if new_is_correct:
                    new_attr_correct += 1
                    per_attr_new_correct[new_cbm_idx] += 1

                # Test 2: Are ALL ORIGINAL attributes predicted as ABSENT?
                # Note: Birds can have multiple active attributes per body part
                original_cbm_indices = find_original_attribute_indices(
                    bird_name, sub_attr_name, cbm_attr_names, bird_to_attrs
                )

                if not original_cbm_indices:
                    unmatched_birds.add(bird_name)
                else:
                    original_attr_total += 1

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

    return {
        "total_samples": total_samples,
        "new_attr_correct": new_attr_correct,
        "original_attr_correct": original_attr_correct,
        "original_attr_total": original_attr_total,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "per_attr_new_correct": per_attr_new_correct,
        "per_attr_new_total": per_attr_new_total,
        "per_attr_original_correct": per_attr_original_correct,
        "per_attr_original_total": per_attr_original_total,
        "cbm_attr_names": cbm_attr_names,
        "unmatched_birds": unmatched_birds,
        "unmatched_attrs": unmatched_attrs,
    }


def print_results(results):
    """Print formatted evaluation results."""
    total = results["total_samples"]
    new_correct = results["new_attr_correct"]
    original_correct = results["original_attr_correct"]
    original_total = results["original_attr_total"]
    both_correct = results["both_correct"]
    both_wrong = results["both_wrong"]
    unmatched_birds = results["unmatched_birds"]
    unmatched_attrs = results["unmatched_attrs"]

    print("\n" + "=" * 70)
    print("SUB BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nTotal samples evaluated: {total}")
    print(f"Samples with original attribute found: {original_total}")

    # Warnings for unmatched items
    if unmatched_attrs:
        print(
            f"\nWARNING: {len(unmatched_attrs)} SUB attributes could not be mapped to CBM:"
        )
        for attr in sorted(unmatched_attrs)[:5]:  # Show first 5
            print(f"   - {attr}")
        if len(unmatched_attrs) > 5:
            print(f"   ... and {len(unmatched_attrs) - 5} more")

    if unmatched_birds:
        print(
            f"\nWARNING: {len(unmatched_birds)} bird classes could not find original attribute:"
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

    original_acc = 100 * original_correct / original_total if original_total > 0 else 0
    print("\n2. Original Attribute Detection (should predict original attr as ABSENT):")
    print(f"   Accuracy: {original_acc:.2f}% ({original_correct}/{original_total})")

    # Adaptation vs Memorization analysis
    print("\n" + "-" * 70)
    print("ADAPTATION vs MEMORIZATION ANALYSIS")
    print("-" * 70)

    if original_total > 0:
        adapt_rate = 100 * both_correct / original_total
        memorize_rate = 100 * both_wrong / original_total
        mixed_rate = 100 * (original_total - both_correct - both_wrong) / original_total

        print(
            f"\n   Adapts (new=1, original=0):    {adapt_rate:5.2f}% ({both_correct}/{original_total})"
        )
        print(
            f"   Memorizes (new=0, original=1): {memorize_rate:5.2f}% ({both_wrong}/{original_total})"
        )
        print(
            f"   Mixed results:                 {mixed_rate:5.2f}% ({original_total - both_correct - both_wrong}/{original_total})"
        )

        print("\n   Interpretation:")
        print("   - 'Adapts': Model correctly recognizes the visual change")
        print("   - 'Memorizes': Model ignores visual evidence, relies on class priors")
        print("   - 'Mixed': Model partially adapts (one metric correct, one wrong)")

    # Per-attribute breakdown
    print("\n" + "-" * 70)
    print("PER-ATTRIBUTE BREAKDOWN")
    print("-" * 70)

    cbm_attr_names = results["cbm_attr_names"]
    per_new_correct = results["per_attr_new_correct"]
    per_new_total = results["per_attr_new_total"]
    per_original_correct = results["per_attr_original_correct"]
    per_original_total = results["per_attr_original_total"]

    print(f"\n{'Attribute':<45} {'New Det.':<15} {'Orig. Absent':<15}")
    print("-" * 75)

    for cbm_idx in range(N_ATTRIBUTES_CBM):
        if per_new_total[cbm_idx] > 0:
            new_acc = 100 * per_new_correct[cbm_idx] / per_new_total[cbm_idx]
            new_str = f"{new_acc:5.1f}% ({int(per_new_correct[cbm_idx]):4d}/{int(per_new_total[cbm_idx]):4d})"

            if per_original_total[cbm_idx] > 0:
                orig_acc = (
                    100 * per_original_correct[cbm_idx] / per_original_total[cbm_idx]
                )
                orig_str = f"{orig_acc:5.1f}% ({int(per_original_correct[cbm_idx]):4d}/{int(per_original_total[cbm_idx]):4d})"
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
    
    sys.stdout = open(path_to_output_txt, 'a')

    print_results(results)
