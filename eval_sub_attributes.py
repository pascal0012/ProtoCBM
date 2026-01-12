"""
Evaluate trained models on the SUB dataset for ATTRIBUTE prediction accuracy.
This is adapted from: https://github.com/ExplainableML/sub/blob/main/CBM_testing/test_ind_cbm_example.py

The script tests:
1. Whether the model correctly predicts the NEW (substituted) attribute as present
2. Optionally (--test_for_complement): Whether the ORIGINAL attribute is predicted as NOT present

This differs from eval_sub.py which evaluates bird CLASS prediction accuracy.
"""

import os
import pickle

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
    cbm_attr_names = [idx_to_attr[cub_idx] for cub_idx in CBM_SELECTED_CUB_ATTRIBUTE_IDS]
    return cbm_attr_names


def create_sub_to_cbm_mapping():
    """
    Create mapping from SUB attribute names to CBM attribute indices.

    SUB uses format: "has_breast_color--red"
    CUB uses format: "has_breast_color::red"

    Returns:
        sub_attr_to_cbm_idx: Dict mapping SUB attr name -> CBM index (0-111)
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


def get_base_attributes_for_bird(bird_name, val_pkl_path, img_dir):
    """
    Get the base (original) attribute labels for a bird class from CUB validation set.

    Args:
        bird_name: Bird species name (e.g., "Cardinal")
        val_pkl_path: Path to validation pkl file
        img_dir: Path to CUB images directory

    Returns:
        attribute_label: List of 112 binary attribute labels for this bird class
    """
    # Find the class index for this bird name
    folders = os.listdir(img_dir)
    bird_class_idx = None
    for folder in folders:
        parts = folder.split(".")
        if len(parts) >= 2:
            # Handle names like "017.Cardinal" -> "Cardinal"
            folder_bird_name = parts[1].replace("_", " ")
            if bird_name.replace("_", " ") == folder_bird_name or bird_name == parts[1]:
                bird_class_idx = int(parts[0]) - 1  # Convert to 0-indexed
                break

    if bird_class_idx is None:
        return None

    # Load validation data and find an example of this class
    data = pickle.load(open(val_pkl_path, "rb"))
    for d in data:
        if d["class_label"] == bird_class_idx:
            return d["attribute_label"]

    return None


def find_original_attribute_idx(bird_name, changed_attr_sub_name, cbm_attr_names, val_pkl_path, img_dir):
    """
    Find which original attribute was replaced by the changed attribute.

    For example, if a Cardinal normally has "has_breast_color::red" and SUB changed it to
    "has_breast_color::blue", this function returns the CBM index for "has_breast_color::red".

    Args:
        bird_name: Bird species name
        changed_attr_sub_name: SUB attribute name that was changed (e.g., "has_breast_color--blue")
        cbm_attr_names: List of CBM attribute names
        val_pkl_path: Path to validation pkl file
        img_dir: Path to CUB images directory

    Returns:
        CBM index of the original attribute that was present before substitution, or None
    """
    # Get the attribute type (e.g., "has_breast_color" from "has_breast_color--blue")
    attr_type = changed_attr_sub_name.split("--")[0]

    # Find all CBM attributes of the same type
    candidate_indices = []
    for cbm_idx, cub_name in enumerate(cbm_attr_names):
        if attr_type in cub_name:
            candidate_indices.append(cbm_idx)

    if not candidate_indices:
        return None

    # Get the base attributes for this bird
    base_attrs = get_base_attributes_for_bird(bird_name, val_pkl_path, img_dir)
    if base_attrs is None:
        return None

    # Find which candidate attribute is present in the base bird
    for cbm_idx in candidate_indices:
        if base_attrs[cbm_idx]:
            return cbm_idx

    return None


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
    sub_limit = getattr(args, "sub_limit", None)

    print(f"Loading SUB dataset from {sub_data_dir}")
    dataset = SUBDataset(sub_data_dir, transform=transform, limit=sub_limit, only_cub_attributes=True)

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
    val_pkl_path = os.path.join(BASE_DIR, args.data_dir, "val.pkl")
    img_dir = os.path.join(BASE_DIR, "data/CUB_200_2011/images")

    # Tracking metrics
    total_samples = 0
    new_attr_correct = 0  # Correctly predicted the NEW attribute as present
    old_attr_correct = 0  # Correctly predicted the OLD attribute as absent (complement test)
    old_attr_total = 0    # Total samples where we could find the old attribute

    # Per-attribute tracking
    per_attr_new_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_new_total = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_old_correct = torch.zeros(N_ATTRIBUTES_CBM)
    per_attr_old_total = torch.zeros(N_ATTRIBUTES_CBM)

    test_complement = getattr(args, "test_for_complement", False)

    with torch.no_grad():
        for batch_idx, (inputs, bird_labels, attr_labels) in enumerate(tqdm(loader, desc="Evaluating SUB Attributes")):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Get attribute predictions
            attr_preds, attr_probs = get_attribute_predictions(model, inputs, device)
            attr_preds = attr_preds.cpu()

            for i in range(batch_size):
                # Get SUB attribute info
                sub_attr_idx = attr_labels[i].item()
                sub_attr_name = dataset.attr_names[sub_attr_idx]
                bird_name = dataset.bird_names[bird_labels[i].item()]

                # Map to CBM index
                cbm_idx = sub_attr_to_cbm_idx.get(sub_attr_name)
                if cbm_idx is None:
                    continue

                total_samples += 1
                per_attr_new_total[cbm_idx] += 1

                # Test 1: Is the NEW (substituted) attribute predicted as present?
                if attr_preds[i, cbm_idx] == 1:
                    new_attr_correct += 1
                    per_attr_new_correct[cbm_idx] += 1

                # Test 2 (optional): Is the ORIGINAL attribute predicted as absent?
                if test_complement:
                    old_cbm_idx = find_original_attribute_idx(
                        bird_name, sub_attr_name, cbm_attr_names, val_pkl_path, img_dir
                    )
                    if old_cbm_idx is not None:
                        old_attr_total += 1
                        per_attr_old_total[old_cbm_idx] += 1
                        # Original should NOT be present (pred == 0 is correct)
                        if attr_preds[i, old_cbm_idx] == 0:
                            old_attr_correct += 1
                            per_attr_old_correct[old_cbm_idx] += 1

    return {
        "total_samples": total_samples,
        "new_attr_correct": new_attr_correct,
        "old_attr_correct": old_attr_correct,
        "old_attr_total": old_attr_total,
        "per_attr_new_correct": per_attr_new_correct,
        "per_attr_new_total": per_attr_new_total,
        "per_attr_old_correct": per_attr_old_correct,
        "per_attr_old_total": per_attr_old_total,
        "cbm_attr_names": cbm_attr_names,
        "test_complement": test_complement,
    }


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = gather_args()

    # Add complement testing flag if not present
    if not hasattr(args, "test_for_complement"):
        args.test_for_complement = False

    print("\n" + "=" * 60)
    print("SUB Dataset - Attribute Prediction Evaluation")
    print("=" * 60 + "\n")

    results = eval_sub_attributes(args)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total = results["total_samples"]
    new_correct = results["new_attr_correct"]

    print(f"\nTotal samples evaluated: {total}")
    print("\nNew Attribute Detection (should predict substituted attr as PRESENT):")
    print(f"  Accuracy: {100 * new_correct / total:.2f}% ({new_correct}/{total})")

    if results["test_complement"]:
        old_total = results["old_attr_total"]
        old_correct = results["old_attr_correct"]
        print("\nOriginal Attribute Complement (should predict original attr as ABSENT):")
        print(f"  Accuracy: {100 * old_correct / old_total:.2f}% ({old_correct}/{old_total})")

    # Per-attribute breakdown for new attribute detection
    print("\n--- Per-Attribute New Detection Accuracies ---")
    cbm_attr_names = results["cbm_attr_names"]
    per_new_correct = results["per_attr_new_correct"]
    per_new_total = results["per_attr_new_total"]

    for cbm_idx in range(N_ATTRIBUTES_CBM):
        if per_new_total[cbm_idx] > 0:
            acc = 100 * per_new_correct[cbm_idx] / per_new_total[cbm_idx]
            correct = int(per_new_correct[cbm_idx].item())
            total = int(per_new_total[cbm_idx].item())
            print(f"  {cbm_attr_names[cbm_idx]}: {acc:.2f}% ({correct}/{total})")
