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
import csv

import numpy as np
import torch
from tqdm import tqdm

from cub.dataset import SUBDataset
from cub.config import BASE_DIR
from utils_protocbm.train_utils import gather_args
from utils_protocbm.eval_utils import get_eval_transform_for_model, create_model_for_eval, sub_to_cbm
from utils_protocbm.mappings import CBM_SELECTED_CUB_ATTRIBUTE_IDS
from torch.utils.data import DataLoader
import pathlib
import pickle
import datasets


BASE_DIR = pathlib.Path(".").absolute()
ATTRIBUTE_FILE = os.path.join(BASE_DIR, 'data/CUB_200_2011/attributes/attributes.txt')
LABEL_PATH = os.path.join(BASE_DIR, 'data/CUB_processed/class_attr_data_10/val.pkl')
IMG_DIR_PATH = os.path.join(BASE_DIR, 'data/CUB_200_2011/images')
SUB_DATASET_PATH = os.path.join(BASE_DIR, 'data/SUB')

def get_list_of_used_attributes():
    """
    Get the names of the 112 CBM-selected attributes.
    Returns list of attribute names indexed by CBM attribute index.
    """
    with open(ATTRIBUTE_FILE, 'r') as f:
        attributes = f.readlines()
    used_attributes = [attributes[m] for m in CBM_SELECTED_CUB_ATTRIBUTE_IDS]
    used_attributes = [a.replace('\n', '').split(' ')[1] for a in used_attributes]
    return used_attributes


def build_class_attr_cache():
    """Load binary attribute labels per class from val.pkl (matches reference get_base_attributes)."""
    folders = os.listdir(IMG_DIR_PATH)
    class_idx_to_bird_name = {}
    for f in folders:
        parts = f.split('.')
        if len(parts) >= 2:
            class_idx = int(parts[0]) - 1
            class_idx_to_bird_name[class_idx] = parts[1]

    data = pickle.load(open(LABEL_PATH, 'rb'))
    # First entry per class (same as reference)
    class_idx_to_attrs = {}
    for d in data:
        cls = d['class_label']
        if cls not in class_idx_to_attrs:
            class_idx_to_attrs[cls] = d['attribute_label']

    bird_to_attrs = {}
    for class_idx, bird_name in class_idx_to_bird_name.items():
        attrs = class_idx_to_attrs.get(class_idx)
        if attrs is not None:
            bird_to_attrs[bird_name] = attrs
    return bird_to_attrs

def get_a_old(bird_name, attr_type_prefix, used_attributes, bird_attr_cache):
    """Find the original active attribute index for a bird, for the given attribute type."""
    to_check = [ua for ua in used_attributes if attr_type_prefix in ua]
    attr_label = bird_attr_cache.get(bird_name)
    
    if attr_label is None:
        return None
    
    for tc in to_check:
        # simply returns the first found original attr. 🤦‍♂️
        if attr_label[used_attributes.index(tc)]:
            return used_attributes.index(tc)
    return None

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
        # CBM always outputs logits, so sigmoid is always applied.
        # With classifier: outputs = [class_logits [B,200], attr0 [B,1], ..., attr111 [B,1]]
        # XC-only (no classifier): outputs = [attr0 [B,1], ..., attr111 [B,1]]
        # Distinguish by checking if outputs[0] is class logits (dim > 1) or an attr (dim == 1)
        if outputs[0].shape[-1] == 1:
            attr_outputs = outputs  # XC-only: all outputs are attr logits
        else:
            attr_outputs = outputs[1:]  # skip class_logits at index 0

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


    print("Loading CBM attributes...")
    used_attributes = get_list_of_used_attributes()

    # Get transforms for the model
    transform = get_eval_transform_for_model(model, args)[0]

    print("Loading val.pkl class attribute cache...")
    bird_attr_cache = build_class_attr_cache()
    print(f"  {len(bird_attr_cache)} bird classes cached")

    print(f"Loading SUB dataset from {SUB_DATASET_PATH}")
    dataset = SUBDataset(
        SUB_DATASET_PATH, transform=transform, only_cub_attributes=True
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



    # Tracking metrics
    s_plus_incorrect = 0
    s_plus_num = 0
    s_minus_incorrect = 0
    s_minus_num = 0


    # Track feature differences between old and new attributes
    feature_diff_stats = {
        'raw_dist': [],  # L1 distance between old and new feature activations (all cases)
        'sig_dist': [],  # distances between probabilities
    }

    with torch.no_grad():
        for images, bird_labels, attr_labels in tqdm(
            loader, desc="Evaluating SUB Attributes"
        ):
            images = images.to(device)
            attr_labels = attr_labels.float().to(device)

            batch_size = images.size(0)

            # Get attribute predictions and raw features
            use_sigmoid = getattr(args, "use_sigmoid_logits", False)
            attr_preds, attr_probs, attr_features = get_attribute_predictions(
                model, (images, attr_labels), return_features=True, use_sigmoid=use_sigmoid
            )

            attr_preds = attr_preds.cpu()
            attr_probs = attr_probs.cpu()
            attr_features = attr_features.cpu()


            for i in range(batch_size):
                # Get SUB attribute info
                sub_attr_idx = int(attr_labels[i].item())               # index of attr (CUB)
                changed_attr = sub_to_cbm(dataset.attr_names[sub_attr_idx])   

                try:
                    idx = used_attributes.index(changed_attr)
                except ValueError:
                    continue

                # S+: is the substituted attribute detected?
                s_plus_num += 1
                if not attr_preds[i][idx]:
                    s_plus_incorrect += 1

                # S-: is the original attribute still hallucinated?
                attr_type = changed_attr.split('::')[0]  # e.g. 'has_wing_color'
                bird_name = dataset.bird_names[bird_labels[i].item()]

                orig_idx = get_a_old(bird_name, attr_type, used_attributes, bird_attr_cache)
                if orig_idx is not None:
                    s_minus_num += 1
                    if attr_preds[i][orig_idx]:
                        s_minus_incorrect += 1

                    # Compute feature differences between old and new attributes
                    new_feature_raw = attr_features[i, idx]
                    old_feature_raw = attr_features[i, orig_idx]

                    new_feature_probs = attr_probs[i, idx]
                    old_feature_probs = attr_probs[i, orig_idx]

                    # L1 distance between raw features
                    feature_diff_stats['raw_dist'].append(
                        torch.abs(new_feature_raw - old_feature_raw).item()
                    )

                    # Distance between probabilities
                    feature_diff_stats['sig_dist'].append(
                        (new_feature_probs - old_feature_probs).item()
                    )


    s_plus_correct = s_plus_num - s_plus_incorrect
    s_minus_correct = s_minus_num - s_minus_incorrect

    print("s_plus_correct:", s_plus_correct)
    print("s_plus_num:", s_plus_num)

    print("s_minus_correct:", s_minus_correct)
    print("s_minus_num:", s_minus_num)

    print("="*40)
    print("=== Results ===")
    print("="*40)
    print(f"S+ total correct:   {s_plus_correct / s_plus_num:.4f}  ({s_plus_correct}/{s_plus_num})")
    print(f"S+ total incorrect: {s_plus_incorrect / s_plus_num:.4f}")
    print(f"S- total correct:   {s_minus_correct / s_minus_num:.4f}  ({s_minus_correct}/{s_minus_num})")
    print(f"S- total incorrect: {s_minus_incorrect / s_minus_num:.4f}")

    print("\n=== Feature Difference Stats ===")
    for k, v in feature_diff_stats.items():
        if len(v) > 0:
            print(f"{k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}")

    return {
        "total_samples": s_plus_num,
        "original_samples": s_minus_num,
        "new_attr_correct": s_plus_num - s_plus_incorrect,
        "original_attr_correct": s_minus_num - s_minus_incorrect,
        "feature_diff_stats": feature_diff_stats,
    }


def append_results_to_global_csv(args, results):
    """Append S+/S- results for this run to outputs/eval_sub_results.csv."""
    csv_path = os.path.join("outputs", "eval_sub_results.csv")
    os.makedirs("outputs", exist_ok=True)

    s_plus_correct = results["new_attr_correct"] / results["total_samples"]
    s_minus_correct = results["original_attr_correct"] / results["original_samples"]

    run_name = f"{getattr(args, 'model_name', 'unknown')}_{getattr(args, 'mode', 'unknown')}_seed{getattr(args, 'seed', 'unknown')}"

    row = {
        "run_name": run_name,
        "model_name": getattr(args, "model_name", ""),
        "mode": getattr(args, "mode", ""),
        "seed": getattr(args, "seed", ""),
        "config_path": getattr(args, "config_path", ""),
        "s_plus_correct": f"{s_plus_correct:.4f}",
        "s_minus_correct": f"{s_minus_correct:.4f}",
        "s_plus_num": results["total_samples"],
        "s_minus_num": results["original_samples"],
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Print to original stdout so it's visible even when stdout is redirected
    print(f"Results appended to {csv_path}", file=sys.__stdout__)


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

    sys.stdout = open(path_to_output_txt, 'a')
    results = eval_sub_attributes(args)
    append_results_to_global_csv(args, results)
