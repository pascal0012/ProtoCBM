"""
Visualize heatmaps for a single image using both localization methods:
1. Argmax method: Select highest activated attribute per body part
2. Aggregation method: Sum all attention maps per body part

Usage:
    python visualize_single_image.py --image_path <path_to_image> --config <path_to_config>
"""

import os
import sys
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from localization.visualise import (
    save_individual_activation_maps,
    save_aggregated_activation_maps,
    visualize_keypoint_distances_with_heatmaps,
    visualize_combined_keypoints,
)
from localization.localization_accuracy import (
    compute_localization_accuracy,
    compute_localization_accuracy_aggregated,
    compute_localization_accuracy_centermean,
)
from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils_protocbm.train_utils import create_model
from utils_protocbm.eval_utils import get_localization_loader
import yaml
from PIL import Image as PILImage


def load_cub_part_keypoints(data_dir, image_path, img_size=299):
    """
    Load ground truth part keypoints for a CUB image.

    Args:
        data_dir: Path to CUB_200_2011 data directory
        image_path: Path to the input image (can be full path or relative to images/)
        img_size: Target image size after transformation (default 299)

    Returns:
        part_keypoints: numpy array of shape (15, 2) with (x, y) coordinates
                       scaled to img_size. Invisible parts have (0, 0).
        part_names: list of part names
        original_size: tuple (width, height) of original image
    """
    # Load images.txt to get image_id mapping
    images_file = os.path.join(data_dir, "images.txt")
    image_id_map = {}  # Maps image path suffix to image_id
    with open(images_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            img_name = parts[1]
            image_id_map[img_name] = img_id

    # Load parts.txt to get part names
    parts_file = os.path.join(data_dir, "parts", "parts.txt")
    part_names = {}
    with open(parts_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            part_id = int(parts[0])
            part_name = " ".join(parts[1:])
            part_names[part_id] = part_name

    # Load part_locs.txt
    part_locs_file = os.path.join(data_dir, "parts", "part_locs.txt")
    part_locs = {}  # Maps (image_id, part_id) to (x, y, visible)
    with open(part_locs_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            part_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            visible = int(parts[4])
            part_locs[(img_id, part_id)] = (x, y, visible)

    # Find the image_id for the input image
    # Extract the relative path (e.g., "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg")
    image_id = None

    # Try to match the image path
    for img_name, img_id in image_id_map.items():
        if image_path.endswith(img_name) or img_name in image_path:
            image_id = img_id
            break

    if image_id is None:
        print(f"Warning: Could not find image '{image_path}' in CUB dataset. Using zero keypoints.")
        return np.zeros((15, 2)), list(part_names.values()), None

    # Get original image size to compute scaling factor
    # Try to load the original image
    original_size = None
    try:
        # Try the provided path first
        if os.path.exists(image_path):
            with PILImage.open(image_path) as img:
                original_size = img.size  # (width, height)
        else:
            # Try to construct the path from data_dir
            for img_name, img_id in image_id_map.items():
                if img_id == image_id:
                    full_path = os.path.join(data_dir, "images", img_name)
                    if os.path.exists(full_path):
                        with PILImage.open(full_path) as img:
                            original_size = img.size
                    break
    except Exception as e:
        print(f"Warning: Could not load original image size: {e}")

    # Extract keypoints for this image
    part_keypoints = np.zeros((15, 2))
    for part_id in range(1, 16):  # Parts are 1-indexed
        if (image_id, part_id) in part_locs:
            x, y, visible = part_locs[(image_id, part_id)]
            if visible == 1 and original_size is not None:
                # Scale coordinates to img_size
                orig_w, orig_h = original_size
                scale_x = img_size / orig_w
                scale_y = img_size / orig_h
                part_keypoints[part_id - 1] = [x * scale_x, y * scale_y]
            elif visible == 1:
                # No scaling info available, use raw coordinates
                part_keypoints[part_id - 1] = [x, y]
            # else: invisible part, keep as (0, 0)

    return part_keypoints, list(part_names.values()), original_size


def prepare_model_fast(model, args, load_weights=False):
    """
    Optimized version of prepare_model that skips compilation for faster loading.
    """
    # Load in weights, if any
    if load_weights:
        if "weight_dir" in vars(args):
            state_dict = torch.load(args.weight_dir, weights_only=False)
        elif (
            hasattr(args, "checkpoint")
            and args.checkpoint
            and os.path.exists(args.checkpoint)
        ):
            state_dict = torch.load(args.checkpoint, weights_only=False)
        else:
            path_to_weights = (
                args.apn_weights_dir
                if args.model_name == "apn"
                else os.path.join(args.log_dir, f"best_model_{args.seed}.pth")
            )
            state_dict = torch.load(path_to_weights, weights_only=False)

        # Compatibility with prior runs that saved the model fully and not only the state dict
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()

        # Remove auxiliary logits and concept mapper, as it is not needed for inference
        if not args.model_name == "apn":
            keys_to_remove = [
                k
                for k in state_dict.keys()
                if "AuxLogits" in k or "aux_concept_mapper" in k
            ]
            for k in keys_to_remove:
                del state_dict[k]
        model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Skip compilation if requested (faster loading)
    skip_compile = getattr(args, "skip_compile", False)
    if not skip_compile:
        model.compile()
    else:
        print("Skipping torch.compile() for faster loading...")

    return model, device


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def dict_to_namespace(d):
    """Convert dictionary to namespace for argparse compatibility."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace


def load_and_preprocess_image(image_path, transform, device):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)


def get_transform(img_size=299):
    """Get the standard transformation for CUB images."""
    mean = [0.5, 0.5, 0.5]
    std = [2.0, 2.0, 2.0]

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform, mean, std


def visualize_single_image(image_path, config_path, output_dir=None, skip_compile=True):
    """
    Visualize heatmaps for a single image.

    Args:
        image_path: Path to the input image
        config_path: Path to the YAML config file
        output_dir: Directory to save visualizations (default: ./outputs/single_image_viz)
        skip_compile: If True, skip torch.compile() for faster loading (default: True)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load configuration
    config_dict = load_config(config_path)
    args = dict_to_namespace(config_dict)

    # Store skip_compile preference
    args.skip_compile = skip_compile

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("outputs", "single_image_viz")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from config: {config_path}")

    # Create the model and load weights
    model = create_model(args)
    model, device = prepare_model_fast(model, args, load_weights=True)
    model.eval()

    # Get image size (default 299 for Inception)
    img_size = 299

    # Get transform and statistics
    transform, transform_mean, transform_std = get_transform(img_size)

    print(f"Loading image: {image_path}")

    # Load and preprocess the image
    img_tensor = load_and_preprocess_image(image_path, transform, device)

    # We need to load the dataset to get attribute names and mappings
    # We'll use batch_size=1 to load just one sample from the dataset
    # For single image visualization, we only need the metadata, not actual data loading
    args.batch_size = 1
    loader, _, _, _ = get_localization_loader(
        model, args.data_dir, args.split_dir, args
    )

    attribute_names = loader.dataset.attribute_names
    part_dict = loader.dataset.part_dict
    map_part_to_attr_loc_acc = loader.dataset.map_part_to_attr_loc_acc

    print(f"Number of attributes: {len(attribute_names)}")
    print(f"Number of body parts: {len(part_dict)}")

    # Create dummy labels (we don't need them for visualization)
    # We'll use zeros since we're only interested in the heatmaps
    attr_labels = torch.zeros(1, len(attribute_names)).to(device)

    print("Generating saliency maps...")

    # Pass through model, get model prediction and saliency map per attribute
    with torch.no_grad():
        pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(
            model, img_tensor, args, attr_labels=attr_labels
        )
        saliency_maps = saliency_maps.to(device)

    print(f"Prediction: Class {pred.argmax().item()}")
    print(f"Saliency maps shape: {saliency_maps.shape}")

    # Load ground truth part keypoints from CUB dataset
    B = 1
    K = len(part_dict)

    # Try to load real keypoints from CUB dataset
    cub_data_dir = os.path.join(args.data_dir, "CUB_200_2011")
    if not os.path.exists(cub_data_dir):
        cub_data_dir = args.data_dir  # Fallback if structure is different

    gt_keypoints, cub_part_names, original_size = load_cub_part_keypoints(
        cub_data_dir, image_path, img_size=img_size
    )

    if original_size is not None:
        print(f"Original image size: {original_size[0]}x{original_size[1]}")
        print(f"Loaded {np.sum(gt_keypoints.sum(axis=1) > 0)} visible part keypoints")
    else:
        print("Could not load ground truth keypoints for this image")

    # Convert to tensor with batch dimension
    part_gts = torch.from_numpy(gt_keypoints).float().unsqueeze(0).to(device)  # [1, K, 2]
    part_bbs = torch.zeros(B, K, 4).to(device)

    # Empty collectors (we won't compute metrics without GT)
    loc_acc_collector_argmax = []
    loc_acc_collector_aggregated = []

    print("Computing localization with ARGMAX method...")

    # ========== METHOD 1: ARGMAX (Original) ==========
    predicted_coords_argmax, dists_argmax, heatmaps_argmax, _ = (
        compute_localization_accuracy(
            scores,
            saliency_maps,
            part_bbs,
            part_gts,
            part_dict,
            map_part_to_attr_loc_acc,
            loc_acc_collector_argmax,
            img_size=img_size,
        )
    )

    print("Computing localization with AGGREGATED method...")

    # ========== METHOD 2: AGGREGATED ==========
    predicted_coords_agg, dists_agg, heatmaps_agg, _ = (
        compute_localization_accuracy_aggregated(
            scores,
            saliency_maps,
            part_bbs,
            part_gts,
            part_dict,
            map_part_to_attr_loc_acc,
            loc_acc_collector_aggregated,
            img_size=img_size,
            interpolate=False,
        )
    )

    # ========== METHOD 3: Centermean ==========
    predicted_coords_center, dists_center, heatmaps_center, _ = (
        compute_localization_accuracy_centermean(
            scores,
            saliency_maps,
            part_bbs,
            part_gts,
            part_dict,
            map_part_to_attr_loc_acc,
            loc_acc_collector_aggregated,
            img_size=img_size,
        )
    )

    print(f"Argmax heatmaps shape: {heatmaps_argmax.shape}")
    print(f"Aggregated heatmaps shape: {heatmaps_agg.shape}")

    # Create separate output directories for each method
    out_dir_argmax = os.path.join(output_dir, "argmax_method")
    out_dir_agg = os.path.join(output_dir, "aggregated_method")
    out_dir_center = os.path.join(output_dir, "centermean_method")
    out_dir_combined = os.path.join(output_dir, "combined_keypoints")

    os.makedirs(out_dir_argmax, exist_ok=True)
    os.makedirs(out_dir_agg, exist_ok=True)
    os.makedirs(out_dir_center, exist_ok=True)
    os.makedirs(out_dir_combined, exist_ok=True)

    # Prepare source paths (just the filename)
    source_paths = [os.path.basename(image_path)]

    batch_idx = 0

    print("Saving visualizations...")

    # Visualize ARGMAX method with interpolated heatmaps
    visualize_keypoint_distances_with_heatmaps(
        part_gts,
        img_tensor,
        source_paths,
        predicted_coords_argmax,
        dists_argmax,
        heatmaps_argmax,  # Already interpolated from compute_localization_accuracy
        0,
        list(part_dict.values()),
        batch_idx=batch_idx,
        t_mean=transform_mean,
        t_std=transform_std,
        save_path=out_dir_argmax,
        img_size=img_size,
    )

    # Visualize AGGREGATED method with interpolated heatmaps
    visualize_keypoint_distances_with_heatmaps(
        part_gts,
        img_tensor,
        source_paths,
        predicted_coords_agg,
        dists_agg,
        heatmaps_agg,  # Original size, will be interpolated in the function
        0,
        list(part_dict.values()),
        batch_idx=batch_idx,
        t_mean=transform_mean,
        t_std=transform_std,
        save_path=out_dir_agg,
        img_size=img_size,
    )

    # Visualize AGGREGATED method with interpolated heatmaps
    visualize_keypoint_distances_with_heatmaps(
        part_gts,
        img_tensor,
        source_paths,
        predicted_coords_center,
        dists_center,
        heatmaps_center,  # Original size, will be interpolated in the function
        0,
        list(part_dict.values()),
        batch_idx=batch_idx,
        t_mean=transform_mean,
        t_std=transform_std,
        save_path=out_dir_center,
        img_size=img_size,
    )

    # ========== COMBINED KEYPOINTS VISUALIZATION ==========
    # Call with 3 prediction types
    visualize_combined_keypoints(
        gts=part_gts,
        imgs=img_tensor,
        img_paths=source_paths,
        predictions={
            "argmax": predicted_coords_argmax,
            "agg": predicted_coords_agg,
            "center": predicted_coords_center,
        },
        distances={
            "argmax": dists_argmax,
            "agg": dists_agg,
            "center": dists_center,
        },
        batch_idx=batch_idx,
        part_names=list(part_dict.values()),
        save_path=out_dir_combined,
    )

    # ========== SAVE INDIVIDUAL ACTIVATION MAPS ==========
    print("Saving individual activation maps...")

    # Create subdirectories for individual activation maps
    out_dir_argmax_individual = os.path.join(out_dir_argmax, "individual_maps")
    out_dir_agg_individual = os.path.join(out_dir_agg, "individual_maps")
    os.makedirs(out_dir_argmax_individual, exist_ok=True)
    os.makedirs(out_dir_agg_individual, exist_ok=True)

    # For ARGMAX method: save individual activation maps per attribute
    save_individual_activation_maps(
        img_tensor,
        saliency_maps,
        attribute_names,
        map_part_to_attr_loc_acc,
        source_paths,
        batch_idx=batch_idx,
        t_mean=transform_mean,
        t_std=transform_std,
        save_path=out_dir_argmax_individual,
    )

    # For AGGREGATED method: save the aggregated heatmaps per body part
    save_aggregated_activation_maps(
        img_tensor,
        heatmaps_agg,
        list(part_dict.values()),
        source_paths,
        batch_idx=batch_idx,
        t_mean=transform_mean,
        t_std=transform_std,
        save_path=out_dir_agg_individual,
        normalize_heatmaps=False,  # Use raw values without rescaling
    )

    print("\nVisualization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Argmax method: {out_dir_argmax}")
    print(f"  - Aggregated method: {out_dir_agg}")
    print(f"  - Combined keypoints: {out_dir_combined}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize heatmaps for a single image"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_protocbm_comparison.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: ./outputs/single_image_viz)",
    )
    parser.add_argument(
        "--no-skip-compile",
        action="store_true",
        help="Enable torch.compile() (slower loading, faster inference for batches)",
    )

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    visualize_single_image(
        args.image_path,
        args.config,
        args.output_dir,
        skip_compile=not args.no_skip_compile,
    )


if __name__ == "__main__":
    main()
