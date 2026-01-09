"""
Evaluate trained models on the official CUB test set with BOTH localization methods for comparison:
1. Argmax method: Select highest activated attribute per body part
2. Aggregation method: Sum all attention maps per body part
"""

import os
import sys
import torch
from tqdm import tqdm
import numpy as np

from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics
from localization.visualise import (
    create_attribute_mosaic,
    visualise_part_segmentations,
    visualise_localization_acc_boxes,
    plot_threshold_curve,
    visualize_keypoint_distances,
    save_individual_activation_maps,
    save_aggregated_activation_maps
)
from localization.localization_accuracy import (
    calculate_average_partwise_localization_accuracy,
    compute_localization_accuracy,
    compute_localization_accuracy_aggregated,  # NEW METHOD
    calculate_average_partwise_localization_distance,
    compute_localization_accuracy_without_argmaxing
)
from models.apn_baseline import load_apn_baseline
from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils_protocbm.mappings import MAP_RESULT_GROUPS_TO_CUB_GROUPS
from utils_protocbm.eval_utils import LocalizationMeter, get_localization_loader
from utils_protocbm.train_utils import AverageMeter, accuracy, binary_accuracy, prepare_model, model_by_mode, gather_args


def create_model(args):
    if args.model_name == "protocbm":
        model = model_by_mode(args)
    elif args.model_name == "cbm":
        model = model_by_mode(args)
    elif args.model_name == "apn":
        model = load_apn_baseline(args)
    else:
        raise ValueError("")
    return model


def eval(args):
    """
        Evaluate model with BOTH localization methods for comparison.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create the model and load weights
    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
    model.eval()

    # Get the localization data loader and additional transform statistics
    loader, transform_mean, transform_std, img_size = get_localization_loader(model, args.data_dir, args.split_dir, args)
    map_attr_id_to_part_seg_group = loader.dataset.map_attr_id_to_part_seg_group
    attribute_names = loader.dataset.attribute_names
    unmatched_attr_mask = loader.dataset.unmatched_attr_mask

    # Collecting metric values across batches for proper mean
    seg_loc_meter = LocalizationMeter(attribute_names, device)

    # Two separate collectors for the two methods
    loc_acc_collector_argmax = []
    loc_acc_collector_aggregated = []

    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Evaluating batches")):
            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data
            attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

            # Pass through model, get model prediction and saliency map per attribute
            pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(model, inputs, args, attr_labels=attr_labels)
            saliency_maps = saliency_maps.to(device)

            # Calculate classification accuracy
            class_acc = accuracy(pred, labels, topk=(1,))
            class_acc_meter.update(class_acc[0], pred.size(0))

            # Calculate attribute accuracy
            attr_acc = binary_accuracy(scores, attr_labels)
            attr_acc_meter.update(attr_acc, pred.size(0))

            # ========== METHOD 1: ARGMAX (Original) ==========
            predicted_coords_argmax, dists_argmax, heatmaps_argmax, _ = compute_localization_accuracy(
                scores, saliency_maps, part_bbs, part_gts, loader.dataset.part_dict,
                loader.dataset.map_part_to_attr_loc_acc, loc_acc_collector_argmax, img_size=img_size
            )

            # ========== METHOD 2: AGGREGATED (New) ==========
            predicted_coords_agg, dists_agg, heatmaps_agg, _ = compute_localization_accuracy_aggregated(
                scores, saliency_maps, part_bbs, part_gts, loader.dataset.part_dict,
                loader.dataset.map_part_to_attr_loc_acc, loc_acc_collector_aggregated, img_size=img_size
            )

            # Visualize both methods for comparison (only for selected samples)
            if args.vis_every_n > 0 and data_idx % args.vis_every_n == 0:
                # Create separate output directories for each method
                out_dir_argmax = os.path.join(args.out_dir_part_seg, "argmax_method")
                out_dir_agg = os.path.join(args.out_dir_part_seg, "aggregated_method")
                os.makedirs(out_dir_argmax, exist_ok=True)
                os.makedirs(out_dir_agg, exist_ok=True)

                # Always visualize the first image in each batch for consistency
                batch_idx = 0

                # Visualize ARGMAX method
                visualize_keypoint_distances(
                    part_gts, inputs, source_paths, predicted_coords_argmax, dists_argmax,
                    data_idx, list(loader.dataset.part_dict.values()),
                    batch_idx=batch_idx,
                    t_mean=transform_mean, t_std=transform_std, save_path=out_dir_argmax
                )

                # Visualize AGGREGATED method
                visualize_keypoint_distances(
                    part_gts, inputs, source_paths, predicted_coords_agg, dists_agg,
                    data_idx, list(loader.dataset.part_dict.values()),
                    batch_idx=batch_idx,
                    t_mean=transform_mean, t_std=transform_std, save_path=out_dir_agg
                )

                # ========== SAVE INDIVIDUAL ACTIVATION MAPS ==========
                # Create subdirectories for individual activation maps
                out_dir_argmax_individual = os.path.join(out_dir_argmax, "individual_maps")
                out_dir_agg_individual = os.path.join(out_dir_agg, "individual_maps")
                os.makedirs(out_dir_argmax_individual, exist_ok=True)
                os.makedirs(out_dir_agg_individual, exist_ok=True)

                # For ARGMAX method: save individual activation maps per attribute
                # We use the original saliency_maps (all attributes) [B, A, H, W]
                save_individual_activation_maps(
                    inputs, saliency_maps, attribute_names,
                    loader.dataset.map_part_to_attr_loc_acc,
                    source_paths,
                    batch_idx=batch_idx,
                    t_mean=transform_mean, t_std=transform_std,
                    save_path=out_dir_argmax_individual
                )

                # For AGGREGATED method: save the aggregated heatmaps per body part
                # heatmaps_agg has shape [B, K, H, W] where K is number of parts
                save_aggregated_activation_maps(
                    inputs, heatmaps_agg,
                    list(loader.dataset.part_dict.values()),
                    source_paths,
                    batch_idx=batch_idx,
                    t_mean=transform_mean, t_std=transform_std,
                    save_path=out_dir_agg_individual
                )

    # ========== COMPUTE STATISTICS FOR BOTH METHODS ==========
    print("\n" + "="*60)
    print("COMPARISON: ARGMAX vs AGGREGATED LOCALIZATION METHODS")
    print("="*60)

    print("\n" + "-"*60)
    print("METHOD 1: ARGMAX (Select highest activated attribute per part)")
    print("-"*60)
    res_argmax, mean_dist_argmax = calculate_average_partwise_localization_distance(
        loc_acc_collector_argmax, MAP_RESULT_GROUPS_TO_CUB_GROUPS, verbose=True
    )

    print("\n" + "-"*60)
    print("METHOD 2: AGGREGATED (Sum all attention maps per part)")
    print("-"*60)
    res_agg, mean_dist_agg = calculate_average_partwise_localization_distance(
        loc_acc_collector_aggregated, MAP_RESULT_GROUPS_TO_CUB_GROUPS, verbose=True
    )

    # Print comparison summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Mean Localization Distance (Argmax):     {mean_dist_argmax:.4f}")
    print(f"Mean Localization Distance (Aggregated): {mean_dist_agg:.4f}")
    print(f"Improvement: {((mean_dist_argmax - mean_dist_agg) / mean_dist_argmax * 100):.2f}%")
    print("="*60)

    # Compute mean classification and attribute accuracies and print
    print("\n--------- ACCURACIES ---------\n")
    print(f"Mean Classification Accuracy: {class_acc_meter.avg.item():.4f}")
    print(f"Mean Attribute Accuracy: {attr_acc_meter.avg.item():.4f}")


if __name__ == '__main__':
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    args = gather_args()

    # Create out folder for any visualizations / eval outputs
    out_folder_path = os.path.join(args.log_dir, f"comparison_{args.saliency_method}")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    # Print everything into separate file
    path_to_output_txt = os.path.join(args.out_dir_part_seg, "comparison_eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")
    sys.stdout = open(path_to_output_txt, 'a')

    # Print all args
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Run main evaluation with comparison
    eval(args)
