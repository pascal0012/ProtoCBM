"""
Evaluate trained models on the official CUB test set
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
    visualize_keypoint_distances
)
from localization.localization_accuracy import (
    calculate_average_partwise_localization_accuracy, 
    compute_localization_accuracy, 
    calculate_average_partwise_localization_distance
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
        TODO
    """

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
    loc_acc_collector = []
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    thresholds = np.arange(0, 1.05, 0.05)
    threshold_ious = torch.zeros((len(thresholds), len(attribute_names)), device=device)
    threshold_counts = torch.zeros((len(thresholds), len(attribute_names)), device=device)

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Evaluating batches")):
            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data
            # attr_labels = torch.stack(attr_labels).t()  # N x A
            attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

            # Pass through model, get model prediction and saliency map per attribute
            pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(model, inputs, args)
            saliency_maps = saliency_maps.to(device)

            # Calculate classification accuracy
            class_acc = accuracy(pred, labels, topk=(1,)) 
            class_acc_meter.update(class_acc[0], pred.size(0))

            # Calculate attribute accuracy
            attr_acc = binary_accuracy(scores, attr_labels)
            attr_acc_meter.update(attr_acc, pred.size(0))
            
            # Compute localization accuracy and collect into our collector
            predicted_coords, dists, resized_heatmaps, max_scores_per_part = compute_localization_accuracy(
                                                                                                    scores,
                                                                                                    saliency_maps,
                                                                                                    part_bbs,
                                                                                                    part_gts,
                                                                                                    loader.dataset.part_dict,
                                                                                                    loader.dataset.map_part_to_attr_loc_acc,
                                                                                                    loc_acc_collector,
                                                                                                    img_size=img_size
                                                                                                    )

            # Compute IoU between part segmentation masks and our saliency maps, for each attribute
            # Map out unmapped attributes from the saliency mask
            iou_scores, spr, cpr, saliency_maps_upsampled, seg_masks_per_attribute = compute_IoU_to_seg_masks(
                saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group
            )
            seg_loc_meter.update(spr, cpr)

            if args.plot_curve:
                #--- compute curve stats ---
                for i, t in enumerate(thresholds):
                    # Compute IoU between part segmentation masks and our saliency maps, for each attribute
                    # Map out unmapped attributes from the saliency mask
                    _, tspr, tcpr, _, _ = compute_IoU_to_seg_masks(
                        saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group, keep_threshold=t
                    )

                    threshold_ious[i] += tspr
                    threshold_counts[i] += tcpr
                #------

            # Visualise part segmentations with saliency
            if args.vis_every_n > 0 and data_idx % args.vis_every_n == 0:
                visualise_part_segmentations(
                    inputs, saliency_maps_upsampled, seg_masks_per_attribute,
                    attribute_names, iou_scores, 
                    source_paths=source_paths, t_mean=transform_mean, t_std=transform_std, save_path=args.out_dir_part_seg
                )

                visualize_keypoint_distances(part_gts,
                                             inputs,
                                             source_paths,
                                             predicted_coords,
                                             dists,
                                             data_idx,
                                             list(loader.dataset.part_dict.values()),
                                             t_mean=transform_mean, 
                                             t_std=transform_std,
                                             save_path=args.out_dir_part_seg
                )

    # Compute statistics over all batches
    seg_loc_meter.compute(map_attr_id_to_part_seg_group, verbose=True)

    if args.plot_curve:
        #stats for curve
        threshold_accs = [compute_mIoU_statistics(threshold_ious[i], threshold_counts[i], attribute_names, map_attr_id_to_part_seg_group, verbose=False) for i in range(len(thresholds))]
        plot_threshold_curve(thresholds, threshold_accs, args.out_dir_part_seg)


    # Compute part localization statistics, considering our grouping of attributes to groups.
    #calculate_average_partwise_localization_accuracy(loc_acc_collector, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS, IoU_thr=args.IoU_threshold)
    calculate_average_partwise_localization_distance(loc_acc_collector, MAP_RESULT_GROUPS_TO_CUB_GROUPS)

    # Compute mean classification and attribute accuracies and print
    print("\n--------- ACCURACIES ---------\n")
    print(f"Mean Classification Accuracy: {class_acc_meter.avg.item():.4f}")
    print(f"Mean Attribute Accuracy: {attr_acc_meter.avg.item():.4f}")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True

    args = gather_args()

    # Create out folder for any visualizations / eval outputs
    out_folder_path = os.path.join(args.log_dir, f"visualization_{args.saliency_method}")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    # Print everything into separate file
    path_to_output_txt = os.path.join(args.out_dir_part_seg, "eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")
    sys.stdout = open(path_to_output_txt, 'a')

    # Print all args
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Run main evaluation
    eval(args)
