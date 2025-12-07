"""
Evaluate trained models on the official CUB test set
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cub.dataset import CUBLocalizationDataset
from cub.config import BASE_DIR
from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics, create_mapping_attributes_to_part_seg_group
from localization.visualise import create_attribute_mosaic, visualise_part_segmentations, visualise_localization_acc_boxes
from localization.localization_accuracy import calculate_average_partwise_localization_accuracy, compute_localization_accuracy, create_part_attribute_mapping_tensor
from models.apn_baseline import load_apn_baseline
from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils_protocbm.mappings import MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS, CBM_SELECTED_CUB_ATTRIBUTE_IDS
from utils_protocbm.index_translation import map_attribute_ids_from_cub_to_cbm
from utils_protocbm.eval_utils import get_eval_transform_for_model
from utils_protocbm.train_utils import accuracy, prepare_model, model_by_mode, gather_args
from utils_protocbm.perf import Timer


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
    with Timer("Creating and preparing model"):
        model = create_model(args)
        model, device = prepare_model(model, args, load_weights=True)
        model.eval()

    transform, mask_transform, transform_mean, transform_std, img_size = get_eval_transform_for_model(model, args)

    # Data management
    pkl_path = os.path.join(BASE_DIR, args.split_dir)
    data_dir = os.path.join(BASE_DIR, args.data_dir)
    dataset = CUBLocalizationDataset(pkl_path, data_dir, img_size, transform, mask_transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,  
        pin_memory=True, 
        persistent_workers=True,
    )

    # Creates a mapping tensor of attribute ids to their respective part segmentation group removes unmatched entries, returns kept attribute names
    map_attr_id_to_part_seg_group, attribute_names, unmatched_attr_mask = create_mapping_attributes_to_part_seg_group(
        args.data_dir,
        device,
        only_cbm_attributes = "cbm" in args.model_name  # Keep all CUB attributes or use only those as used by CBM?
    )

    # Create a mapping tensor for the localization accuracy as well
    loc_acc_collector = []
    if "cbm" in args.model_name:
        map_parts_to_attributes = map_attribute_ids_from_cub_to_cbm(MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS)
    elif "apn" in args.model_name:
        map_parts_to_attributes = MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS
    else:
        raise Exception("Unhandled model name for mapping")

    map_part_to_attr_loc_acc = create_part_attribute_mapping_tensor(map_parts_to_attributes, device)

    # Collecting IoU and accuracy values across batches for proper mean
    iou_sum_per_attr = torch.zeros(len(attribute_names), device=device)
    iou_count_per_attr = torch.zeros(len(attribute_names), device=device)
    acc_sum = 0
    acc_count = 0

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Evaluating batches")):

            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data
            attr_labels = torch.stack(attr_labels).t()  # N x A

            # Pass through model, get model prediction and saliency map per attribute
            pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(model, inputs, args)
            saliency_maps = saliency_maps.to(device)
            
            # Compute classification accuracy
            acc_sum += accuracy(pred, labels, topk=(1,))[0]
            acc_count += 1
            
            # Compute localization accuracy and collect into our collector
            loc_optimal_masks_batch, _, loc_ious, loc_resized_heatmaps, max_scores_per_part = compute_localization_accuracy(
                                                                                                    scores,
                                                                                                    saliency_maps,
                                                                                                    part_bbs,
                                                                                                    dataset.part_dict,
                                                                                                    map_part_to_attr_loc_acc,
                                                                                                    loc_acc_collector,
                                                                                                    img_size=img_size
                                                                                                )

            # Compute IoU between part segmentation masks and our saliency maps, for each attribute
            # Map out unmapped attributes from the saliency mask
            spr, cpr, saliency_maps_upsampled, seg_masks_per_attribute = compute_IoU_to_seg_masks(
                saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group
            )
            iou_sum_per_attr += spr
            iou_count_per_attr += cpr

            #create_attribute_mosaic(inputs, saliency_maps, attribute_names, scores)
            #if data_idx > 12:
            #    break

            # Visualise part segmentations with saliency
            if args.vis_every_n > 0 and data_idx % args.vis_every_n == 0:
                visualise_part_segmentations(
                    inputs, saliency_maps_upsampled, seg_masks_per_attribute, attribute_names,
                    data_idx, t_mean=transform_mean, t_std=transform_std, save_path=args.out_dir_part_seg
                )

                visualise_localization_acc_boxes(
                    inputs,
                    source_paths,
                    part_gts,
                    loc_resized_heatmaps,
                    loc_optimal_masks_batch,
                    part_bbs,
                    max_scores_per_part,
                    data_idx,
                    loc_ious,
                    list(dataset.part_dict.values()),
		            t_mean=transform_mean,
                    t_std=transform_std,
                    save_path=args.out_dir_part_seg
                )

    # Compute mIoU statistics for part segmentations
    compute_mIoU_statistics(iou_sum_per_attr, iou_count_per_attr, attribute_names, map_attr_id_to_part_seg_group)

    # Compute part localization statistics, considering our grouping of attributes to groups.
    calculate_average_partwise_localization_accuracy(loc_acc_collector, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS, IoU_thr=args.IoU_threshold)

    # Compute mean classification accuracy and print
    mean_acc = acc_sum / (acc_count + 1e-7)
    print("\n--------- CLASSIFICATION ACCURACY ---------\n")
    print(f"Mean Classification Accuracy: {mean_acc.item():.4f}")


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
