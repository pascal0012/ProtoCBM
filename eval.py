"""
Evaluate trained models on the official CUB test set
"""

import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
import yaml

from cub.dataset import CUBLocalizationDataset
from cub.config import BASE_DIR
from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics, create_mapping_attributes_to_part_seg_group
from localization.visualise import visualise_part_segmentations
from localization.localization_accuracy import calculate_average_partwise_localization_accuracy, compute_localization_accuracy, create_part_attribute_mapping_tensor
from models.apn_baseline import load_apn_baseline
from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils.mappings import MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS
from utils.eval_utils import get_eval_transform_for_model
from utils.train_utils import prepare_model, model_by_mode


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

    print("ALLO")
    # Create the model and load weights
    model = create_model(args)
    model = torch.load(os.path.join(args.log_dir, "best_model_1.pth"))
    model, device = prepare_model(model)
    model.eval()
    print(model)

    transform, transform_mean, transform_std, img_size = get_eval_transform_for_model(model, args)
    # TODO: APN uses 312 attributes, ours only 112, so adjust the code to work for both

    # Data management
    pkl_path = os.path.join(BASE_DIR, args.split_dir)
    data_dir = os.path.join(BASE_DIR, args.data_dir)
    print(pkl_path)
    dataset = CUBLocalizationDataset(pkl_path, data_dir, img_size, transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,  
        pin_memory=True, 
        persistent_workers=True,
    )

    # Creates a mapping tensor of attribute ids to their respective part segmentation group, removes unmatched entries, returns kept attribute names
    map_attr_id_to_part_seg_group, attribute_names = create_mapping_attributes_to_part_seg_group(args.image_dir, device)

    # Create a mapping tensor for the localization accuracy as well
    loc_acc_collector = []
    map_part_to_attr_loc_acc = create_part_attribute_mapping_tensor(MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS)

    # Collecting IoU values across batches for proper mean
    iou_sum_per_attr = torch.zeros(len(attribute_names), device=device)
    iou_count_per_attr = torch.zeros(len(attribute_names), device=device)

    with torch.no_grad():
        for data_idx, data in enumerate(loader):

            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks, part_bbs = data
            attr_labels = torch.stack(attr_labels).t()  # N x A

            # Pass through model, get model prediction and saliency map
            output = model(inputs.to(device))
            pred, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(output, args)

            # Compute localization accuracy and collect into our collector
            compute_localization_accuracy(
                scores,
                saliency_maps,
                part_bbs,
                dataset.part_dict,
                map_part_to_attr_loc_acc,
                loc_acc_collector,
                img_size=img_size
            )

            # Compute IoU between part segmentation masks and our saliency maps, for each attribute
            spr, cpr, saliency_maps_upsampled, seg_masks_per_attribute = compute_IoU_to_seg_masks(saliency_maps, part_seg_masks)
            iou_sum_per_attr += spr
            iou_count_per_attr += cpr

            # Visualise part segmentations with saliency
            if data_idx % args.vis_every_n == 0:
                visualise_part_segmentations(
                    inputs, saliency_maps_upsampled, seg_masks_per_attribute, attribute_names,
                    data_idx, t_mean=transform_mean, t_std=transform_std, save_path=args.out_dir_part_seg
                )

    # Compute mIoU statistics for part segmentations
    compute_mIoU_statistics(iou_sum_per_attr, iou_count_per_attr, attribute_names, map_attr_id_to_part_seg_group)

    # Compute part localization statistics, considering our grouping of attributes to groups.
    calculate_average_partwise_localization_accuracy(loc_acc_collector, MAP_PART_SEG_GROUPS_TO_CUB_GROUPS, IoU_thr=args["IoU_threshold"])


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/eval_protocbm.yaml",
        help="Path to evaluation config file (YAML)",
    )
    cli_args = parser.parse_args()

    # Load the config yaml
    with open(cli_args.config) as f:
        args = yaml.safe_load(f)

    # Add run name, keep as namespace to be able to access like args.param
    args = argparse.Namespace(**args, config_path=cli_args.config)

    # Create out folder for the part segmentation visualization
    out_folder_path = os.path.join(args.log_dir, "part_seg_vis")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    # Create .txt for output of this script, write everything to there
    log_file = os.path.join(args.log_dir, "eval.txt")
    sys.stdout = open(log_file, "w")

    # Run main evaluation
    eval(args)
