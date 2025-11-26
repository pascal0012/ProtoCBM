"""
Evaluate trained models on the official CUB test set
"""

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cub.dataset import CUBDatasetPartSegmentations
from cub.config import BASE_DIR, N_ATTRIBUTES
from saliency.saliency import get_saliency_map_and_prediction
from localization.part_seg_iou import compute_IoU_to_seg_masks
from utils.eval_utils import get_eval_transform_for_model
from utils.train_utils import prepare_model, model_by_mode
from utils.mappings import PART_SEG_GROUPS
from ProtoCBM.utils.index_translation import map_attribute_ids_to_part_seg_group_id, get_attribute_names



def create_model(args):
    if args.model_name == "protocbm":
        model = model_by_mode(args)
    elif args.model_name == "cbm":
        model = model_by_mode(args)
    elif args.model_name == "apn":
        model = create_apn_baseline(args)
    else:
        raise ValueError("")
    return model


def eval(args, model_str):
    """
        TODO
    """

    # Create the model and load weights
    model = create_model(args)
    model = torch.load(os.path.join(args.model_dir, "best_model_1.pth"))
    model = prepare_model(model)
    device = model.device
    model.eval()

    transform, transform_mean, transform_std, img_size = get_eval_transform_for_model(model, args)
    # TODO: APN uses 312 attributes, ours only 112, so adjust the code to work for both

    # Data management
    data_dir = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
    dataset = CUBDatasetPartSegmentations(data_dir, args.use_attr, args.image_dir, args.part_seg_dir, img_size, transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,  
        pin_memory=True, 
        persistent_workers=True,
    )

    # This is our final tensor that maps each attribute by its id in the attention map to the respective part seg group
    map_attr_id_to_part_seg_group, unmatched_attr_id_mask = map_attribute_ids_to_part_seg_group_id()
    map_attr_id_to_part_seg_group = map_attr_id_to_part_seg_group.to(device=device)
    unmatched_attr_id_mask = unmatched_attr_id_mask.to(device=device)

    # Get attribute names, remove unmatched attributes from it
    attribute_names = get_attribute_names("/".join(args.image_dir.split("/")[:-1]), used_attributes_only=True)
    unmatched_mask_list = unmatched_attr_id_mask.cpu().detach().tolist()
    attribute_names = [name for name, keep in zip(attribute_names, unmatched_mask_list) if keep]

    # Collecting IoU values across batches for proper mean
    iou_sum_per_attr = torch.zeros(len(attribute_names), device=device)
    iou_count_per_attr = torch.zeros(len(attribute_names), device=device)

    with torch.no_grad():
        for data_idx, data in enumerate(loader):

            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks = data
            attr_labels = torch.stack(attr_labels).t()  # N x A

            # Pass through model, get model prediction and saliency map
            output = model(inputs.to(device))
            saliency_maps = get_saliency_map_and_prediction(output, args)

            # Compute IoU between part segmentation masks and our saliency maps, for each attribute
            spr, cpr = compute_IoU_to_seg_masks(saliency_maps, part_seg_masks, )
            iou_sum_per_attr += spr
            iou_count_per_attr += cpr

            # Visualise
            if data_idx % args.vis_every_n == 0:
                visualise(
                    inputs, attention_maps_upsampled, seg_masks_per_attribute, attribute_names,
                    data_idx, t_mean=transform_mean, t_std=transform_std, save_path=args.out_folder_path
                )
        


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-part_seg_dir', default='', help='directory to the part segmentations')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')   
    parser.add_argument('-vis_every_n', type=int, default=15, help='Visualize a random example every vis_every_n batches')  
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16

    # Create out folder
    out_folder_path = os.path.join(args.log_dir, "part_seg_vis")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_folder_path = out_folder_path

    # Create .txt for output of this script, write everything to there
    log_file = os.path.join(out_folder_path, "eval.txt")
    sys.stdout = open(log_file, "w")

    print(args)
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        result = eval(args)

