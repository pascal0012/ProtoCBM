from argparse import Namespace
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List

from cub.config import BASE_DIR
from cub.dataset import CUBLocalizationDataset
from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics


def get_eval_transform_for_model(model: nn.Module, args: Namespace):
    """
        Creates the proper torchvision.Transform given the used model. Also creates the transform
        for the part segmentation masks, as they must match the spatial transforms like Cropping,
        but not intensity ones (e.g. Normalize, Jitter).
    """
    if args.model_name == "cbm" or args.model_name == "protocbm":

        if "dino" in args.backbone:
            transform_mean = (0.485, 0.456, 0.406)
            transform_std = (0.229, 0.224, 0.225)
            img_size = model.backbone.image_size
            transform = transforms.Compose([
                transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size), # Resolution
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(
                    mean = transform_mean,
                    std = transform_std)
            ])
            mask_transform = transforms.Compose([
                transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(img_size),         
                transforms.ToTensor(),                          
                lambda t: (t > 0.5).float()  # binarize
            ])

        if args.backbone == "inception":
            transform_mean = (0.5, 0.5, 0.5)
            transform_std = (2, 2, 2)
            img_size = model.backbone.image_size
            transform = transforms.Compose([
                transforms.CenterCrop(img_size), # Resolution
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(
                    mean = transform_mean,
                    std = transform_std)
            ])
            mask_transform = transforms.Compose([
                transforms.CenterCrop(img_size),         
                transforms.ToTensor(),                          
                lambda t: (t > 0.5).float()  # binarize
            ])
    elif args.model_name == "apn":
        transform_mean = (0.485, 0.456, 0.406)
        transform_std = (0.229, 0.224, 0.225)
        img_size = 224
        transform= transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std
            ), 
        ])
        mask_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),                       
            lambda t: (t > 0.5).float()  # binarize
        ])
    else:
        raise ValueError(f"Invalid model name {args.model_name} encountered in get_eval_preprocess_for_model.")
    return transform, mask_transform, transform_mean, transform_std, img_size


def get_localization_loader(model: nn.Module, data_dir: str, split_dir: str, args: Namespace):

    # Get the respective transforms for this model
    transform, mask_transform, transform_mean, transform_std, img_size = get_eval_transform_for_model(model, args)

    # Data management
    pkl_path = os.path.join(BASE_DIR, split_dir)
    data_dir = os.path.join(BASE_DIR, data_dir)
    dataset = CUBLocalizationDataset(pkl_path, data_dir, img_size, transform, mask_transform, cbm_attributes="cbm" in args.model_name)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,  
        pin_memory=True, 
        persistent_workers=True,
    )
    return loader, transform_mean, transform_std, img_size


class LocalizationMeter():
    def __init__(self, attribute_names, device):
        self.attribute_names = attribute_names
        self.iou_sum_per_attr = torch.zeros(len(attribute_names), device=device)
        self.iou_count_per_attr = torch.zeros(len(attribute_names), device=device)
    
    def update(self, sum_per_attr, count_per_attr):
        self.iou_sum_per_attr += sum_per_attr
        self.iou_count_per_attr += count_per_attr
    
    def compute(self, map_attr_id_to_part_seg_group, verbose=False):
        return compute_mIoU_statistics(
            self.iou_sum_per_attr, self.iou_count_per_attr, self.attribute_names, map_attr_id_to_part_seg_group, verbose=verbose
        )


def eval_part_segmentation_iou(
    saliency_maps: torch.Tensor,
    part_seg_masks: torch.Tensor,
    map_attr_id_to_part_seg_group: torch.Tensor,
    unmatched_attr_mask: torch.Tensor,
):
    # Compute IoU between part segmentation masks and our saliency maps, for each attribute
    # Map out unmapped attributes from the saliency mask
    _, spr, cpr, _, _ = compute_IoU_to_seg_masks(
        saliency_maps[:, unmatched_attr_mask], part_seg_masks, map_attr_id_to_part_seg_group
    )
    return spr, cpr
