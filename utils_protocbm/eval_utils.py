from argparse import Namespace
from copy import copy
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from cub.config import BASE_DIR
from cub.dataset import CUBLocalizationDataset, SUBDataset
from localization.part_seg_iou import compute_IoU_to_seg_masks, compute_mIoU_statistics
from utils_protocbm.train_utils import create_model, prepare_model


def create_model_for_eval(args: Namespace) -> Tuple[nn.Module, str, Optional[nn.Module]]:
    """ Create and prepare model(s) for evaluation.

    For independent models (mode: independent), loads both:
      - XC model (xc_checkpoint): image -> concept scores + saliency
      - CY model (cy_checkpoint): concept scores -> class predictions
    For all other modes, loads the single joint model and returns cy_model=None.

    Returns:
        model:    The XC (or joint) model
        device:   The device string
        cy_model: The CY classifier model, or None for non-independent modes
    """
    if getattr(args, "mode", None) == "independent":
        xc_args = copy(args)
        xc_args.mode = "XC"
        xc_args.checkpoint = getattr(args, "xc_checkpoint", None)
        xc_model = create_model(xc_args)
        xc_model, device = prepare_model(xc_model, xc_args, load_weights=True)

        cy_args = copy(args)
        cy_args.mode = "CY"
        cy_args.checkpoint = getattr(args, "cy_checkpoint", None)
        cy_model = create_model(cy_args)
        cy_model, _ = prepare_model(cy_model, cy_args, load_weights=True)

        return xc_model, device, cy_model
    else:
        model = create_model(args)
        model, device = prepare_model(model, args, load_weights=True)
        return model, device, None


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
    dataset = CUBLocalizationDataset(pkl_path, data_dir, img_size, transform, mask_transform, cbm_attributes="cbm" in args.model_name, dataset=args.dataset)
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


def get_sub_loader(model: nn.Module, args: Namespace):
    """
    Get data loader for the SUB (Synthetic Attribute Substitutions) benchmark dataset.

    Args:
        model: The model to evaluate (used to determine image size and transforms)
        args: Namespace with sub_data_dir, batch_size, and other settings

    Returns:
        loader: DataLoader for SUB dataset
        transform_mean: Mean used for normalization
        transform_std: Std used for normalization
        img_size: Image size
        dataset: The SUB dataset object (for accessing metadata)
    """
    # Get transforms (reuse existing transform logic)
    transform, _, transform_mean, transform_std, img_size = get_eval_transform_for_model(model, args)

    # Get SUB data directory
    sub_data_dir = getattr(args, 'sub_data_dir', os.path.join(BASE_DIR, 'data/SUB'))
    sub_data_dir = os.path.join(BASE_DIR, sub_data_dir) if not os.path.isabs(sub_data_dir) else sub_data_dir
    sub_limit = getattr(args, 'sub_limit', None)

    # Create dataset
    dataset = SUBDataset(sub_data_dir, transform=transform, limit=sub_limit)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return loader, transform_mean, transform_std, img_size, dataset


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


def sub_to_cbm(name):
    return name.replace('--', '::')