from argparse import Namespace
import torch.nn as nn
from torchvision import transforms


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
