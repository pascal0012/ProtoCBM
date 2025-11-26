from argparse import Namespace
import torch.nn as nn
from torchvision import transforms


def get_eval_transform_for_model(model: nn.Module, args: Namespace):
    """
        Creates the proper torchvision.Transform given the used model.
    """
    if args.model_name == "cbm" or args.model_name == "protocbm":
        transform_mean = 0.5
        transform_std = 2
        img_size = model.backbone.img_size
        transform = transforms.Compose([
            transforms.CenterCrop(img_size), # Resolution
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(
                mean = [transform_mean, transform_mean, transform_mean],
                std = [transform_std, transform_std, transform_std])
        ])
    elif args.model_name == "apn":
        transform_mean = 0.485
        transform_std = 0.229
        img_size = 224
        transform= transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[transform_mean, transform_mean, transform_mean],
                std=[transform_std, transform_std, transform_std]
            ), 
        ])
    else:
        raise ValueError(f"Invalid model name {args.model_name} encountered in get_eval_preprocess_for_model.")
    return transform, transform_mean, transform_std, img_size
