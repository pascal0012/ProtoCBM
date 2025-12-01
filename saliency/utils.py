from utils.mappings import (
    MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS,
    CBM_SELECTED_CUB_ATTRIBUTE_IDS
)

from cub.config import BASE_DIR
from torch.utils.data import DataLoader, Dataset
import torch 
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

import torch.nn as nn


def part_to_model_index(part_name):
    """ Based on the part name, return the corresponding model indices (0-111).

    Loads the CUB part attribute list and maps it into the model index range
    using the predefined mappings.

    Args:
        part_name (str): Name of the part (e.g., 'eye', 'beak').

    Returns:
        list: List of corresponding model indices in the range 0-111.
    """
    if part_name not in MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS:
        raise ValueError(f"Part name '{part_name}' is not recognized.")
    
    attr_indices = MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS[part_name]
    model_indices = [attr_index_to_model_index(attr_idx) for attr_idx in attr_indices]

    return model_indices


def attr_index_to_model_index(attr_index):
    """
    Convert an attribute index (0-312) to the corresponding model index (0-111)
    for CBM-selected attributes.

    Args:
        attr_index (int): Attribute index in the range 0-312.

    Returns:
        int: Corresponding model index in the range 0-111.
    """
    if attr_index not in CBM_SELECTED_CUB_ATTRIBUTE_IDS:
        raise ValueError(f"Attribute index {attr_index} is not in the selected CBM attributes.")
    
    model_index = CBM_SELECTED_CUB_ATTRIBUTE_IDS.index(attr_index)
    return model_index


class CUBDBasicDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, 
        split: str = "train",  # "train" or "val"
        data_path: str = "../ProtoCBM/data/CUB_200_2011/",
        cub_attributes_list: list[int] = None,
        no_img: bool = False,
        transform=None,
    ):
        """
        CUBDataset constructor which processes CUB data.

        Arguments:
            split: "train" or "val" to specify which split to use
            data_path: path to CUB_200_2011 directory
            cub_attributes_list: list of selected attribute IDs. If None, uses all attributes
            no_img: whether to skip loading images
            transform: image transformation pipeline
        """
        self.no_img = no_img
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, "images")
        self.transform = load_transform(split) if transform is None else transform
        self.split = split
        
        # Load train/test split
        split_file = os.path.join(data_path, "train_test_split.txt")
        split_df = pd.read_csv(
            split_file,
            sep=" ",
            header=None,
            names=["image_id", "is_training"],
        )
        
        # Filter image_ids based on split
        if split == "train":
            image_ids = split_df[split_df["is_training"] == 1]["image_id"].tolist()
        elif split == "val":
            image_ids = split_df[split_df["is_training"] == 0]["image_id"].tolist()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        # Load image addresses
        image_addresses_file = os.path.join(data_path, "images.txt")
        self.image_addresses = pd.read_csv(
            image_addresses_file,
            sep=" ",
            header=None,
            names=["image_id", "image_path"],
        )
        
        # Load image attributes
        image_attributes_file = os.path.join(data_path, "attributes/image_attribute_labels.txt")
        self.image_attributes = pd.read_csv(
            image_attributes_file,
            sep=" ",
            header=None,
            names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        )
        
        # Load attribute names
        attributes_names_file = os.path.join(data_path, "attributes/attributes.txt")
        attributes_names_df = pd.read_csv(
            attributes_names_file,
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        )
        self.attributes_names = attributes_names_df.set_index("attribute_id")["attribute_name"].to_dict()
        
        # Set attribute list
        if cub_attributes_list is None:
            self.cub_attributes_list = sorted(self.image_attributes["attribute_id"].unique())
        else:
            self.cub_attributes_list = cub_attributes_list
        
        # Filter attributes
        self.image_attributes = self.image_attributes[
            (self.image_attributes["image_id"].isin(image_ids)) &
            (self.image_attributes["attribute_id"].isin(self.cub_attributes_list))
        ]
        self.image_addresses = self.image_addresses[
            self.image_addresses["image_id"].isin(image_ids)
        ]
        
        # Build dataset entries
        self.data = []
        for image_id in image_ids:
            img_attrs = self.image_attributes[self.image_attributes["image_id"] == image_id]
            img_path_row = self.image_addresses[self.image_addresses["image_id"] == image_id]
            
            if len(img_path_row) == 0:
                continue
                
            img_path = img_path_row["image_path"].values[0]
            
            # Create binary attribute vector
            attribute_labels = np.zeros(len(self.cub_attributes_list), dtype=np.float32)
            for _, row in img_attrs.iterrows():
                if row["is_present"] == 1:
                    attr_idx = self.cub_attributes_list.index(row["attribute_id"])
                    attribute_labels[attr_idx] = 1.0
            
            self.data.append({
                "img_path": img_path,
                "image_id": image_id,
                "attribute_labels": attribute_labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Fetches the items based on the dataset index."""
        img_data = self.data[idx]
        img_path = img_data['img_path']
        
        if self.no_img:
            img_tensor = None
        else:
            full_path = os.path.join(self.image_dir, img_path)
            img = Image.open(full_path).convert('RGB')
            img_tensor = self.transform(img)
        
        return {
            "image": img_tensor,
            "attributes": torch.tensor(img_data["attribute_labels"]),
            "image_id": img_data["image_id"],
            "img_path": img_path
        }
    
def load_transform(mode: str, resol: int = 299):
    if mode == "train":
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
        ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
        ])

    return transform


def find_fist_conv(model: nn.Module, n_last: int = 20):
    """
    Find the first convolutional layer in the model's state dictionary.

    Args:
        state_dict (dict): The state dictionary of the model.

    Returns:
        str: The name of the first convolutional layer.
    """

    layer = None
    for i in list(model.state_dict().keys())[-n_last:][::-1]:
        if i.endswith("conv.weight"):
            print(i)
            layer_name = i.replace(".weight", "")
            layer = model.get_submodule(layer_name)
            break

    if layer is None:
        raise ValueError("Could not find a convolutional layer in the model's state dictionary.")
    
    return layer




def norm_tensor(matrix: torch.Tensor):
    """ Normalize Grad-CAM matrix to [0, 1] range.

    Args:
        matrix (torch.Tensor): The Grad-CAM matrix to normalize.

    Returns:
        torch.Tensor: The normalized Grad-CAM matrix.
    """
    mat_min = matrix.min()
    mat_max = matrix.max()
    normalized = (matrix - mat_min) / (mat_max - mat_min + 1e-7)
    return normalized