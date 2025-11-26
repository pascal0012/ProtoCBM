"""
General utils for training, evaluation and data loading
"""
import pickle
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.mappings import PART_SEG_GROUPS
from cub.config import BASE_DIR, N_ATTRIBUTES


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, 
        pkl_file_paths: list[str], 
        no_img: bool, 
        image_dir: str, 
        transform=None
    ):
        """
        CUBDataset constructor which reads all data from specified (train/val/test) pkl files.

        For the construction all pickel files are loaded and combined into one large list.

        Arguments:
            pkl_file_paths: list of full path to all the pkl data
            no_img: whether to load the images (e.g. False for A -> Y model)
            image_dir: default = 'images'. Will be append to the parent dir
            transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []

        # Train has to be loaded (val is optional)
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])

        # load the pickel files into one large list
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        
        self.transform = transform
        self.no_img = no_img
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Fetches the items based on the location of the pickl files.

        Since the pickel files are constant in our codebase we can safely
        extract the correct path from the predetermined img_paths of the pickel files
        The source path looks like this:
            '/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/
            images/022.Chuck_will_Widow/Chuck_Will_Widow_0059_796982.jpg'
        """
        img_data = self.data[idx]
        img_path = img_data['img_path']

        # load the image using correct path
        path_parts = img_path.split('/')
        cub_index = path_parts.index("images")
        img_source_path = os.path.join(self.image_dir , "/".join(path_parts[cub_index+1:]))
        img = Image.open(img_source_path).convert('RGB')
    
        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        attr_label = img_data['attribute_label']
        if self.no_img:
            return attr_label, class_label
        else:
            return img, class_label, attr_label


class CUBDatasetPartSegmentations(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, test_pkl_path, image_dir, part_seg_dir, resol, transform):
        """
        Arguments:
        test_pkl_path: full path to test pkl
        image_dir: default = 'images'. Will be append to the parent dir
        part_seg_dir: Path to part segmentation directory
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        assert "test" in test_pkl_path
        self.data = pickle.load(open(test_pkl_path, 'rb'))

        # Filter data to remove all classes for which no part segmentations exist (--> class_label > 70), +1 as it is zero-indexed
        self.data = [d for d in self.data if d.get("class_label") + 1 <= 70]

        self.transform = transform
        self.resol = resol
        self.image_dir = image_dir
        self.part_seg_dir = part_seg_dir

        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(299),         
            transforms.ToTensor(),                          
            lambda t: (t > 0.5).float()  # binarize
        ])

    def __len__(self):
        return len(self.data)

    def _load_mask(self, path_to_mask):
        # Not all parts are segmented for each img (e.g. occlusion), fill with zeros
        if not os.path.exists(path_to_mask):
            return torch.zeros(1, self.resol, self.resol, dtype=torch.float32)

        # If it exists: get part segmentation mask for this image / part pair
        mask = Image.open(path_to_mask).convert("L")  # 'L' = grayscale

        # Apply transformations to it (center crop like image, then binarize and add dummy channel dim)
        return self.mask_transform(mask)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        
        # Trim unnecessary paths
        path_parts = img_path.split('/')
        cub_index = path_parts.index("images")
        img_source_path = os.path.join(self.image_dir , "/".join(path_parts[cub_index+1:]))
        img = Image.open(img_source_path).convert('RGB')

        # Get the class number, e.g. 002 for Laysan_Albatross, strip initial numbers, also get img name w/out extension
        class_nr = str(int(img_path.split("/")[-2][:3]))
        img_name = img_path.split("/")[-1].split(".")[0]

        # Collect all part segmentation masks
        part_lst = []
        for part in PART_SEG_GROUPS:

            # IMPORTANT: For left / right distinctions, we add the masks to one.
            part_list = [part]
            if part in ["eye", "wing", "leg"]:
                part_list = [f"right_{part}", f"left_{part}"]

            tmp_masks = []
            for tmp_part in part_list:
                tmp_masks.append(
                    self._load_mask(
                        os.path.join(self.part_seg_dir, "AnnotationMasksPerclass", class_nr, f"{img_name}_{tmp_part}.png")
                    )
                )
            
            # Combine masks, if needed, and add to part_lst.
            if len(tmp_masks) == 1:
                combined_mask = tmp_masks[0]
            else:
                combined_mask = ((tmp_masks[0] > 0) | (tmp_masks[1] > 0)).to(tmp_masks[0].dtype)
            part_lst.append(combined_mask)

        part_seg_masks = torch.cat(part_lst, dim=0)  # (num_parts, H, W)

        class_label = img_data['class_label']

        attr_label = img_data['attribute_label']
        return img, class_label, attr_label, part_seg_masks


# TODO: load pkl from path + corresponding file name
def load_data(args, split: Literal["train", "val", "test"], resol=299):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    # TODO: ONLY TEMP FIX
    pkl_paths = [os.path.join(BASE_DIR, args.data_dir, f"{split}.pkl")]
    
    resized_resol = int(resol * 256/224)
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
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

    dataset = CUBDataset(pkl_paths, args.mode=="CY", args.image_dir, transform)
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=4,  # Multi-process data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
    )
    return loader

def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio
