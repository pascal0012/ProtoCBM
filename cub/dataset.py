"""
    Contains the code regarding datasets and data loading
"""
import pickle
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
        no_img: bool,  #! -> wollen wir das direkt auf mode mappen? 
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


class CUBLocalizationDataset(Dataset):

    def __init__(self, pkl_path, data_dir, img_size, transform, mask_transform):
        """
        Args:
            pkl_path: Full path to test or validation pkl
            data_dir: Full path to the data directory of the CUB dataset, should point to the root.
            img_size: The image size, assumes quadratic images
            transform: Transform to apply to the image.
            mask_transform: The transformation to apply to the segmentation masks. The spatial transforms must match that of the image transform.
        """

        # Load pickled data
        self.data = []
        assert "test" in pkl_path or "val" in pkl_path
        self.data = pickle.load(open(pkl_path, 'rb'))

        # Construct paths
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.part_seg_dir = os.path.join(data_dir, "part_segmentations")

        # Create / store transforms
        self.transform = transform
        self.img_size = img_size
        self.mask_transform = mask_transform

        # Paths for localization accuracy, and create necessary dictionaries 
        self.imgID_imgName_mapping_path = os.path.join(self.data_dir, "images.txt")
        self.parts_locs_path = os.path.join(self.data_dir, "parts", "part_locs.txt") 
        self.parts_mapping_path = os.path.join(self.data_dir, "parts", "parts.txt")
        self.bird_BB_path = os.path.join(self.data_dir, "bounding_boxes.txt")
        self._create_localization_accuracy_dicts()

        #done like this in case we dont hardcode transformations later
        self.center_crop_size = None
        self.resize_size = None

        for t in transform.transforms:
            if isinstance(t, transforms.CenterCrop):
                self.center_crop_size = t.size
            if isinstance(t, transforms.Resize):
                self.resize_size = t.size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        
        # Trim unnecessary paths, load image
        path_parts = img_path.split('/')
        cub_index = path_parts.index("images")
        img_source_path = os.path.join(self.image_dir , "/".join(path_parts[cub_index+1:]))
        img = Image.open(img_source_path).convert('RGB')

        og_img_w, og_img_h = img.size

        if self.transform:
            img = self.transform(img)

        # Get class and attribute labels
        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        # Get localization data, i.e. part segmentation masks and bounding box data
        part_seg_masks = self._get_part_seg_masks(img_path, class_label)
        part_bbs = self._get_bounding_box_data(img_source_path, (og_img_w, og_img_h))

        #uncomment this for resize adjustment
        if self.resize_size != None:
            #adjust boxes to resizing
            part_bbs = self.resize_bounding_boxes(part_bbs, [og_img_w, og_img_h], self.resize_size if isinstance(self.resize_size, list) else [self.resize_size, self.resize_size])
            og_img_w, og_img_h = self.resize_size if isinstance(self.resize_size, list) else [self.resize_size, self.resize_size]

        if self.center_crop_size != None:
            #we have center crop adjust bounding boxes
            part_bbs = self._adjust_to_center_crop(part_bbs, og_img_w, og_img_h, self.center_crop_size)
        
        return img, class_label, attr_label, part_seg_masks, part_bbs

    def resize_bounding_boxes(self, box_tensor, og_size, new_size):
        #adjusts the BBs to resize transform
        # og_size, new_size -> W, H

        scale_x = new_size[0] / og_size[0]
        scale_y = new_size[1] / og_size[1]

        boxes = box_tensor.clone().float()

        # apply scaling only to non-zero boxes
        boxes[:, 0] *= scale_x  # x1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 3] *= scale_y  # y2

        return boxes.round().short()


    def _adjust_to_center_crop(self, bounding_boxes, og_w, og_h, crop_size):
        #adjusts bounding box coords to acknowledge center crop transformation
        #bounding boxes is the [K, 4] tensor that comes from the bounding box creation in get_item
        
        left = (og_w - crop_size[0]) / 2
        top  = (og_h - crop_size[1]) / 2

        bounding_boxes = bounding_boxes.float()

        orig_center_x = (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2
        orig_center_y = (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2

        visible_mask = (
            (orig_center_x >= left) & (orig_center_x <= left + crop_size[0]) &
            (orig_center_y >= top)  & (orig_center_y <= top + crop_size[1])
        )

        # Shift all boxes in place
        bounding_boxes[:, 0].sub_(left)  # xmin
        bounding_boxes[:, 1].sub_(top)   # ymin
        bounding_boxes[:, 2].sub_(left)  # xmax
        bounding_boxes[:, 3].sub_(top)   # ymax

        # Clip in place
        bounding_boxes[:, 0].clamp_(0, crop_size[0])
        bounding_boxes[:, 1].clamp_(0, crop_size[1])
        bounding_boxes[:, 2].clamp_(0, crop_size[0])
        bounding_boxes[:, 3].clamp_(0, crop_size[1])

        bounding_boxes[~visible_mask] = 0

        bounding_boxes = bounding_boxes.round().short()

        return bounding_boxes
    
    def _get_part_seg_masks(self, img_path, class_label):

        # Only for 70 classes, the part segmentations exist, for any other classes, the masks will be zero-filled 
        # (--> class_label > 70), +1 as it is zero-indexed
        if class_label + 1 > 70:
            return torch.zeros(len(PART_SEG_GROUPS), self.img_size, self.img_size, dtype=torch.float32)

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

        return torch.cat(part_lst, dim=0)  # (num_parts, H, W)

    def _load_mask(self, path_to_mask):
        # Not all parts are segmented for each img (e.g. occlusion), fill with zeros
        if not os.path.exists(path_to_mask):
            return torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        # If it exists: get part segmentation mask for this image / part pair
        mask = Image.open(path_to_mask).convert("L")  # 'L' = grayscale

        # Apply transformations to it (center crop like image, then binarize and add dummy channel dim)
        return self.mask_transform(mask)
    
    def _get_bounding_box_data(self, img_path, img_size):
        # Get image id [class path is also in mapping name]
        img_name = os.sep.join(img_path.split(os.sep)[-2:])
        img_id = self.imgName_to_imgID[img_name]

        # Get location and visibility infos of parts of image
        part_infos = self.img_id_to_part_locs[img_id]

        # Get bird bb, fun fact readme says it is x, y, width, height, but APN later wants x1, y1, x2, y2
        bird_bb = self.imgID_to_birdBB[img_id]

        # For each part get the gt bounding box, stack to one tensor
        return torch.stack(self._get_BB_per_part(bird_bb, part_infos, img_size), dim=0) # [K, 4]
    
    def _get_BB_per_part(self, bird_bb, part_infos, img_size, scale=4):
        # Generate a bounding box mask per part, empty if part is not visible
        # BB format returned is (x1, y1, x2, y2)
        width = bird_bb[2]
        height = bird_bb[3]

        mask_w = width / scale
        mask_h = height / scale

        part_masks = []
        for info in part_infos:
            #info: part_id, x1, y1, visible
            if info[-1] == 0.0: #part not visible, no gt
                part_masks.append(torch.zeros(4, dtype=torch.int16)) # Must fill with dummy values for batching
                continue
            gt = (info[1], info[2])
            #change from x, y, w, h to x1, y1, x2, y2
            transformed_bird_BB = [bird_bb[0], bird_bb[1], bird_bb[0] + bird_bb[2], bird_bb[1] + bird_bb[3]]
            part_masks.append(torch.tensor(self.get_KP_BB(gt, mask_h, mask_w, transformed_bird_BB, img_size), dtype=torch.int16))
        
        return part_masks

    def get_KP_BB(self, gt_point, mask_h, mask_w, bird_BB, img_size, KNOW_BIRD_BB=False):
        KP_best_x, KP_best_y = gt_point[0], gt_point[1]
        KP_x1 = KP_best_x - int(mask_w / 2)
        KP_x2 = KP_best_x + int(mask_w / 2)
        KP_y1 = KP_best_y - int(mask_h / 2)
        KP_y2 = KP_best_y + int(mask_h / 2)
        if KNOW_BIRD_BB:
            Bound = bird_BB
        else:
            Bound = [0, 0, img_size[0], img_size[1]]
        if KP_x1 < Bound[0]:
            KP_x1, KP_x2 = Bound[0], Bound[0] + mask_w
        elif KP_x2 > Bound[2]:
            KP_x1, KP_x2 = Bound[2] - mask_w, Bound[2]
        if KP_y1 < Bound[1]:
            KP_y1, KP_y2 = Bound[1], Bound[1] + mask_h
        elif KP_y2 > Bound[3]:
            KP_y1, KP_y2 = Bound[3] - mask_h, Bound[3]
        return [KP_x1, KP_y1, KP_x2, KP_y2]#{'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}

    def _create_localization_accuracy_dicts(self):
        # Maps part ID to part name
        self.part_dict = {}
        with open(self.parts_mapping_path) as ps:
            for line in ps:
                line = line.strip()
                id, part_name = line.split(" ", maxsplit=1)
                self.part_dict[int(id)] = part_name
                
        # Maps image id to image name
        # read here later into dictionary
        self.imgName_to_imgID = {}
        with open(self.imgID_imgName_mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                id_str, *text_parts = line.split()
                key = " ".join(text_parts)
                self.imgName_to_imgID[key] = int(id_str)

        # Information about parts bounding boxes and in-image occurence
        self.img_id_to_part_locs = {}
        with open(self.parts_locs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                id_str, *info = line.split()
                
                info = [float(x) for x in info]

                if int(id_str) not in self.img_id_to_part_locs:
                    self.img_id_to_part_locs[int(id_str)] = [info]
                else:
                    self.img_id_to_part_locs[int(id_str)] = self.img_id_to_part_locs[int(id_str)] + [info]

        # Information about bird bounding boxes per image id
        self.imgID_to_birdBB = {}
        with open(self.bird_BB_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                id_str, *bb_info = line.split()

                bb_info = [float(x) for x in bb_info]
                
                self.imgID_to_birdBB[int(id_str)] = bb_info


# TODO: load pkl from path + corresponding file name
def load_data(args, split: Literal["train", "val", "test"], resol=299):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary

    Required Args:
        args.data_dir: directory where the pkl files are stored
        args.batch_size: batch size for data loader
        args.image_dir: directory where the images are stored
        args.mode: either "CY" or "AY"
    """
    # TODO: ONLY TEMP FIX
    pkl_paths = [os.path.join(BASE_DIR, args.data_dir, f"{split}.pkl")]
    
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])

    dataset = CUBDataset(
        pkl_paths, 
        args.mode == "CY", 
        args.image_dir, 
        transform
    )

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
        num_workers=2,  # Multi-process data loading
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
