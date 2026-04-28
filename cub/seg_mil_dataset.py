"""Dataset for SEG-MIL-CBM training on cached segment bags.

Each cached file (one per CUB image) is a torch dict with:
    crops:         uint8 tensor [N_s, 3, H, W] (already resized to backbone input)
    valid:         bool tensor  [N_s]
    clip_concepts: float16 tensor [N_s, K]
    label:         int (class index 0..199)

The cache is produced offline by `scripts/seg_mil_cbm/preprocess_cub.py`.
Splits mirror the existing CUB pickle splits — we read image paths from the
same pickle files used by `cub.dataset.CUBDataset` and look up the cached
segment file by image id.
"""
from __future__ import annotations

import os
import pickle

import torch
from torch.utils.data import Dataset


# ImageNet stats (used for ResNet-50). The cache stores uint8 [0, 255] crops.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _img_id_from_path(img_path: str) -> str:
    # img_path like ".../images/022.Chuck_will_Widow/Chuck_Will_Widow_0059_796982.jpg"
    base = os.path.basename(img_path)
    return os.path.splitext(base)[0]

# todo: move to cub.dataset

class CUBSegMILDataset(Dataset):
    def __init__(self, pkl_file_paths: list[str], cache_dir: str):
        self.cache_dir = cache_dir
        all_entries = []
        for path in pkl_file_paths:
            with open(path, "rb") as f:
                all_entries.extend(pickle.load(f))
        self.entries = [
            e for e in all_entries
            if os.path.exists(os.path.join(
                cache_dir, f"{_img_id_from_path(e['img_path'])}.pt"))
        ]
        n_skipped = len(all_entries) - len(self.entries)
        if n_skipped > 0:
            print(f"[CUBSegMILDataset] {cache_dir}: using "
                  f"{len(self.entries)}/{len(all_entries)} entries "
                  f"({n_skipped} have no cached bag)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        item = self.entries[idx]
        img_id = _img_id_from_path(item["img_path"])
        cache_path = os.path.join(self.cache_dir, f"{img_id}.pt")
        cached = torch.load(cache_path, map_location="cpu")

        crops = cached["crops"].float() / 255.0          # [N_s, 3, H, W]
        crops = (crops - _IMAGENET_MEAN.squeeze(0)) / _IMAGENET_STD.squeeze(0)
        valid = cached["valid"].bool()                   # [N_s]
        clip_concepts = cached["clip_concepts"].float()  # [N_s, K]
        label = int(item["class_label"])
        return crops, valid, clip_concepts, label
