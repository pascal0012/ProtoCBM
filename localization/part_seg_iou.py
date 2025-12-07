import math
from typing import List
import torch
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils_protocbm.mappings import PART_SEG_GROUPS
from utils_protocbm.index_translation import map_attribute_ids_to_part_seg_group_id, get_attribute_names


def compute_soft_iou(att: torch.Tensor, gt: torch.Tensor, eps = 1e-7):
    """
        Computes the Intersection-over-Union between the attention masks and ground-truth
        segmentation masks, per attribute / part. Soft version, so the attention maps are
        raw and not binarized.

        Args:
            att: [B, A, H, W] in [0,1].
            gt : [B, A, H, W] in binary.
        Returns:
            soft_iou: The soft IoU score, per attribute and batch [B, A]
    """
    attf = att.float()
    gtf = gt.float()
    inter = (attf * gtf).sum(dim=(2,3))
    union = attf.sum(dim=(2,3)) + gtf.sum(dim=(2,3)) - inter
    return (inter + eps) / (union + eps)


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps = 1e-7):
    """
        Computes the Intersection-over-Union between the attention masks and ground-truth
        segmentation masks, per attribute / part. Hard version, so predicted masks must
        be binarized.

        Args:
            att: [B, A, H, W] in binary.
            gt : [B, A, H, W] in binary.
        Returns:
            iou: The IoU score, per attribute and batch [B, A]
    """
    pred = pred_mask.float()
    gt = gt_mask.float()
    inter = (pred * gt).sum(dim=(2,3))
    union = (pred + gt - pred*gt).sum(dim=(2,3))
    return (inter + eps) / (union + eps)


def percentile_threshold_maps(att: torch.Tensor, keep_ratio: float = 0.5):
    """
        Thresholds the attention maps to produce binary masks, with the threshold being
        computed dynamically per map to maintain keep_ratio% of the top activations for each map.

        Args:
            att: [B, A, H, W] in [0,1].
            keep_ratio: The ratio of top activations to keep, defaults to 0.5
        Returns:
            mask: Binarized attention masks [B, A, H, W].
    """
    B,A,H,W = att.shape
    N = H*W
    q = 1.0 - keep_ratio
    k = int(math.ceil(q * N))
    k = min(max(1, k), N)
    flat = att.reshape(B*A, N)
    kth_vals, _ = torch.kthvalue(flat, k, dim=1)   # k-th smallest
    thresh = kth_vals.view(B, A, 1, 1)
    mask = att > thresh
    return mask


def get_presence_mask(gt: torch.Tensor):
    """
        Given a ground truth tensor, this fct returns a boolean mask indicating the attribute
        presence for each [B, A] pair, as the ground truth segmentations for some images might
        exist and should thus be accounted for to not distort IoU computation.

        Args:
            gt: [B, A, H, W] binary ground truth segmentation mask per attribute
        Returns:
            mask: [B, A] boolean mask indicating concept presence
    """
    return gt.view(gt.size(0), gt.size(1), -1).any(dim=2)


def create_mapping_attributes_to_part_seg_group(data_dir, device, only_cbm_attributes=True):
    """
        Creates a mapping tensor from each attribute ID in range [0, n_attributes) to its respective part segmentation group,
        as defined in utils/mappings.py and utils/index_translation.py. 
        
        IMPORTANT: For any attributes that do NOT have a matching part group segmentation (e.g. "has_shape", all defined in utils
        as mentioned above), it will POP it from the mapping tensor and names of attributes.
        
        Thus, the names of the REMAINING and thus USED attributes is returned, as well.

        Args:
            data_dir: The path to the dataset root
            device: The device we are on
            only_cbm_attributes: Whether we only utilize the subset of attributes as selected by CBM.
        Returns:
            map_attr_id_to_part_seg_group: The mapping tensor of shape [A_hat], where A_hat = #num_kept_attributes, its respective value is the part seg group IDX
            attribute_names: The names of kept attributes
            unmatched_attr_id_mask: Mask that can be applied alongside the attribute dimension to remove unused parts
    """
    # This is our final tensor that maps each attribute by its id in the attention map to the respective part seg group
    map_attr_id_to_part_seg_group, unmatched_attr_id_mask = map_attribute_ids_to_part_seg_group_id(only_cbm_attributes=only_cbm_attributes)
    map_attr_id_to_part_seg_group = map_attr_id_to_part_seg_group.to(device=device)
    unmatched_attr_id_mask = unmatched_attr_id_mask.to(device=device)

    # Get attribute names, remove unmatched attributes from it
    attribute_names = get_attribute_names(data_dir, only_cbm_attributes=only_cbm_attributes)
    unmatched_mask_list = unmatched_attr_id_mask.cpu().detach().tolist()
    attribute_names = [name for name, keep in zip(attribute_names, unmatched_mask_list) if keep]
    return map_attr_id_to_part_seg_group, attribute_names, unmatched_attr_id_mask


def compute_IoU_to_seg_masks(saliency_maps: torch.Tensor, part_seg_masks: torch.Tensor, map_attr_id_to_part_seg_group: torch.Tensor, hard_iou=True):
    """
        Given a saliency map per attribute of any model and the corresponding part segmentation mask for that attribute, we compute the IoU
        between them. The saliency map does not have to have the same shape as the part segmentation masks, this method resizes them to
        the same size.

        Args:
            saliency_maps: The saliency maps for each attribute generated for our model output of shape [B, A, hs, ws]
            part_seg_maps: The corresponding segmentation maps, IMPORTANT: Per part seg group (G), is matched to attributes. [B, G, H, W], where H,W = img size
            map_attr_id_to_part_seg_group: The mapping of attribute IDs to the corresponding part segmentation group
            hard_iou: Whether IoU is to be computed on binarized saliency maps (True) or not (False)
        Returns:
            iou_sum_per_attr: The sum of IoU per attribute, over all batches [A]
            iou_count_per_attr: The count of valid IoU entries per attribute, over all batches [A]
            saliency_maps_upsampled: The saliency maps, upsampled to part segmentation shape [B, A, H, W]
            seg_masks_per_attribute: The segmentation masks, with each attribute being matched its proper part [B, A, H, W]
    """
    
    # For each of the attributes saliency map, get its respective group (and ID) and retrieve the segmentation mask for that group.
    _, _, H, W = part_seg_masks.shape
    group_idx = map_attr_id_to_part_seg_group.unsqueeze(0).expand(saliency_maps.shape[0], -1)  # Adapt to current batch size [B, A]
    group_idx = group_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Adapt to part segmentation size [B,A,H,W]

    # Gather segmentation mask for each attribute
    seg_masks_per_attribute = torch.gather(part_seg_masks, 1, group_idx)         # [B,A,H,W]

    # Get binary mask to denote for each image which attribute segmentations exist and which dont
    gt_presence_mask = get_presence_mask(seg_masks_per_attribute)   # [B, A]

    # Upsample saliency maps to segmentation mask shape, bilinear for smooth --> maybe choose other?
    saliency_maps_upsampled = F.interpolate(saliency_maps, size=(H, W), mode='bilinear', align_corners=False)

    # Compute hard IoU on binarized attention masks
    if hard_iou:
        binarized_saliency_maps = percentile_threshold_maps(saliency_maps).float()
        binarized_saliency_maps_upsampled = F.interpolate(binarized_saliency_maps, size=(H, W), mode='nearest')
        iou_per_attr = compute_iou(binarized_saliency_maps_upsampled, seg_masks_per_attribute)  # [B, A]
    # Compute soft IoU
    else:
        iou_per_attr = compute_soft_iou(saliency_maps_upsampled, seg_masks_per_attribute) # [B, A]

    # Collect sum and count IoU for each attribute, mask out invalid entries
    iou_per_attr[~gt_presence_mask] = 0 
    iou_sum_per_attr = iou_per_attr.sum(dim=0)  # Sum per attribute
    iou_count_per_attr = gt_presence_mask.sum(dim=0)  # Count valid entries
    return iou_sum_per_attr, iou_count_per_attr, saliency_maps_upsampled, seg_masks_per_attribute


def compute_mIoU_statistics(
        iou_sum_per_attr: torch.Tensor,
        iou_count_per_attr: torch.Tensor,
        attribute_names: List[str],
        map_attr_id_to_part_seg_group: torch.Tensor,
) -> None:
    """
        Computes and prints mIoU statistics: 1) globally, 2) per attribute, 3) per part segmentation group.
    """
    # Compute global mIoU and attribute-wise mIoU
    miou_per_attr = iou_sum_per_attr / (iou_count_per_attr + 1e-7)
    miou_global = iou_sum_per_attr.sum() / iou_count_per_attr.sum()

    print("\n--------- ATTRIBUTE-WISE mIoU ---------\n")
    for attr_id in range(miou_per_attr.shape[0]):
        print(f"Attr {attribute_names[attr_id]} - mIoU: {miou_per_attr[attr_id].item():.4f}")

    print("\n--------- PARTSEG-GROUP-WISE mIoU ---------\n")
    num_groups = map_attr_id_to_part_seg_group.max().item() + 1
    miou_group_sum = torch.zeros(num_groups, device=miou_per_attr.device)
    miou_group_count = torch.zeros(num_groups, device=miou_per_attr.device)

    # Accumulate scores into groups
    for attr_id in range(len(miou_per_attr)):
        g = map_attr_id_to_part_seg_group[attr_id].item()
        miou_group_sum[g] += miou_per_attr[attr_id]
        miou_group_count[g] += 1
    miou_per_group = miou_group_sum / (miou_group_count + 1e-7)

    for group_id in range(miou_per_group.shape[0]):
        print(f"Group {PART_SEG_GROUPS[group_id]} - mIoU: {miou_per_group[group_id].item():.4f}")
    
    print(f"Mean mIoU: {miou_global.item():.4f}")
