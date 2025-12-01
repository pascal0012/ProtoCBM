#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


#TO-DO: CUB parts to Sebbis seg groups mapping einbauen
def compute_localization_accuracy(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor], 
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
):
    """
        This method.... TODO

        Args:
            pre_attri: TODO
            attention: TODO
            part_bounding_boxes: The bounding box per part, for each image, as given by its coordinates [B, K, 4]
            part_dict: TODO
            part_attribute_mapping: TODO
            collector_list: List to collect the mIoU results into.
            img_size: The image size.
        
        Returns:
            TODO
    """
    # part_attribute_mapping means a dict that maps from a part (CUB/parts/parts.txt) to a list of attribute IDs
    # sum of all attribute IDs must not be more that number of attention maps returned/attributes used by model. also is
    # required to be 0 indexed and correctly match the attribute order in the model
    #reimplementation of paper description

    B = attention.size(0)
    K = len(part_dict)

    #get argmax attribute per part
    argmax_per_part = [] #max index per part, -1 if part not present
    
    for part in part_dict.values():
        #take argmax of each part group
        subset = pre_attri[:, part_attribute_mapping_tensor[part]]
        argmax_in_subset = subset.argmax(dim=1)
        result = part_attribute_mapping_tensor[part][argmax_in_subset] #now res should be batchsize shape

        argmax_per_part.append(result)

    # Create a mask to track which part bounding boxes are non-empty
    valid_mask = (bounding_box_per_part.sum(dim=-1) != 0)  # [B, K]

    # Take heatmaps
    # TODO: What happens here, I think this is not correct and one should use torch.gather
    # H, W = attention.shape
    # heatmaps = attention.gather(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))
    idx = torch.stack(tuple(argmax_per_part)) #15xbatchsize
    idx = idx.t() 
    batch_indices = torch.arange(B).unsqueeze(1)
    heatmaps = attention[batch_indices, idx] #should be batchisizex15x8x8

    """
    # Resize with opencv because original code does that
    np_maps = heatmaps.cpu().numpy()
    resized_heatmaps = np.zeros((heatmaps.shape[0], heatmaps.shape[1], img_size, img_size), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            resized_heatmaps[b, k] = cv2.resize(
                np_maps[b, k],
                (img_size, img_size)
            )
    """
    # Resize heatmaps to image size
    resized_heatmaps = torch.nn.functional.interpolate(heatmaps, size=img_size, mode='bilinear', align_corners=False)

    # Compute sliding window from heatmaps
    optimal_masks_batch = compute_optimal_masks_per_mask(torch.from_numpy(resized_heatmaps), bounding_box_per_part.to(heatmaps.device))
    optimal_masks_batch[~valid_mask] = 0

    # Get the IoU between the bounding boxes and our heatmaps, per part over all images, set to -1 for non-existing parts
    ious = bbox_iou_tensor(optimal_masks_batch, bounding_box_per_part)
    valid_mask = ((optimal_masks_batch.sum(-1) > 0) & (bounding_box_per_part.sum(-1) > 0))
    ious[~valid_mask] = -1

    # Take computed IoUs per part and store them to dict for later calculation
    for i in range(ious.shape[0]):
        sub_res = dict(zip(list(part_dict.values()), ious[i].tolist()))
        collector_list.append(sub_res)


def create_part_attribute_mapping_tensor(part_attribute_mapping: Dict[str, List[int]]):
    part_attribute_mapping_tensor = {}
    for part, attr_list in part_attribute_mapping.items():
        part_attribute_mapping_tensor[part] = torch.tensor(attr_list).to("cuda" if torch.cuda.is_available() else "cpu")
    return part_attribute_mapping_tensor


def compute_optimal_masks_per_mask(heatmaps, part_bbs):
    """
    Computes optimal bounding box per heatmap using convolution with mask size.

    Args:
        heatmaps: torch.Tensor of shape (B, K, H, W)
        part_bbs: torch.Tensor of shape (B, K, 4)  containing [x1,y1,x2,y2] per part

    Returns:
        boxes: torch.Tensor of shape (B, K, 4) with [x1,y1,x2,y2] per part
    """

    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    boxes = torch.zeros((B, K, 4), device=device, dtype=torch.int64)

    for b in range(B):
        for k in range(K):
            bb = part_bbs[b, k]
            x1, y1, x2, y2 = bb.tolist()

            # Skip empty bounding boxes
            if (x2 <= x1) or (y2 <= y1):
                boxes[b, k] = 0
                continue

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Get single heatmap
            heatmap = heatmaps[b, k:k+1].unsqueeze(0)  # shape (1,1,H,W)

            # Build convolution kernel
            kernel = torch.ones((1, 1, mask_h, mask_w), device=device, dtype=torch.float)

            # Compute response map
            response = F.conv2d(heatmap.float(), kernel).squeeze(0).squeeze(0)  # shape: (H-mask_h+1, W-mask_w+1)

            # Find argmax
            flat_idx = response.argmax()
            best_y = (flat_idx // response.shape[1]).item()
            best_x = (flat_idx % response.shape[1]).item()

            # Compute final bounding box
            x1_opt = max(0, best_x - mask_w // 2)
            y1_opt = max(0, best_y - mask_h // 2)
            x2_opt = min(H, best_x + mask_w // 2)
            y2_opt = min(W, best_y + mask_h // 2)

            boxes[b, k] = torch.tensor([x1_opt, y1_opt, x2_opt, y2_opt], device=device)

    return boxes


def calculate_average_partwise_localization_accuracy(all_ious:list[dict], subgroup_mapping:dict, IoU_thr: float=0.5):
    #compute acc with ious and threshold for all images per part
    #all ious = list with each item having a matching from part to iou

    #preprocessing, merge groups that belong together and take the max iou value
    processed_ious = []
    for iou in all_ious:
        new_part = {}
        for part in subgroup_mapping.keys():
            grouped_parts = subgroup_mapping[part]
            max_iou = -1
            for g_part in grouped_parts:
                if g_part in iou.keys():
                    if iou[g_part] > max_iou:
                        max_iou = iou[g_part]
            new_part[part] = max_iou
        processed_ious.append(new_part)

    #collect part ious over all images
    collect = {}
    for part in processed_ious[0].keys():
        collect[part] = []

    for iou in processed_ious:
        for part, value in iou.items():
            if value == -1: #this part was not present in the image, no iou
                continue
            collect[part].append(1 if value >= IoU_thr else 0) #binary results for acc

    #divide sum by amount
    res = {}
    for part, collected_ious in collect.items():
        res[part] = sum(collected_ious)/len(collected_ious) if len(collected_ious) != 0 else -1

    mean_iou = [x for x in res.values() if x != -1]
    mean_iou_acc = sum(mean_iou)/len(mean_iou)

    return res, mean_iou_acc

    
def compute_optimal_masks(heatmaps, mask_sizes):
    """
    heatmaps:  (B, K, H, W) tensor
    mask_sizes: (B, K, 2)   tensor containing [mask_w, mask_h]
    returns:  (B, K, 4)     [x1, y1, x2, y2] for each heatmap
    """

    B, K, H, W = heatmaps.shape
    N = B * K  # total number of maps

    # Flatten to (N, 1, H, W)
    maps = heatmaps.reshape(N, 1, H, W)

    # Extract mask sizes flattened to length N
    mask_w = mask_sizes[..., 0].reshape(N).long()
    mask_h = mask_sizes[..., 1].reshape(N).long()

    # Prepare grouped convolution kernels
    max_h, max_w = mask_h.max(), mask_w.max()

    # Create full kernels (each with its own mask size)
    kernels = torch.zeros((N, 1, max_h, max_w), device=maps.device)
    for i in range(N):
        kernels[i, 0, :mask_h[i], :mask_w[i]] = 1.0

    # Convolve using grouped convolution
    response = F.conv2d(maps, kernels, groups=N)

    # Find argmax per heatmap
    flat = response.reshape(N, -1)
    max_idx = flat.argmax(dim=1)

    # Convert to x, y
    Ry = response.shape[2]
    Rx = response.shape[3]

    best_y = (max_idx // Rx)
    best_x = (max_idx % Rx)

    # Compute final bounding boxes
    x1 = (best_x - mask_w // 2).clamp(min=0)
    y1 = (best_y - mask_h // 2).clamp(min=0)
    x2 = (best_x + mask_w // 2).clamp(max=W)
    y2 = (best_y + mask_h // 2).clamp(max=H)

    # Reshape back to (B, K, 4)
    boxes = torch.stack([x1, y1, x2, y2], dim=1).reshape(B, K, 4)

    return boxes


def get_optimal_mask(heatmap:np.ndarray, part_bb:list):
    #takes the heatmap and
    
    if heatmap.size == 0 or len(part_bb) == 0: #normally both should be empty but just in case make or
        return []
    
    #mask height and width
    mask_w = int(part_bb[2] - part_bb[0])
    mask_h = int(part_bb[3] - part_bb[1])

    #conv for response map, take maximum value
    kernel = torch.ones((1, 1, mask_h, mask_w))
    response = F.conv2d(torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    
    #argmax from flatten, then convert back to 2d pos
    flat_response = torch.flatten(response)
    max_pos = flat_response.argmax()
    best_y = (max_pos // heatmap.shape[-1]).item()
    best_x = (max_pos %  heatmap.shape[-1]).item()

    #x1, y1, x2, y2
    return [max(0, best_x - int(mask_w/2)), max(0, best_y - int(mask_h/2)), min(heatmap.shape[0], best_x + int(mask_w/2)), min(heatmap.shape[1], best_y + int(mask_h/2))]


def bbox_iou_tensor(boxes1, boxes2):
    """
        Compute IoU for batches of bounding boxes.

        Args:
            boxes1: (B, K, 4) or (K, 4)
            boxes2: (B, K, 4) or (K, 4)
        
        Returns:
            ious: (B, K)
    """
    # Intersection coords
    x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
    y_top = torch.max(boxes1[..., 1], boxes2[..., 1])
    x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
    y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

    # Intersection width/height
    inter_w = (x_right - x_left).clamp(min=0)
    inter_h = (y_bottom - y_top).clamp(min=0)
    intersection = inter_w * inter_h

    # Area of boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)
    return iou
