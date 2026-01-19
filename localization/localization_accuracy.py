#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


def compute_localization_accuracy_centermean(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts: torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor],
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
):
    """
    Compute part localization accuracy by averaging distances across all attributes per part.

    This method evaluates localization by:
    1. Computing predicted coordinates for every attribute using its attention map
    2. Computing the distance from each attribute's prediction to its part's ground truth
    3. Aggregating distances by taking the mean across all attributes belonging to each part

    Args:
        pre_attri: attribute activation scores, shape: [B, A]
        attention: Attention/saliency heatmaps per attribute, shape [B, A, H_ATT, W_ATT].
        bounding_box_per_part: Ground truth bounding boxes per part, shape [B, K, 4]
        part_gts: Ground truth part keypoint coordinates, shape [B, K, 2].
        part_dict: Mapping from part IDs (str) to part names (str).
        part_attribute_mapping_tensor: Mapping from part names to attribute indices.
        collector_list: Accumulator list for per-image results. Each image's results
            are appended as a dict mapping part names to mean Euclidean distances.
        img_size: Target size for resizing attention maps. Default 299.

    Returns:
        tuple: (predicted_coords_attr, dist_per_part, resized_heatmaps, aggregated_scores)
            - predicted_coords_attr: Predicted location per attribute, shape [B, A, 2]
            - dist_per_part: Mean distance to GT per part, shape [B, K]. (-1 if not present)
            - resized_heatmaps: im size heatmap, shape [B, A, img_size, img_size]
            - aggregated_scores: Maximum attribute activation per part, shape [B, K]

    Note:
        Attributes not belonging to any part will have distance set to -1.
    """

    B, A, H_att, W_att = attention.shape

    # Resize heatmaps to image size
    resized_heatmaps = torch.nn.functional.interpolate(
        attention, size=img_size, mode='bilinear', align_corners=False
    )  # [B, A, img, img]

    # ---- 1) Compute predicted coordinates for EVERY ATTRIBUTE -------------
    B, A, H, W = resized_heatmaps.shape
    flat = resized_heatmaps.view(B, A, -1)
    max_idx = flat.argmax(dim=2)  # [B, A]

    # Find position in image coordinate
    y_attr = max_idx // H
    x_attr = max_idx % W
    predicted_coords_attr = torch.stack((x_attr, y_attr), dim=2)  # [B, A, 2]

    # ---- 2) Compute distance for EVERY ATTRIBUTE to its part GT -----------
    # part_gts is [B, K, 2]. We need each attribute to know which part it belongs to.
    #
    # Create lookup table attr -> part index
    attr_to_part = torch.full((A,), -1, dtype=torch.long)
    for part_idx, (part_id, part_name) in enumerate(part_dict.items()):
        attrs = part_attribute_mapping_tensor[part_name]
        attr_to_part[attrs] = part_idx

    # Compute distance per attribute (for attributes that belong to a part)
    dist_attr = torch.zeros(B, A, device=attention.device)

    # For each attribute, compute distance to its part GT
    for a in range(A):
        part_idx = attr_to_part[a].item()
        if part_idx == -1:
            dist_attr[:, a] = -1
            continue

        diff = part_gts[:, part_idx, :].float() - predicted_coords_attr[:, a, :].float()
        dist_attr[:, a] = torch.norm(diff, dim=1)

    # ---- 3) Aggregate per part -------------------------------------------
    K = len(part_dict)
    dist_per_part = torch.zeros(B, K, device=attention.device)
    aggregated_scores = torch.zeros(B, K, device=attention.device)
    mean_points = torch.zeros(B, K, 2, device=attention.device)

    for part_idx, (part_id, part_name) in enumerate(part_dict.items()):
        attrs = part_attribute_mapping_tensor[part_name]

        # distances for all attributes of this part
        sub_dists = dist_attr[:, attrs]  # [B, n_attr]

        # choose min distance among attributes (you may change to mean/max)
        # choose keypoint of attribute wiht min distance from all attributes of that part
        mean_dist = sub_dists.mean(dim=1)
        dist_per_part[:, part_idx] = mean_dist

        # aggregate scores (e.g., max activation)
        aggregated_scores[:, part_idx] = pre_attri[:, attrs].max(dim=1).values
        mean_points[:, part_idx, :] = predicted_coords_attr[:, attrs, :].float().mean(dim=1)

    # ---- 4) Set distances to -1 for parts not present ---------------------
    valid_mask = (part_gts.sum(dim=-1) != 0)  # [B, K]
    dist_per_part[~valid_mask] = -1

    # ---- 5) Collect results -----------------------------------------------
    for i in range(B):
        sub_res = dict(zip(list(part_dict.values()), dist_per_part[i].tolist()))
        collector_list.append(sub_res)

    return mean_points, dist_per_part, resized_heatmaps, aggregated_scores


def compute_localization_accuracy_aggregated(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts:torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor],
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
    interpolate=True
):
    """
    Compute part localization accuracy by summing attention maps of all attributes per part.

    This method evaluates localization by:
    1. Summing all attention maps belonging to each body part
    2. Finding the maximum activation location in the aggregated heatmap
    3. Computing Euclidean distance between predicted and ground truth coordinates

    Args:
        pre_attri: Attribute activation scores, shape: [B, A]
        attention: Attention/saliency heatmaps per attribute, shape [B, A, H_ATT, W_ATT].
        bounding_box_per_part: Ground truth bounding boxes per part, shape [B, K, 4].
        part_gts: Ground truth part keypoint coordinates, shape [B, K, 2].
        part_dict: Mapping from part IDs (str) to part names (str).
        part_attribute_mapping_tensor: Mapping from part names to attribute indices.
        collector_list: Accumulator list for per-image results. Each image's results
            are appended as a dict mapping part names to Euclidean distances.
        img_size: Target size for resizing attention maps. Default 299.
        interpolate: If True, return resized heatmaps. If False, return original resolution.

    Returns:
        tuple: (predicted_coords, dist, resized_heatmaps, aggregated_scores)
            - predicted_coords: Predicted part locations, shape [B, K, 2]
            - dist: Euclidean distances to ground truth, shape [B, K]. (-1 if not present)
            - resized_heatmaps: Aggregated heatmaps per part, shape [B, K, H', W']
            - aggregated_scores: Summed attribute activations per part, shape [B, K]

    Note:
        Unlike compute_localization_accuracy which uses argmax attribute selection,
        this method combines all attribute maps per part.
    """

    B, A, H_att, W_att = attention.shape
    K = len(part_dict)

    # Create a mask to track which parts are present
    valid_mask = (part_gts.sum(dim=-1) != 0)  # [B, K]

    # Aggregate attention maps per part by SUMMING all attributes belonging to that part
    aggregated_heatmaps = []
    aggregated_scores = []

    for part in part_dict.values():
        # Get indices of attributes belonging to this part
        attr_indices = part_attribute_mapping_tensor[part]

        # Sum attention maps for all attributes in this part [B, H, W]
        part_attention_sum = attention[:, attr_indices].sum(dim=1)  # [B, H_att, W_att]
        aggregated_heatmaps.append(part_attention_sum)

        # Also sum the activation scores for this part
        part_score_sum = pre_attri[:, attr_indices].sum(dim=1)  # [B]
        aggregated_scores.append(part_score_sum)

    # Stack to create [B, K, H, W]
    heatmaps = torch.stack(aggregated_heatmaps, dim=1)  # [B, K, H_att, W_att]
    aggregated_scores = torch.stack(aggregated_scores, dim=1)  # [B, K]

    # Resize heatmaps to image size
    resized_heatmaps = torch.nn.functional.interpolate(heatmaps, size=img_size, mode='bilinear', align_corners=False)

    # Get indices of max values in the aggregated heatmaps
    B, K, H, W = resized_heatmaps.shape
    flat = resized_heatmaps.view(B, K, -1)  # [B, K, HxW]

    max_idx = flat.argmax(dim=2)

    y = max_idx // W
    x = max_idx % W

    predicted_coords = torch.stack((x, y), dim=2)
    assert predicted_coords.shape == part_gts.shape

    # Compute euclidean distance
    diff = part_gts.float() - predicted_coords.float()
    dist = torch.norm(diff, dim=2)  # [B, K]

    # Set distance of not present parts to -1
    dist[~valid_mask] = -1

    for i in range(dist.shape[0]):
        sub_res = dict(zip(list(part_dict.values()), dist[i].tolist()))
        collector_list.append(sub_res)

    if interpolate:
        return predicted_coords, dist, resized_heatmaps, aggregated_scores
    else:
        # return orginal heatmaps (better for plotting)
        return predicted_coords, dist, heatmaps, aggregated_scores



def compute_localization_accuracy(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts: torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor],
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
):
    """
    Compute part localization accuracy using argmax attribute selection per part.

    This method evaluates how well a model localizes body parts by:
    1. For each part, selecting the attribute with the highest activation (argmax)
    2. Using that attribute's attention/saliency map to predict the part location
    3. Computing the Euclidean distance between predicted and ground truth coordinates

    Args:
        pre_attri: Attribute activation scores, shape: [B, A]
        attention: Attention/saliency heatmaps per attribute, shape [B, A, H_ATT, W_ATT].
        bounding_box_per_part: Ground truth bounding boxes per part, shape [B, K, 4].
        part_gts: Ground truth part keypoint coordinates, shape [B, K, 2].
        part_dict: Mapping from part IDs (str) to part names (str).
        part_attribute_mapping_tensor: Mapping from part names to attribute indices.
        collector_list: Accumulator list for per-image results. Each image's results
            are appended as a dict mapping part names to Euclidean distances.
        img_size: Target size for resizing attention maps. Default 299.

    Returns:
        tuple: (predicted_coords, dist, resized_heatmaps, max_scores_per_part)
            - predicted_coords: Predicted part locations, shape [B, K, 2]
            - dist: Euclidean distances to ground truth, shape [B, K]. (-1 if not present)
            - resized_heatmaps: im size heatmap, shape [B, K, img_size, img_size]
            - max_scores_per_part: Maximum attribute activation per part, shape [B, K]

    Note:
        Unlike compute_localization_accuracy_aggregated which sums all attribute
        maps per part, this method uses only the single highest-activated attribute's map.
    """
    # Step 1: For each part, find the attribute with highest activation (argmax)
    argmax_per_part: list[int] = []        # Index per part [K tensors of shape B]
    max_scores_per_part: list[float] = []    # Activation value of the winning attribute (float)

    for part in part_dict.values():
        # Find which attribute in this part has highest activation per image
        subset = pre_attri[:, part_attribute_mapping_tensor[part]]  # [B, num_attrs_in_part]
        argmax_in_subset = subset.argmax(dim=1)                     # [B] - index within subset

        # Map back to global attribute index
        result = part_attribute_mapping_tensor[part][argmax_in_subset]  # [B]
        argmax_per_part.append(result)

        # Store the actual activation value of the winning attribute
        max_score = pre_attri[torch.arange(pre_attri.shape[0]), result]  # [B]
        max_scores_per_part.append(max_score)

    # Stack scores: [K, B] -> transpose to [B, K]
    max_scores_per_part = torch.stack(max_scores_per_part).t()

    # Step 2: Create validity mask for parts present in each image
    # Parts with ground truth coordinates [0, 0] are considered not visible
    valid_mask = (part_gts.sum(dim=-1) != 0)  # [B, K]

    # Step 3: Select attention maps for the winning attributes
    idx = torch.stack(tuple(argmax_per_part)).to(attention.device)  # [K, B]
    idx = idx.t()                                                   # [B, K]

    batch_indices = torch.arange(attention.shape[0]).unsqueeze(1)
    heatmaps = attention[batch_indices, idx]  # [B, K, H, W]

    # Step 4: Resize heatmaps from feature map resolution to image resolution
    resized_heatmaps = torch.nn.functional.interpolate(
        heatmaps, size=img_size, mode='bilinear', align_corners=False
    )

    # Step 5: Find predicted part location as the maximum activation point
    B, K, H, W = resized_heatmaps.shape
    flat = resized_heatmaps.view(B, K, -1)  # [B, K, H*W]
    max_idx = flat.argmax(dim=2)  # [B, K] - flattened index of max value

    # Convert flattened index to 2D coordinates
    y = max_idx // W
    x = max_idx % W
    predicted_coords = torch.stack((x, y), dim=2)  # [B, K, 2]

    assert predicted_coords.shape == part_gts.shape, \
        f"Shape mismatch: predicted {predicted_coords.shape} vs gt {part_gts.shape}"

    # Step 6: Compute Euclidean distance between predicted and ground truth
    diff = part_gts.float() - predicted_coords.float()
    dist = torch.norm(diff, dim=2)  # [B, K]

    # Mark distance as -1 for parts not present in the image
    dist[~valid_mask] = -1

    # Step 7: Collect per-image results for later aggregation
    for i in range(dist.shape[0]):
        sub_res = dict(zip(list(part_dict.values()), dist[i].tolist()))
        collector_list.append(sub_res)

    return predicted_coords, dist, resized_heatmaps, max_scores_per_part


def create_part_attribute_mapping_tensor(part_attribute_mapping: Dict[str, List[int]], device):
    """
    Convert part-to-attribute mapping from lists to GPU tensors.

    Args:
        part_attribute_mapping: Dictionary mapping part names to attribute indices.
        device: PyTorch device where tensors will be placed.

    Returns:
        Dict[str, torch.IntTensor]: Dictionary with values converted to tensors.
    """
    part_attribute_mapping_tensor = {}
    for part, attr_list in part_attribute_mapping.items():
        part_attribute_mapping_tensor[part] = torch.tensor(attr_list).to(device)
    return part_attribute_mapping_tensor


def compute_optimal_masks_per_mask(heatmaps, part_bbs):
    """
    Compute optimal bounding box placement per heatmap using sliding window convolution.

    For each part, finds the position where a bounding box of the ground truth size
    would capture the maximum total activation in the heatmap.

    Args:
        heatmaps: Attention/saliency maps, shape [B, K, H, W].
        part_bbs: Ground truth bounding boxes defining mask sizes, shape [B, K, 4].

    Returns:
        torch.Tensor: Optimal bounding boxes, shape [B, K, 4]. Returns zeros for invalid boxes.

    Note:
        Per-sample loop implementation. For batched processing, use compute_optimal_masks.
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

            #best_y = H - best_y
            #best_x = W - best_x


            # Compute final bounding box
            x1_opt = max(0, best_x)
            y1_opt = max(0, best_y)
            x2_opt = min(W, best_x + mask_w)
            y2_opt = min(H, best_y + mask_h)

            boxes[b, k] = torch.tensor([x1_opt, y1_opt, x2_opt, y2_opt], device=device)

    return boxes


def calculate_average_partwise_localization_accuracy(all_ious:list[dict], subgroup_mapping:dict, IoU_thr: float=0.5, blacklist=[]):
    """
    Calculate per-part localization accuracy from IoU scores with thresholding.

    Args:
        all_ious: List of dicts mapping part names to IoU values. (-1 if not present)
        subgroup_mapping: Dict mapping merged part names to lists of original part names.
        IoU_thr: IoU threshold for correct localization. Default 0.5.
        blacklist: Part names to exclude from mean accuracy computation.

    Returns:
        tuple: (res, mean_iou_acc)
            - res: Dict mapping part names to accuracy values. (-1 if never present)
            - mean_iou_acc: Mean accuracy across all non-blacklisted parts.
    """
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


    mean_iou = [x for name, x in res.items() if x != -1 and name not in blacklist]
    mean_iou_acc = sum(mean_iou)/len(mean_iou)

    print("\n--------- LOCALIZATION ACCURACY ---------\n")

    for group_name, acc in res.items():
        print(f"Group {group_name} - LocAcc: {acc:.4f}")
    print(f"\nMean LocAcc: {mean_iou_acc:.4f}")

    return res, mean_iou_acc


def calculate_average_partwise_localization_distance(all_distances:list[dict], subgroup_mapping:dict, verbose=True):
    """
    Calculate average Euclidean distance per part across all images.

    Args:
        all_distances: List of dicts mapping part names to distances. (-1 if not present)
        subgroup_mapping: Dict mapping merged part names to lists of original part names.
        verbose: If True, print formatted results to stdout. Default True.

    Returns:
        tuple: (res, mean_dist)
            - res: Dict mapping part names to mean distance values. (-1 if never present)
            - mean_dist: Mean distance across all parts.

    Note:
        Lower distance values indicate better localization performance.
    """
    #compute acc with ious and threshold for all images per part
    #all ious = list with each item having a matching from part to iou

    #preprocessing, merge groups that belong together and take the max iou value
    processed_distances = []
    for dist in all_distances:
        new_part = {}
        for merged_part, group_parts in subgroup_mapping.items():
            best_dist = None

            for p in group_parts:
                if p in dist and dist[p] != -1:
                    if best_dist is None or dist[p] < best_dist:
                        best_dist = dist[p]

            # if no valid distance found set to -1
            new_part[merged_part] = best_dist if best_dist is not None else -1
        
        processed_distances.append(new_part)

    #collect part ious over all images
    collect = {}
    for part in processed_distances[0].keys():
        collect[part] = []

    for dist in processed_distances:
        for part, value in dist.items():
            if value == -1: #this part was not present in the image, no iou
                continue
            collect[part].append(value) #binary results for acc

    #divide sum by amount

    res = {}
    for part, collected_dists in collect.items():
        res[part] = sum(collected_dists)/len(collected_dists) if len(collected_dists) != 0 else -1


    mean_dist = [x for _, x in res.items() if x != -1]
    mean_dist = sum(mean_dist)/len(mean_dist)

    if verbose:
        print("\n--------- LOCALIZATION DISTANCE ---------\n")

        for group_name, acc in res.items():
            print(f"Group {group_name} - LocDist: {acc:.4f}")
        print(f"\nMean LocDist: {mean_dist:.4f}")

    return res, mean_dist

    
def compute_optimal_masks(heatmaps, mask_sizes):
    """
    Compute optimal bounding box placement using batched grouped convolution.

    Efficiently finds the position where a mask of given size captures maximum
    activation for all heatmaps simultaneously.
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