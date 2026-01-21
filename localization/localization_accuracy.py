#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F



def compute_localization_distance( 
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts:torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor], 
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
    use_argmax=False):

    if use_argmax:
        return compute_localization_distance_with_argmaxing(
            pre_attri,
            attention,
            bounding_box_per_part,
            part_gts,
            part_dict,
            part_attribute_mapping_tensor, 
            collector_list,
            img_size
        )
    else:
        return compute_localization_distance_without_argmaxing(
            pre_attri,
            attention,
            bounding_box_per_part,
            part_gts,
            part_dict,
            part_attribute_mapping_tensor, 
            collector_list,
            img_size
        )


def compute_localization_distance_without_argmaxing(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts: torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor],
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
    attr_thresh = 0.5
):
    """
    Computes localization distance per part without using argmax over attributes.
    Instead, computes predicted coordinates for every attribute, then aggregates
    distances for attributes belonging to each part.

    Returns:
        predicted_coords_attr: [B, A, 2]
        dist_per_part:        [B, K]
        resized_heatmaps:     [B, A, H', W']
        aggregated_scores:    [B, K] (max activation per part group)
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

    y_attr = max_idx // W
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

    for part_idx, (part_id, part_name) in enumerate(part_dict.items()):
        attrs = part_attribute_mapping_tensor[part_name]
        #threshold attribute activations and select indices based on that
        above_threshold = (pre_attri[:, attrs] > attr_thresh).any(dim=0)

        attrs = attrs[above_threshold]

        #if nothing left it sucks, skip for now
        if attrs.numel() == 0:
            continue

        # distances for all attributes of this part
        sub_dists = dist_attr[:, attrs]  # [B, n_attr]

        # choose min distance among attributes
        min_dists = sub_dists.mean(dim=1)
        dist_per_part[:, part_idx] = min_dists

        # aggregate scores (e.g., max activation)
        aggregated_scores[:, part_idx] = pre_attri[:, attrs].max(dim=1).values

    # ---- 4) Set distances to -1 for parts not present ---------------------
    valid_mask = (part_gts.sum(dim=-1) != 0)  # [B, K]
    dist_per_part[~valid_mask] = -1

    # ---- 5) Collect results -----------------------------------------------
    for i in range(B):
        sub_res = dict(zip(list(part_dict.values()), dist_per_part[i].tolist()))
        collector_list.append(sub_res)

    return predicted_coords_attr, dist_per_part, resized_heatmaps, aggregated_scores


def compute_localization_distance_with_argmaxing(
    pre_attri: torch.FloatTensor,
    attention: torch.FloatTensor,
    bounding_box_per_part: torch.IntTensor,
    part_gts:torch.IntTensor,
    part_dict: Dict[str, int],
    part_attribute_mapping_tensor: Dict[str, torch.IntTensor], 
    collector_list: List[Dict[str, float]],
    img_size: int = 299,
):
    """
        Args:
            pre_attri: activations for each attribute in the model
            attention: heatmaps/saliency maps per attribute, shape [B, A, H, W]
            part_bounding_boxes: The bounding box per part, for each image, as given by its coordinates [B, K, 4]
            part_dict: Mapping from part segmentation groups to CUB original parts
            part_attribute_mapping: Mapping from CUB parts to relative attribute IDs. Amount of IDs should be A.
            collector_list: List to collect the mIoU results into.
            img_size: The image size.
        
        Returns:
            TODO
    """
    # part_attribute_mapping means a dict that maps from a part (CUB/parts/parts.txt) to a list of attribute IDs
    # sum of all attribute IDs must not be more that number of attention maps returned/attributes used by model. also is
    # required to be 0 indexed and correctly match the attribute order in the model
    #reimplementation of paper description

    #get argmax attribute per part
    # Per part k \in K, get the argmax index (attribute idx that had the highest activation for that part) for each img in the batch
    argmax_per_part = [] # max index per part, -1 if part not present
    max_scores_per_part = []
    for part in part_dict.values():
        #take argmax of each part group
        subset = pre_attri[:, part_attribute_mapping_tensor[part]]
        
        argmax_in_subset = subset.argmax(dim=1)
        result = part_attribute_mapping_tensor[part][argmax_in_subset] #now res should be batchsize shape
        
        argmax_per_part.append(result)

        max_score = pre_attri[torch.arange(pre_attri.shape[0]), result]  # [batch_size]
        max_scores_per_part.append(max_score)
        

    max_scores_per_part = torch.stack(max_scores_per_part).t()

    # Create a mask to track which part bounding boxes are non-empty
    valid_mask = (bounding_box_per_part.sum(dim=-1) != 0)  # [B, K]
    
    #part is [0, 0] if it doesnt exist
    valid_mask = (part_gts.sum(dim=-1) != 0)  # [B, K]

    # Take heatmaps: For each part, get the heatmap of the attribute belonging to that part that had the highest activation
    # TODO: Check this if this is correct! Maybe need torch.gather
    idx = torch.stack(tuple(argmax_per_part)).to(attention.device) # [K, B]
    idx = idx.t()  # [B, K]
    batch_indices = torch.arange(attention.shape[0]).unsqueeze(1)
    heatmaps = attention[batch_indices, idx] # [B, K, H, W] --> e.g. for inception: [B, 15, 8, 8]
    
    # Resize heatmaps to image size
    resized_heatmaps = torch.nn.functional.interpolate(heatmaps, size=img_size, mode='bilinear', align_corners=False)

    #get indices of max vals
    B, A, H, W = resized_heatmaps.shape
    flat = resized_heatmaps.view(B, A, -1) # [B, A(15), HxW]

    max_idx = flat.argmax(dim=2)

    y = max_idx // W
    x = max_idx % W

    predicted_coords = torch.stack((x, y), dim=2)
    assert predicted_coords.shape == part_gts.shape
    

    #now we have our part_gts [B, A(15), 2] and our predicted coords [B, A(15), 2] and can compute euclidean distance
    diff = part_gts.float() - predicted_coords.float()
    dist = torch.norm(diff, dim=2) # [B, A(15)]

    #now we set the distance of not present parts to -1
    dist[~valid_mask] = -1

    for i in range(dist.shape[0]):
        sub_res = dict(zip(list(part_dict.values()), dist[i].tolist()))
        collector_list.append(sub_res)

    return predicted_coords, dist, resized_heatmaps, max_scores_per_part
    



def calculate_average_partwise_localization_distance(all_distances:list[dict], subgroup_mapping:dict, verbose=True):
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

    





