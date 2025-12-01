from utils.mappings import MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS, CBM_SELECTED_CUB_ATTRIBUTE_IDS

def get_head_attributes():

    # CUB head attrributes from 0-311
    head_cub = MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS["head"]
    
    # Mapped to the valid 0-111 attributes used in our model
    head_valid = [x in CBM_SELECTED_CUB_ATTRIBUTE_IDS for x in head_cub]
    head_indices = [CBM_SELECTED_CUB_ATTRIBUTE_IDS.index(cub_index) for (cub_index, valid) in zip(head_cub, head_valid) if valid]

    return head_indices


