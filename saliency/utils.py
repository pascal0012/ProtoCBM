from utils.mappings import (
    MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS,
    CBM_SELECTED_CUB_ATTRIBUTE_IDS
)

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
