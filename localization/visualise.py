import matplotlib.pyplot as plt
import numpy as np
import os
import random


def visualise_part_segmentations(
    imgs,
    saliency_maps,
    seg_masks,
    attribute_names,
    batch_nr,
    attributes=10,
    batch_idx=None,
    t_mean=0.5,
    t_std=2,
    save_path=""
):
    """
        imgs: torch.Tensor of [B, C, H, W] the respective images
        saliency_maps: torch.Tensor of [B, A, H, W] of the saliency maps matched to the segmentation mask shape, per attribute A
        seg_masks: torch.Tensor of [B, A, H, W] the segmentation masks, per attribute A
        attribute_names: List of [A], giving each attribute its name
        batch_nr: The current batch we are looking at
        attributes: Either a list of attributes that are to be visualized, or a number of how many random attributes to sample.
        batch_idx: The index in the batch from which we extract images, defaults to random one
        t_mean: The mean applied during preprocessing of the image
        t_std: The std applied during preprocessing of the image
        save_path: Path where to save the visualizations
    """

    B, A, H, W = saliency_maps.shape
    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)
    if isinstance(attributes, int):
        attributes = random.sample(range(A), attributes)
    n_attributes = len(attributes)

    img = imgs[batch_idx]
    masks = saliency_maps[batch_idx][attributes]
    seg_masks = seg_masks[batch_idx][attributes]
    attr_names = [attribute_names[i] for i in attributes]

    # Denorm image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(2, n_attributes, figsize=(2*n_attributes, 5))

    for col in range(n_attributes):

        # Attribute name
        axes[0, col].set_title(attr_names[col], fontsize=10, pad=4)

        # Attention mask overlay
        attn_mask = masks[col].cpu().detach().numpy()
        axes[0, col].imshow(img_np)
        axes[0, col].imshow(attn_mask, cmap='jet', alpha=0.5)
        axes[0, col].axis('off')

        # Segmentation mask overlay
        seg_mask = seg_masks[col].cpu().detach().numpy()
        axes[1, col].imshow(img_np)
        axes[1, col].imshow(seg_mask, cmap='spring', alpha=0.5)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("ProtoNet Map")
    axes[1, 0].set_ylabel("Part Segmentation")

    # Create string for this img
    attr_str = "_".join([str(a) for a in attributes])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"b{batch_nr}_id{batch_idx}_attr{attr_str}.png"), dpi=200, bbox_inches='tight')
    plt.close()
