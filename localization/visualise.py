import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import os
import random

import torch
import torch.nn.functional as F


def create_attribute_mosaic(
    images,             # torch tensor, shape (I, C, H, W)
    heatmaps,           # torch tensor, shape (I, A, H, W)
    attribute_names,    # list of strings, len A
    scores,             # numpy or torch, shape (I, A)
    alpha=0.5,
    resize_to=None,
    font_path=None,
    font_size=20,
    header_height=40,
    score_height=30,
):

    # -------------------------
    # Validate inputs
    # -------------------------
    I, C, H, W = images.shape
    _, A, H_hm, W_hm = heatmaps.shape

    # If images are not in [0,1], normalize?
    # We'll assume they are already suitable for visualization.

    # -------------------------
    # Load font
    # -------------------------
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # -------------------------
    # Resize heatmaps to match image resolution
    # -------------------------
    if (H_hm != H) or (W_hm != W):
        heatmaps = F.interpolate(
            heatmaps,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

    # -------------------------
    # Convert images to PIL
    # -------------------------
    proc_images = [tensor_to_pil(images[i]) for i in range(I)]

    # -------------------------
    # Resize images & heatmaps if requested
    # -------------------------
    if resize_to is not None:
        new_w, new_h = resize_to
        proc_images = [img.resize((new_w, new_h), Image.LANCZOS) for img in proc_images]

        heatmaps = F.interpolate(
            heatmaps,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )
        H, W = new_h, new_w

    # -------------------------
    # Prepare output mosaic canvas
    # -------------------------
    cell_w, cell_h = W, H
    mosaic_w = A * cell_w
    mosaic_h = I * (cell_h + score_height) + header_height

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), "white")
    draw = ImageDraw.Draw(mosaic)

    # -------------------------
    # Draw attribute column headers
    # -------------------------
    for a, name in enumerate(attribute_names):
        x_center = a * cell_w + cell_w // 2
        y_center = header_height // 2

        bbox = draw.textbbox((0, 0), name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.text(
            (x_center - text_w // 2, y_center - text_h // 2),
            name,
            fill="black",
            font=font
        )

    # -------------------------
    # Draw each cell (image + heatmap + score)
    # -------------------------
    heatmaps_np = heatmaps.detach().cpu().numpy()

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    for i in range(I):
        for a in range(A):
            base_img = proc_images[i]

            heatmap = heatmaps_np[i, a]  # (H, W)
            heatmap_color = plt_colormap(heatmap)  # (H, W, 3)
            heatmap_img = Image.fromarray((heatmap_color * 255).astype(np.uint8)).convert("RGBA")

            base_rgba = base_img.convert("RGBA")
            blended = Image.blend(base_rgba, heatmap_img, alpha)

            # Position
            top = header_height + i * (cell_h + score_height)
            left = a * cell_w
            mosaic.paste(blended, (left, top))

            # Score text
            score_str = f"{scores[i, a]:.3f}"

            bbox = draw.textbbox((0,0), score_str, font=font)
            text_w = bbox[2] - bbox[0]

            y_text = top + cell_h + 5
            x_text = left + cell_w // 2 - text_w // 2

            draw.text((x_text, y_text), score_str, fill="black", font=font)

    mosaic.save("outputs/APN/mosaic.png")



def plt_colormap(x):
    """Simple jet-like colormap for heatmaps."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def tensor_to_pil(img_tensor):
    """
    Convert torch tensor (C, H, W) in [0,1] (or [0,255]) to PIL.Image.
    Supports 1-channel or 3-channel.
    """
    img = img_tensor.detach().cpu()

    if img.max() <= 1.0:
        img = img * 255.0

    img = img.byte()

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    img_np = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img_np)


def visualise_localization_acc_boxes(
    imgs,
    img_paths,
    part_gts,
    saliency_maps,
    predicted_bounding_boxes,
    gt_part_boxes,
    scores,
    batch_nr,
    batch_ious,
    part_names,
    batch_idx=0,
    t_mean=(0.5, 0.5, 0.5),
    t_std=(2, 2, 2),
    save_path=""
):
    """
        Takes image, visualizes all bounding boxes for given parts and the heatmaps with predicted bounding boxes

        imgs: torch.Tensor of [B, C, H, W] the respective images
        saliency_maps: torch.Tensor of [B, K, H, W] of the saliency maps matched to the segmentation mask shape, per attribute A
        predicted_bounding_boxes: torch.Tensor of [B, K, 4], the bounding boxes per CUB part K, BB shape (x1, y1, x2, y2)
        part_boxes: torch.IntTensor of [B, K, 4], the ground truth bounding boxes per CUB part K, BB shape (x1, y1, x2, y2)
        batch_nr: The current batch we are looking at
        batch_ious: Tensor of shape [B, K], the IoU values for each part in this batch
        part_names: A list of part names that are to be visualized.
        batch_idx: The index in the batch from which we extract images, defaults to random one
        t_mean: The mean applied during preprocessing of the image
        t_std: The std applied during preprocessing of the image
        save_path: Path where to save the visualizations
    """

    B, A, H, W = saliency_maps.shape
    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)
    n_parts = predicted_bounding_boxes.shape[1]


    img = imgs[batch_idx]
    img_path = img_paths[batch_idx]
    masks = saliency_maps[batch_idx]
    predicted_bounding_boxes = predicted_bounding_boxes[batch_idx]
    gt_part_boxes = gt_part_boxes[batch_idx]
    ious = batch_ious[batch_idx]
    scores = scores[batch_idx]
    gts = part_gts[batch_idx]

    # Denorm image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(2, n_parts, figsize=(2*n_parts, 5))

    for col in range(n_parts):
        iou = ious[col].item()
        # Part name
        axes[0, col].set_title("{}: IoU {}, score {}".format(part_names[col], round(iou, 4), round(scores[col].item(), 3)), fontsize=10, pad=4)

        # Image with BB
        axes[0, col].imshow(img_np)
        axes[0, col].axis('off')

        gt = gts[col].cpu().detach().numpy()
        #if gt.sum() != 0:
        axes[0, col].scatter(gt[0], gt[1], s=1, c="red")

        gt_box = gt_part_boxes[col].cpu().detach().numpy()
        x1, y1, x2, y2 = gt_box
        width  = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),    
            width,
            height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        axes[0, col].add_patch(rect)

    
        # Segmentation mask with BB
        attn_mask = masks[col].cpu().detach().numpy()
        axes[1, col].imshow(attn_mask, cmap='jet')
        axes[1, col].axis('off')

        pred_box = predicted_bounding_boxes[col].cpu().detach().numpy()
        x1, y1, x2, y2 = pred_box
        width  = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),    
            width,
            height,
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        axes[1, col].add_patch(rect)

        # Show this box on image too to assess overlap
        rect = patches.Rectangle(
            (x1, y1),    
            width,
            height,
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        axes[0, col].add_patch(rect)


    axes[0, 0].set_ylabel("Image with GT BB")
    axes[1, 0].set_ylabel("Attention with Pred BB")

    img_name = "_".join(img_path.split(os.sep)[-2:]).rstrip(".jpg")

    # Create string for this img
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"b{batch_nr}_id{batch_idx}_part_viz_{img_name}.png"), dpi=200, bbox_inches='tight')
    plt.close()



def visualise_part_segmentations(
    imgs,
    saliency_maps,
    seg_masks,
    attribute_names,
    iou_scores,
    source_paths=None,
    attributes=10,
    batch_idx=None,
    t_mean=(0.5, 0.5, 0.5),
    t_std=(2, 2, 2),
    save_path=""
):
    """
        imgs: torch.Tensor of [B, C, H, W] the respective images
        saliency_maps: torch.Tensor of [B, A, H, W] of the saliency maps matched to the segmentation mask shape, per attribute A
        seg_masks: torch.Tensor of [B, A, H, W] the segmentation masks, per attribute A
        attribute_names: List of [A], giving each attribute its name
        iou_scores: The IoU scores for each img, per attribute [B, A]
        source_paths: The paths to the image sources, will be rendered into the image if given
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
    ious = iou_scores[batch_idx][attributes].cpu().detach().numpy()
    if source_paths is not None:
        img_name = source_paths[batch_idx]
        img_name = "_".join(img_name.split(os.sep)[-2:]).rstrip(".jpg")
    else:
        img_name = f"id{batch_idx}"

    # Denorm image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(2, n_attributes, figsize=(2*n_attributes, 5))

    for col in range(n_attributes):

        # Attribute name along with the IoU score for that (gt, pred) pair
        axes[0, col].set_title(f"{attr_names[col]}\n{ious[col]:.4f}", fontsize=9, pad=4)

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
    plt.savefig(os.path.join(save_path, f"{img_name}_attr{attr_str}.png"), dpi=200, bbox_inches='tight')
    plt.close()
