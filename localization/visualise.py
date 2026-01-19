from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import os
import random

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

from localization.vis_utils import DEFAULT_STYLES, GT_STYLE, PredictionStyle, build_title, denormalize_image, extract_image_name, plot_line_to_gt, plot_prediction, tensor_to_numpy


def visualize_keypoint_distances(gts,
                                imgs,
                                img_paths,
                                preds,
                                dists,
                                batch_nr,
                                part_names,
                                batch_idx=None,
                                t_mean=(0.5, 0.5, 0.5),
                                t_std=(2, 2, 2),
                                save_path=""):
    """
    gt:   torch.Tensor (15,2)   -> ground truth xy
    pred: torch.Tensor (15,2)   -> predicted xy
    image: numpy or torch image HxWx3
    names: list of 15 strings
    scores: list or tensor of 15 floats
    """

    B, C, H, W = imgs.shape
    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)
    n_parts = gts.shape[1]

    gt = gts[batch_idx]
    img = imgs[batch_idx]
    img_path = img_paths[batch_idx]
    pred = preds[batch_idx]
    dists = dists[batch_idx]
    #gts = part_gts[batch_idx]

    # Convert tensors to numpy
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    dists_np = dists.cpu().numpy()

    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    # Make 15 subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(15):
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis("off")

        gx, gy = gt_np[i]
        px, py = pred_np[i]

        if gx == 0 and gy == 0:
            ax.set_title(f"{part_names[i]}\ndist: {dists_np[i]:.1f}")
            continue

        # Ground truth point (green)
        ax.scatter(gx, gy, c="lime", s=40, marker="o", label="GT")

        # Predicted point (red)
        ax.scatter(px, py, c="red", s=40, marker="x", label="Pred")

        # Line between them (dotted)
        ax.plot([gx, px], [gy, py], "w--", linewidth=1.5)

        # Title with name, score, and distance
        ax.set_title(f"{part_names[i]}\ndist: {dists_np[i]:.1f}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"b{batch_nr}_id{batch_idx}_part_viz.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_keypoint_distances_with_heatmaps(gts,
                                imgs,
                                img_paths,
                                preds,
                                dists,
                                heatmaps,
                                batch_nr,
                                part_names,
                                batch_idx=None,
                                t_mean=(0.5, 0.5, 0.5),
                                t_std=(2, 2, 2),
                                save_path="",
                                img_size=299):
    """
    Visualize keypoints with interpolated heatmaps overlayed on the image.

    gt:   torch.Tensor (B, K, 2)   -> ground truth xy
    pred: torch.Tensor (B, K, 2)   -> predicted xy
    heatmaps: torch.Tensor (B, K, H, W) -> heatmaps per body part (original or interpolated)
    image: torch.Tensor (B, C, H, W)
    names: list of K strings
    """

    B, C, H, W = imgs.shape
    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)

    n_parts = gts.shape[1]
    gt = gts[batch_idx]
    img = imgs[batch_idx]
    img_path = img_paths[batch_idx]
    pred = preds[batch_idx] 
    dist = dists[batch_idx]
    heatmap = heatmaps[batch_idx]  # [K, H_hm, W_hm]

    # Convert tensors to numpy
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    dists_np = dist.cpu().numpy()

    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    # Interpolate heatmaps to image size if needed
    H_hm, W_hm = heatmap.shape[1], heatmap.shape[2]
    if H_hm != img_size or W_hm != img_size:
        heatmap_interp = F.interpolate(
            heatmap.unsqueeze(0),  # [1, K, H_hm, W_hm]
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [K, img_size, img_size]
    else:
        heatmap_interp = heatmap

    heatmap_np = heatmap_interp.cpu().numpy()

    # Make 15 subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(n_parts):
        ax = axes[i]

        # Show image with heatmap overlay
        ax.imshow(img_np)

        # Normalize heatmap for this part for better visualization
        hm = heatmap_np[i]
        if hm.max() > hm.min():
            hm_norm = (hm - hm.min()) / (hm.max() - hm.min())
        else:
            hm_norm = hm
        ax.imshow(hm_norm, cmap='jet', alpha=0.5)
        ax.axis("off")

        gx, gy = gt_np[i]
        px, py = pred_np[i]

        # Always plot predicted point (red X) - this is the highest activation keypoint
        ax.scatter(px, py, c="red", s=100, marker="x", linewidths=3, label="Pred (max act.)")

        if gx == 0 and gy == 0:
            # No GT available
            ax.set_title(f"{part_names[i]}\nNo GT", fontsize=10)
            continue

        # Ground truth point (green circle)
        ax.scatter(gx, gy, c="lime", s=100, marker="o", edgecolors='black', linewidths=1, label="GT")

        # Line between them (white dotted)
        ax.plot([gx, px], [gy, py], "w--", linewidth=2)

        # Title with name and distance
        ax.set_title(f"{part_names[i]}\ndist: {dists_np[i]:.1f} px", fontsize=10)

    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"b{batch_nr}_id{batch_idx}_keypoints_with_heatmaps.png"), dpi=200, bbox_inches='tight')
    plt.close()


def visualize_combined_keypoints(
    gts: torch.Tensor,
    imgs: torch.Tensor,
    img_paths: list[str],
    predictions: dict[str, torch.Tensor],
    distances: dict[str, torch.Tensor],
    part_names: list[str],
    batch_idx: Optional[int] = None,
    t_mean: Tuple = (0.5, 0.5, 0.5),
    t_std: Tuple = (2, 2, 2),
    save_path: str = "",
    styles: Optional[dict[str, PredictionStyle]] = None,
):
    """
    Visualize multiple keypoint prediction types on the same image.

    Args:
        gts: torch.Tensor (B, K, 2) -> ground truth xy coordinates
        imgs: torch.Tensor (B, C, H, W) -> input images
        img_paths: list of image paths
        predictions: dict mapping method names to tensors (B, K, 2) of predicted xy coordinates
                    e.g., {"argmax": preds_argmax, "agg": preds_agg, "weighted": preds_weighted}
        distances: dict mapping method names to tensors (B, K) of distances
                   e.g., {"argmax": dists_argmax, "agg": dists_agg, "weighted": dists_weighted}
        batch_nr: batch number for filename
        part_names: list of K part name strings
        batch_idx: which sample in batch to visualize (random if None)
        t_mean: normalization mean
        t_std: normalization std
        save_path: output directory
        img_size: target image size
        styles: optional dict mapping method names to PredictionStyle objects
    """
    B, C, H, W = imgs.shape
    if batch_idx is None:
        batch_idx = random.randint(0, B - 1)
        
    n_parts = gts.shape[1]

    # Use default styles if not provided
    if styles is None:
        styles = DEFAULT_STYLES

    # Extract single sample data
    gt_np = tensor_to_numpy(gts[batch_idx])
    img_np = denormalize_image(imgs[batch_idx], t_mean, t_std)
    img_name = extract_image_name(img_paths[batch_idx])

    # Convert predictions and distances to numpy
    preds_np = {name: tensor_to_numpy(pred[batch_idx]) for name, pred in predictions.items()}
    dists_np = {name: tensor_to_numpy(dist[batch_idx]) for name, dist in distances.items()}

    # Create subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    method_names = list(predictions.keys())

    for i in range(n_parts):
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis("off")

        gx, gy = gt_np[i]
        has_gt = not (gx == 0 and gy == 0)
        is_first = (i == 0)

        # Plot each prediction type
        for method_name in method_names:
            px, py = preds_np[method_name][i]
            style = styles.get(method_name, DEFAULT_STYLES.get(method_name, 
                PredictionStyle(color="gray", marker="o", size=80, label=method_name)))
            
            plot_prediction(ax, px, py, style, show_label=is_first)
            
            if has_gt:
                plot_line_to_gt(ax, gx, gy, px, py, style)

        # Plot GT if available
        if has_gt:
            plot_prediction(ax, gx, gy, GT_STYLE, show_label=is_first)

        # Build title with distances
        title_distances = {method_name: dists_np[method_name][i] for method_name in method_names}
        ax.set_title(build_title(part_names[i], title_distances, has_gt), fontsize=9)

    # Add legend to first subplot
    axes[0].legend(loc="upper left", fontsize=8)

    # Build dynamic title
    method_labels = " vs ".join([styles.get(m, DEFAULT_STYLES.get(m, 
        PredictionStyle(color="", marker="", size=0, label=m))).label for m in method_names])
    fig.suptitle(
        f"Combined Keypoint Comparison: GT (green) vs {method_labels}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{img_name}_{batch_idx}_combined_keypoints.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_threshold_curve(thresholds, accs, save_path="", label_area=True):
    #print(save_path)
    #print(thresholds)
    #print(accs)
    thresholds = np.array(thresholds)
    accuracies = np.array(accs)

    # Calculate Area Under the Curve
    auc = np.trapz(accuracies, thresholds)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("mIoU")
    plt.title("Threshold-mIoU Tradeoff Curve")

    # Annotate AUC on plot
    plt.text(
        0.05, 0.05,
        f"AUC = {auc:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    plt.grid(True)

    plt.savefig(os.path.join(save_path, "mIoU_threshold_curve.png"), bbox_inches='tight')
    plt.close()



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
    batch_idx=None,
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



def save_individual_activation_maps(
    imgs,
    saliency_maps,
    attribute_names,
    part_attribute_mapping,
    source_paths,
    batch_idx=0,
    t_mean=(0.5, 0.5, 0.5),
    t_std=(2, 2, 2),
    save_path=""
):
    """
    Save individual activation maps for each body part and attribute.
    Creates two plots per attribute:
    1) Scaled heatmap overlayed on image
    2) Original size heatmap

    Creates folder structure: save_path/body_part/attribute_name_[scaled|original].png

    Args:
        imgs: torch.Tensor of [B, C, H, W] - input images
        saliency_maps: torch.Tensor of [B, A, H, W] - activation maps per attribute
        attribute_names: List of [A] attribute names
        part_attribute_mapping: Dict mapping part names to attribute indices
        source_paths: List of image paths
        batch_idx: Which image in batch to visualize
        t_mean: Mean applied during preprocessing
        t_std: Std applied during preprocessing
        save_path: Base path where to save visualizations
    """
    B, A, H_map, W_map = saliency_maps.shape

    # Get the specific image and its activation maps
    img = imgs[batch_idx]
    masks = saliency_maps[batch_idx]  # [A, H_map, W_map]
    img_path = source_paths[batch_idx]

    # Extract image identifier from path
    img_name = "_".join(img_path.split(os.sep)[-2:]).rstrip(".jpg")

    # Only have part segmentations for first 70 classes
    # Try to extract class ID, skip if it's not a standard CUB dataset format
    try:
        class_id = int(img_name[:3])
        if class_id > 70:
            return
    except (ValueError, IndexError):
        # Not a standard CUB format (e.g., custom image), continue anyway
        pass

    # Denormalize image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H_img x W_img x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)
    H_img, W_img = img_np.shape[:2]

    # Resize all attention maps to image size once
    # Convert to torch tensor for interpolation
    masks_resized = F.interpolate(
        masks.unsqueeze(0),  # [1, A, H_map, W_map]
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # [A, H_img, W_img]

    # Iterate through each body part
    for part_name, attr_indices in part_attribute_mapping.items():
        # Create folder for this body part
        part_folder = os.path.join(save_path, part_name)
        os.makedirs(part_folder, exist_ok=True)

        # Save each attribute's activation map for this part
        for attr_idx in attr_indices:
            if attr_idx >= len(attribute_names):
                continue

            attr_name = attribute_names[attr_idx]

            # Get both original and scaled versions
            attn_mask_original = masks[attr_idx].cpu().detach().numpy()  # [H_map, W_map]
            attn_mask_scaled = masks_resized[attr_idx].cpu().detach().numpy()  # [H_img, W_img]

            # Sanitize filename once
            safe_attr_name = attr_name.replace('/', '_').replace(' ', '_')

            # ========== PLOT 1: Scaled heatmap overlayed on image ==========
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
            ax1.imshow(img_np)
            ax1.imshow(attn_mask_scaled, cmap='jet', alpha=0.5)
            ax1.axis('off')
            ax1.set_title(f'{attr_name}\n(Scaled to Image Size)', fontsize=10)

            plt.tight_layout()
            save_file_scaled = os.path.join(part_folder, f"{img_name}_{safe_attr_name}_scaled.png")
            plt.savefig(save_file_scaled, dpi=150, bbox_inches='tight')
            plt.close(fig1)

            # ========== PLOT 2: Original size heatmap ==========
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
            ax2.imshow(attn_mask_original, cmap='jet')
            ax2.axis('off')
            ax2.set_title(f'{attr_name}\n(highest)', fontsize=10)

            plt.tight_layout()
            save_file_original = os.path.join(part_folder, f"{img_name}_{safe_attr_name}_original.png")
            plt.savefig(save_file_original, dpi=150, bbox_inches='tight')
            plt.close(fig2)


def save_aggregated_activation_maps(
    imgs,
    aggregated_heatmaps,
    part_names,
    source_paths,
    batch_idx=0,
    t_mean=(0.5, 0.5, 0.5),
    t_std=(2, 2, 2),
    save_path="",
    normalize_heatmaps=False
):
    """
    Save aggregated activation maps per body part.
    Creates two plots per part:
    1) Scaled heatmap overlayed on image
    2) Original size heatmap

    Creates folder structure: save_path/body_part/part_name_[scaled|original].png

    Args:
        imgs: torch.Tensor of [B, C, H, W] - input images
        aggregated_heatmaps: torch.Tensor of [B, K, H, W] - aggregated activation maps per body part
        part_names: List of K part names
        source_paths: List of image paths
        batch_idx: Which image in batch to visualize
        t_mean: Mean applied during preprocessing
        t_std: Std applied during preprocessing
        save_path: Base path where to save visualizations
        normalize_heatmaps: If True, normalize heatmaps to [0, 1] range per heatmap. If False, use raw values.
    """
    B, K, H_map, W_map = aggregated_heatmaps.shape

    # Get the specific image and its activation maps
    img = imgs[batch_idx]
    heatmaps = aggregated_heatmaps[batch_idx]  # [K, H_map, W_map]
    img_path = source_paths[batch_idx]

    # Extract image identifier from path
    img_name = "_".join(img_path.split(os.sep)[-2:]).rstrip(".jpg")

    # Only have part segmentations for first 70 classes
    # Try to extract class ID, skip if it's not a standard CUB dataset format
    try:
        class_id = int(img_name[:3])
        if class_id > 70:
            return
    except (ValueError, IndexError):
        # Not a standard CUB format (e.g., custom image), continue anyway
        pass

    # Denormalize image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H_img x W_img x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)
    H_img, W_img = img_np.shape[:2]

    # Resize all heatmaps to image size once
    heatmaps_resized = F.interpolate(
        heatmaps.unsqueeze(0),  # [1, K, H_map, W_map]
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # [K, H_img, W_img]

    # Iterate through each body part
    for part_idx, part_name in enumerate(part_names):
        # Create folder for this body part
        part_folder = os.path.join(save_path, part_name)
        os.makedirs(part_folder, exist_ok=True)

        # Get both original and scaled versions
        attn_mask_original = heatmaps[part_idx].cpu().detach().numpy()  # [H_map, W_map]
        attn_mask_scaled = heatmaps_resized[part_idx].cpu().detach().numpy()  # [H_img, W_img]

        # Normalize if requested, otherwise use raw values
        if normalize_heatmaps:
            # Normalize each heatmap individually to [0, 1]
            if attn_mask_original.max() > attn_mask_original.min():
                attn_mask_original = (attn_mask_original - attn_mask_original.min()) / (attn_mask_original.max() - attn_mask_original.min())
            if attn_mask_scaled.max() > attn_mask_scaled.min():
                attn_mask_scaled = (attn_mask_scaled - attn_mask_scaled.min()) / (attn_mask_scaled.max() - attn_mask_scaled.min())

        # Sanitize filename
        safe_part_name = part_name.replace('/', '_').replace(' ', '_')

        # ========== PLOT 1: Scaled heatmap overlayed on image ==========
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.imshow(img_np)
        ax1.imshow(attn_mask_scaled, cmap='jet', alpha=0.5, vmin=0 if not normalize_heatmaps else None, vmax=None)
        ax1.axis('off')
        title_suffix = "(Scaled to Image Size)" if normalize_heatmaps else "(Scaled, Raw Values)"
        ax1.set_title(f'{part_name} (Aggregated)\n{title_suffix}', fontsize=10)

        plt.tight_layout()
        save_file_scaled = os.path.join(part_folder, f"{img_name}_{safe_part_name}_aggregated_scaled.png")
        plt.savefig(save_file_scaled, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ========== PLOT 2: Original size heatmap ==========
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        ax2.imshow(attn_mask_original, cmap='jet', vmin=0 if not normalize_heatmaps else None, vmax=None)
        ax2.axis('off')
        title_suffix = f"(Original Size: {H_map}x{W_map})" if normalize_heatmaps else f"(Original Size: {H_map}x{W_map}, Raw Values)"
        ax2.set_title(f'{part_name} (Aggregated)\n{title_suffix}', fontsize=10)

        plt.tight_layout()
        save_file_original = os.path.join(part_folder, f"{img_name}_{safe_part_name}_aggregated_original.png")
        plt.savefig(save_file_original, dpi=150, bbox_inches='tight')
        plt.close(fig2)


def visualise_part_segmentations(
    imgs,
    saliency_maps,
    seg_masks,
    attribute_names,
    iou_scores,
    source_paths=None,
    attributes=100,
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

        # Only have part segmentations for first 70 classes, so no need to plot for other classes
        class_id = int(img_name[:3])
        if class_id > 70:
            return
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
    #plt.savefig(os.path.join(save_path, f"{img_name}_attr{attr_str}.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f"{img_name}_attr.png"), dpi=200, bbox_inches='tight')
    plt.close()
