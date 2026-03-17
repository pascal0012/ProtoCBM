import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


def visualize_keypoints_to_figure(
    imgs:torch.Tensor,
    part_gts:torch.Tensor,
    attention_maps:torch.Tensor,
    similarity_scores:torch.Tensor,
    part_dict: dict,
    part_attribute_mapping: dict,
    img_size: int,
    t_mean: tuple=(0.5, 0.5, 0.5),
    t_std: tuple=(2, 2, 2),
) -> plt.Figure:
    """
    Visualize GT vs predicted keypoints for a batch of images.
    Predicted keypoints are derived from attention maps using per-part argmax.

    Args:
        imgs:                [N, C, H, W] normalized image tensors
        part_gts:            [N, K, 2] ground truth keypoint coordinates
        attention_maps:      [N, A, H, W] attention maps from the model
        similarity_scores:   [N, A] per-attribute similarity scores
        part_dict:           dict mapping part_id -> part_name (15 CUB parts)
        part_attribute_mapping: dict mapping part_name -> tensor of attribute indices
        img_size:            int, image resolution (299 or 224)
        t_mean:              tuple, normalization mean for denormalization
        t_std:               tuple, normalization std for denormalization

    Returns:
        matplotlib.Figure with N rows x 15 columns showing GT (green) vs pred (red)
    """
    N = imgs.shape[0]
    K = len(part_dict)
    part_names = list(part_dict.values())

    # --- Compute predicted keypoints per part (argmax approach) ---
    # For each part, pick the attribute with the highest similarity score
    argmax_per_part = []
    for part_name in part_names:
        attrs = part_attribute_mapping[part_name].cpu()
        subset = similarity_scores[:, attrs]  # [N, n_attrs_for_part]
        argmax_in_subset = subset.argmax(dim=1)  # [N]
        result = attrs[argmax_in_subset]  # [N] global attribute indices
        argmax_per_part.append(result)

    idx = torch.stack(argmax_per_part).t()  # [N, K]
    batch_indices = torch.arange(N).unsqueeze(1)
    heatmaps = attention_maps[batch_indices, idx]  # [N, K, H_att, W_att]

    # Resize heatmaps to image resolution
    resized = F.interpolate(
        heatmaps.float(), size=img_size, mode="bilinear", align_corners=False
    )  # [N, K, img_size, img_size]

    # Argmax to get predicted (x, y) coordinates
    flat = resized.view(N, K, -1)
    max_idx = flat.argmax(dim=2)  # [N, K]
    W = resized.shape[-1]
    pred_x = max_idx % W
    pred_y = max_idx // W
    predicted_coords = torch.stack((pred_x, pred_y), dim=2)  # [N, K, 2]

    # --- Create figure: N rows x 15 columns ---
    fig, axes = plt.subplots(
        N, min(K, 15), figsize=(3 * min(K, 15), 3 * N), squeeze=False
    )

    for row in range(N):
        # Denormalize image
        img_np = imgs[row].permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array(t_std) + np.array(t_mean)
        img_np = np.clip(img_np, 0, 1)

        for col in range(min(K, 15)):
            ax = axes[row, col]
            ax.imshow(img_np)
            ax.axis("off")

            gx, gy = part_gts[row, col].cpu().numpy()
            px, py = predicted_coords[row, col].cpu().numpy()

            if gx == 0 and gy == 0:
                ax.set_title(f"{part_names[col]}\n(not visible)", fontsize=7)
                continue

            # GT keypoint (green circle)
            ax.scatter(gx, gy, c="lime", s=40, marker="o", zorder=5)
            # Predicted keypoint (red X)
            ax.scatter(px, py, c="red", s=40, marker="x", zorder=5)
            # Dashed line between them
            ax.plot([gx, px], [gy, py], "w--", linewidth=1.5)

            dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            ax.set_title(f"{part_names[col]}\nd={dist:.1f}", fontsize=7)

    fig.tight_layout()
    return fig


def visualize_keypoint_distances(gts: torch.Tensor,
                                imgs:torch.Tensor | np.ndarray,
                                preds:torch.Tensor,
                                dists:torch.Tensor,
                                batch_nr:int,
                                part_names:list,
                                batch_idx:int =None,
                                t_mean:tuple =(0.5, 0.5, 0.5),
                                t_std:tuple =(2, 2, 2),
                                save_path=""):
    """
    Visualize Distance predictions on sampled image for a batch, used for eval.

    gt (torch.Tensor): (B,15,2), ground truth xy of parts
    imgs: numpy or torch image [B,H,W,3]  original images
    pred (torch.Tensor): (B,15,2), predicted xy of parts
    dists (torch.Tensor): (B,15), distances for each part prediction vs GT
    batch_nr (int): the batch number we are looking at
    part_names (list[str]): list of part names for the 15 CUB parts, in order of the tensors
    batch_idx (int or None): if int, the index in the batch to visualize, if None then randomly sample one from the batch
    t_mean (tuple): the mean and std used for normalizing the images
    save_path (str): where to save the visualizations
    """

    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        B, _, _, _ = imgs.shape
        batch_idx = random.randint(0, B-1)

    #sample
    gt = gts[batch_idx]
    img = imgs[batch_idx]
    
    pred = preds[batch_idx]
    dists = dists[batch_idx]

    # Convert tensors to numpy
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    dists_np = dists.cpu().numpy()

    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    # Make 15 subplots
    _, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(15):
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis("off")

        gx, gy = gt_np[i]
        px, py = pred_np[i]

        #skip nonexistent parts
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



def plot_threshold_curve(thresholds:np.ndarray, 
                         accs: list, 
                         save_path:str =""):
    """Make threshold-mIoU curve and calculate AUC

    Args:
        thresholds (np.array[float]): threshold value for curve
        accs (list[float]): accuracies at threshold values
        save_path (str, optional): Save path for ROC figure. Defaults to "".
    """

    #conversion
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



def create_attribute_mosaic(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    attribute_names: list,
    scores: np.ndarray | torch.Tensor,
    alpha:float =0.5,
    resize_to:tuple =None,
    font_path:str=None,
    font_size:int=20,
    header_height:int=40,
    score_height:int=30,
):
    """This method takes a batch of images, corresponding heatmaps and scores, and creates a large mosaic
    visualizing the heatmaps overlaid on the images, with attribute names as column headers and scores below each image.

    Args:
        images (torch.Tensor[float]): original images, shape (B, C, H, W)
        heatmaps (torch.Tensor[float]): heatmaps to overlay, shape (B, A, H_hm, W_hm)
        attribute_names (list[str]): list of attribute names, length A
        scores (np.ndarray[float] | torch.Tensor[float]): attribute scores for each image, shape (B, A)
        alpha (float, optional): transparency for heatmap overlay. Defaults to 0.5.
        resize_to (tuple, optional): target resize dimensions. Defaults to None.
        font_path (str, optional): path to font file. Defaults to None.
        font_size (int, optional): size of font. Defaults to 20.
        header_height (int, optional): height of attribute name headers. Defaults to 40.
        score_height (int, optional): height of score displays. Defaults to 30.
    """

    I, _, H, W = images.shape
    _, A, H_hm, W_hm = heatmaps.shape

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()


    # Resize heatmaps to match image resolution if needed
    if (H_hm != H) or (W_hm != W):
        heatmaps = F.interpolate(
            heatmaps,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

    # Convert to PIL
    proc_images = [tensor_to_pil(images[i]) for i in range(I)]

    # Resize if given
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

    # Prepare output mosaic canvas
    cell_w, cell_h = W, H
    mosaic_w = A * cell_w
    mosaic_h = I * (cell_h + score_height) + header_height

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), "white")
    draw = ImageDraw.Draw(mosaic)

    # Draw attribute column headers
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

    # draw cells
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



def plt_colormap(x: np.ndarray[float]) -> np.ndarray[float]:
    """Simple jet-like colormap for heatmaps."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def tensor_to_pil(img_tensor:torch.Tensor) -> Image.Image:
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


def visualise_part_segmentations(
    imgs:torch.Tensor,
    saliency_maps:torch.Tensor,
    seg_masks:torch.Tensor,
    attribute_names:list[str],
    iou_scores:torch.Tensor,
    source_paths:list[str]=None,
    attributes=100,
    batch_idx=None,
    t_mean=(0.5, 0.5, 0.5),
    t_std=(2, 2, 2),
    save_path="",
    preds=None
):
    """
        imgs(torch.Tensor): [B, C, H, W] the respective images
        saliency_maps(torch.Tensor): [B, A, H, W] of the saliency maps matched to the segmentation mask shape, per attribute A
        seg_masks(torch.Tensor): [B, A, H, W] the segmentation masks, per attribute A
        attribute_names(list[str]): List of [A], giving each attribute its name
        iou_scores(torch.Tensor): The IoU scores for each img, per attribute [B, A]
        source_paths(list[str]): The paths to the image sources, will be rendered into the image if given
        attributes(int or list[int]): Either a list of attributes that are to be visualized, or a number of how many random attributes to sample.
        batch_idx(int): The index in the batch from which we extract images, defaults to random one
        t_mean(tuple): The mean applied during preprocessing of the image
        t_std(tuple): The std applied during preprocessing of the image
        save_path(str): Path where to save the visualizations
        pred_mask(torch.Tensor): Optional mask with attribute predictions, if given then only plot the ones where we predict an attribute to be present
    """

    B, A, _, _ = saliency_maps.shape

    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)
    if isinstance(attributes, int):
        attributes = random.sample(range(A), attributes)
    n_attributes = len(attributes)

    #mask out attributes predicted as absent if preds are given
    pred_mask = None
    if preds is not None:
        preds = preds[batch_idx][attributes]
        pred_mask = (preds >= 0.5).float()

    #sample
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

    if pred_mask is not None:
        assert len(pred_mask) == n_attributes, f"pred_mask should have length equal to number of attributes, but have {len(pred_mask)} and {n_attributes}"
        n_plot_attributes = int(pred_mask.sum().item())
        if n_plot_attributes == 0:
            print(f"No predicted attributes for image {img_name}")
            return
    else:
        n_plot_attributes = n_attributes


    _, axes = plt.subplots(2, n_plot_attributes, figsize=(2*n_plot_attributes, 5))

    #iterate only over attributes that we want to keep for the plot
    i = 0
    for col in range(n_attributes):

        if pred_mask is not None and pred_mask[col] == 0:
            continue

        # Attribute name along with the IoU score for that (gt, pred) pair
        axes[0, i].set_title(f"{attr_names[col]}\n{ious[col]:.4f}\n{preds[col]:.4f}", fontsize=9, pad=4)

        # Attention mask overlay
        attn_mask = masks[col].cpu().detach().numpy()
        axes[0, i].imshow(img_np)
        axes[0, i].imshow(attn_mask, cmap='jet', alpha=0.5)
        axes[0, i].axis('off')

        # Segmentation mask overlay
        seg_mask = seg_masks[col].cpu().detach().numpy()
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(seg_mask, cmap='spring', alpha=0.5)
        axes[1, i].axis('off')

        i = i + 1

    axes[0, 0].set_ylabel("ProtoNet Map")
    axes[1, 0].set_ylabel("Part Segmentation")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{img_name}_attr.png"), dpi=200, bbox_inches='tight')
    plt.close()