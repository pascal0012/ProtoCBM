"""
Generate presentation-quality visualizations of the segmentation-based noise occlusion.

Produces:
  1. Overview figure: Original | Bird silhouette | Background noise
  2. Per-part isolation grid: one row per part showing [Original | Part mask | Isolated (only this part clean)]
  3. Summary grid: columns = samples, rows = [Original, Bg noise, Head isolated, Wing isolated, ...]

Usage:
    python visualize_seg_noise_samples.py --config configs/protocbm/eval/occlusion/protocbm_independent.yaml
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils_protocbm.eval_utils import get_localization_loader, create_model_for_eval
from utils_protocbm.train_utils import gather_args
from utils_protocbm.mappings import PART_SEG_GROUPS
from eval_segmentation_noise import add_noise_to_region


def denormalize(img_tensor, mean, std):
    mean = torch.tensor(mean, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def visualize_per_part_isolation(img_np, part_seg_masks, noise_std, img_tensor,
                                 transform_mean, transform_std, save_path):
    """
    One row per part: [Original] [Part mask (green overlay)] [Isolated image]
    """
    visible_parts = []
    for g_idx, group in enumerate(PART_SEG_GROUPS):
        if part_seg_masks[g_idx].sum() > 0:
            visible_parts.append((g_idx, group))

    if not visible_parts:
        return

    n_rows = len(visible_parts)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    mask_cmap = LinearSegmentedColormap.from_list("part", [
        (0, 0, 0, 0), (0.2, 0.8, 0.3, 0.6)
    ])

    for row, (g_idx, group) in enumerate(visible_parts):
        target_mask = part_seg_masks[g_idx]  # [H, W]
        noise_region = 1.0 - target_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        isolated = add_noise_to_region(
            img_tensor.unsqueeze(0), noise_region.to(img_tensor.device), noise_std
        )
        isolated_denorm = denormalize(isolated[0], transform_mean, transform_std)
        isolated_np = isolated_denorm.cpu().permute(1, 2, 0).numpy()

        # Original
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(group, fontsize=12, rotation=0, labelpad=50, va='center',
                                fontweight='bold')
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=12)

        # Mask overlay
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(target_mask.cpu().numpy(), cmap=mask_cmap, vmin=0, vmax=1)
        if row == 0:
            axes[row, 1].set_title("Part mask", fontsize=12)

        # Isolated
        axes[row, 2].imshow(np.clip(isolated_np, 0, 1))
        if row == 0:
            axes[row, 2].set_title("Isolated (noise elsewhere)", fontsize=12)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Part Isolation via Segmentation Masks", fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-part isolation: {save_path}")


def visualize_bg_noise(img_np, bird_mask, noise_std, img_tensor,
                       transform_mean, transform_std, save_path):
    """
    Single row: [Original] [Bird silhouette] [Background noise]
    """
    bg_mask = 1.0 - bird_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    noisy = add_noise_to_region(
        img_tensor.unsqueeze(0), bg_mask.to(img_tensor.device), noise_std
    )
    noisy_denorm = denormalize(noisy[0], transform_mean, transform_std)
    noisy_np = noisy_denorm.cpu().permute(1, 2, 0).numpy()

    silhouette_cmap = LinearSegmentedColormap.from_list("sil", [
        (0, 0, 0, 0), (0.2, 0.6, 1.0, 0.5)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=12)

    axes[1].imshow(img_np)
    axes[1].imshow(bird_mask.cpu().numpy(), cmap=silhouette_cmap, vmin=0, vmax=1)
    axes[1].set_title("Bird silhouette", fontsize=12)

    axes[2].imshow(np.clip(noisy_np, 0, 1))
    axes[2].set_title("Background noise", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Background Noise Occlusion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved bg noise: {save_path}")


def visualize_summary_grid(images_data, noise_std, save_path, n_cols=4):
    """
    Summary grid for presentation:
    Row 0: Original
    Row 1: Background noise
    Row 2-N: Part isolated (one row per part that exists in at least one sample)
    """
    n = min(n_cols, len(images_data))

    # Find parts present across samples
    all_parts = set()
    for d in images_data[:n]:
        for g_idx in range(len(PART_SEG_GROUPS)):
            if d['seg_masks'][g_idx].sum() > 0:
                all_parts.add(g_idx)
    part_order = sorted(all_parts)

    n_rows = 2 + len(part_order)
    fig, axes = plt.subplots(n_rows, n, figsize=(3.5 * n, 2.8 * n_rows))
    if n == 1:
        axes = axes[:, None]

    row_labels = ["Original", "Bg noise"] + [PART_SEG_GROUPS[g] + " only" for g in part_order]

    for col in range(n):
        d = images_data[col]
        img_np = d['img_np']
        img_t = d['img_tensor']
        seg = d['seg_masks']
        mean, std = d['mean'], d['std']
        dev = img_t.device

        bird_mask = seg.max(dim=0)[0]  # [H, W]

        # Row 0: original
        axes[0, col].imshow(img_np)

        # Row 1: bg noise
        bg = 1.0 - bird_mask.unsqueeze(0).unsqueeze(0)
        noisy = add_noise_to_region(img_t.unsqueeze(0), bg.to(dev), noise_std)
        noisy_np = denormalize(noisy[0], mean, std).cpu().permute(1, 2, 0).numpy()
        axes[1, col].imshow(np.clip(noisy_np, 0, 1))

        # Rows 2+: isolated parts
        for row_off, g_idx in enumerate(part_order):
            row = 2 + row_off
            target = seg[g_idx]  # [H, W]
            if target.sum() > 0:
                noise_region = 1.0 - target.unsqueeze(0).unsqueeze(0)
                iso = add_noise_to_region(img_t.unsqueeze(0), noise_region.to(dev), noise_std)
                iso_np = denormalize(iso[0], mean, std).cpu().permute(1, 2, 0).numpy()
                axes[row, col].imshow(np.clip(iso_np, 0, 1))
            else:
                axes[row, col].imshow(np.ones_like(img_np) * 0.85)
                axes[row, col].text(0.5, 0.5, "N/A", ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=11, color='gray')

    for row in range(n_rows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10, rotation=0, labelpad=65, va='center',
                                fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Segmentation-Based Noise Occlusion Overview", fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved summary grid: {save_path}")


def visualize_noise_levels_single(img_np, bird_mask, part_seg_masks, noise_scales,
                                   img_tensor, transform_mean, transform_std, save_path):
    """
    Grid showing noise levels for one sample.
    Row 0: Background noise at each scale
    Row 1+: Part isolation at each scale (one row per visible part)
    First column: Original image (no noise)
    """
    visible_parts = []
    for g_idx, group in enumerate(PART_SEG_GROUPS):
        if part_seg_masks[g_idx].sum() > 0:
            visible_parts.append((g_idx, group))

    n_scales = len(noise_scales)
    n_rows = 1 + len(visible_parts)  # bg noise + parts
    n_cols = 1 + n_scales  # original + each scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    dev = img_tensor.device

    # Row 0: Background noise
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].set_ylabel("Bg noise", fontsize=10, rotation=0, labelpad=55, va='center', fontweight='bold')

    bg_region = 1.0 - bird_mask.unsqueeze(0).unsqueeze(0).to(dev)
    for s_idx, scale in enumerate(noise_scales):
        col = 1 + s_idx
        noisy = add_noise_to_region(img_tensor.unsqueeze(0), bg_region, scale)
        noisy_np = denormalize(noisy[0], transform_mean, transform_std).cpu().permute(1, 2, 0).numpy()
        axes[0, col].imshow(np.clip(noisy_np, 0, 1))
        axes[0, col].set_title(f"σ = {scale}", fontsize=10)

    # Rows 1+: Part isolation
    for row_off, (g_idx, group) in enumerate(visible_parts):
        row = 1 + row_off
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"{group}\nonly", fontsize=10, rotation=0, labelpad=55, va='center', fontweight='bold')

        target = part_seg_masks[g_idx]
        noise_region = (1.0 - target.unsqueeze(0).unsqueeze(0)).to(dev)
        for s_idx, scale in enumerate(noise_scales):
            col = 1 + s_idx
            iso = add_noise_to_region(img_tensor.unsqueeze(0), noise_region, scale)
            iso_np = denormalize(iso[0], transform_mean, transform_std).cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(np.clip(iso_np, 0, 1))

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Noise Level Comparison", fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved noise levels: {save_path}")


def visualize_noise_levels_grid(images_data, noise_scales, save_path, n_samples=4):
    """
    Compact grid: columns = noise scales, rows = samples.
    Shows background noise across scales for multiple samples.
    """
    n = min(n_samples, len(images_data))
    n_cols = 1 + len(noise_scales)  # original + scales

    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row in range(n):
        d = images_data[row]
        img_np, img_t = d['img_np'], d['img_tensor']
        bird_mask = d['bird_mask']
        mean, std = d['mean'], d['std']
        dev = img_t.device

        bg_region = 1.0 - bird_mask.unsqueeze(0).unsqueeze(0).to(dev)

        axes[row, 0].imshow(img_np)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=10)

        for s_idx, scale in enumerate(noise_scales):
            col = 1 + s_idx
            noisy = add_noise_to_region(img_t.unsqueeze(0), bg_region, scale)
            noisy_np = denormalize(noisy[0], mean, std).cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(np.clip(noisy_np, 0, 1))
            if row == 0:
                axes[row, col].set_title(f"σ = {scale}", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Background Noise — Scale Comparison", fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved noise levels grid: {save_path}")


if __name__ == '__main__':
    args = gather_args()

    model, device, _ = create_model_for_eval(args)
    model.eval()

    loader, transform_mean, transform_std, img_size = get_localization_loader(
        model, args.data_dir, args.split_dir, args
    )

    noise_std = getattr(args, "noise_std", 1.0)
    noise_scales = getattr(args, "noise_scales", [noise_std])
    n_samples = getattr(args, "vis_n_samples", 6)

    save_dir = os.path.join(args.log_dir, "seg_noise_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    collected = []
    sample_idx = 0

    with torch.no_grad():
        for data in loader:
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]
            if args.dataset == "waterbirds":
                inputs, labels, attr_labels, part_seg_masks, _, source_paths, _, _ = data
            else:
                inputs, labels, attr_labels, part_seg_masks, _, source_paths, _ = data

            B = inputs.shape[0]
            for b in range(B):
                if sample_idx >= n_samples:
                    break

                # Skip samples without seg masks (class > 70)
                if part_seg_masks[b].sum() == 0:
                    continue

                img_denorm = denormalize(inputs[b], transform_mean, transform_std)
                img_np = img_denorm.cpu().permute(1, 2, 0).numpy()
                bird_mask = part_seg_masks[b].max(dim=0)[0]  # [H, W]

                entry = {
                    'img_np': img_np,
                    'img_tensor': inputs[b],
                    'seg_masks': part_seg_masks[b],
                    'bird_mask': bird_mask,
                    'mean': transform_mean,
                    'std': transform_std,
                }
                collected.append(entry)

                # Per-sample figures for first 2
                if sample_idx < 2:
                    print(f"Sample {sample_idx}:")
                    visualize_per_part_isolation(
                        img_np, part_seg_masks[b], noise_std, inputs[b],
                        transform_mean, transform_std,
                        os.path.join(save_dir, f"per_part_sample_{sample_idx}.png")
                    )
                    visualize_bg_noise(
                        img_np, bird_mask, noise_std, inputs[b],
                        transform_mean, transform_std,
                        os.path.join(save_dir, f"bg_noise_sample_{sample_idx}.png")
                    )
                    # Noise level comparison per sample
                    visualize_noise_levels_single(
                        img_np, bird_mask, part_seg_masks[b], noise_scales,
                        inputs[b], transform_mean, transform_std,
                        os.path.join(save_dir, f"noise_levels_sample_{sample_idx}.png")
                    )

                sample_idx += 1

            if sample_idx >= n_samples:
                break

    if len(collected) > 0:
        visualize_summary_grid(
            collected, noise_std,
            os.path.join(save_dir, "summary_grid.png"),
            n_cols=min(4, len(collected))
        )
        # Multi-sample noise level comparison grid
        visualize_noise_levels_grid(
            collected, noise_scales,
            os.path.join(save_dir, "noise_levels_grid.png"),
            n_samples=min(4, len(collected))
        )

    print(f"\nAll visualizations saved to: {save_dir}")
