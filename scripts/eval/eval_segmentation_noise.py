"""
Segmentation-Based Noise Occlusion Evaluation

Uses ground-truth part segmentation masks to selectively add noise to image regions.
Two conditions per target part:
  1. Background noise:  Add noise to everything outside the bird silhouette.
  2. Isolate part:      Keep ONLY the target part visible; add noise to background + all other parts.

For a model that learned correct features, condition 2 should preserve accuracy
for the target part's concepts while degrading accuracy for other parts' concepts.

Only evaluated on the 70 classes that have part segmentation annotations.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from utils_protocbm.eval_utils import get_localization_loader, create_model_for_eval
from utils_protocbm.train_utils import AverageMeter, accuracy, gather_args
from utils_protocbm.index_translation import map_attribute_ids_from_cub_to_cbm
from utils_protocbm.mappings import (
    PART_SEG_GROUPS,
    MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS,
)


def get_concept_scores(model, inputs, attr_labels, args):
    """
    Run a forward pass and return (class_pred, concept_logits).
    class_pred: [B, C] or None (for XC-only models).
    concept_logits: [B, A] raw logits.
    """
    is_independent = getattr(args, "mode", None) == "independent"

    if args.concept_mapper == "protomod":
        out = model(inputs, attr_labels)
        return out[0], out[1]
    else:
        out = model(inputs, attr_labels)
        if is_independent:
            scores = torch.stack(out, dim=1).squeeze(-1)
            return None, scores
        else:
            class_pred = out[0]
            scores = torch.stack(out[1:], dim=1).squeeze(-1)
            return class_pred, scores


def add_noise_to_region(inputs, mask, noise_std=1.0):
    """
    Add Gaussian noise to image regions indicated by mask.

    Args:
        inputs: [B, 3, H, W] normalized image tensor
        mask: [B, 1, H, W] binary mask (1 = add noise here)
        noise_std: std of Gaussian noise (in normalized image space)
    Returns:
        noisy: [B, 3, H, W] image with noise in masked region
    """
    noise = torch.randn_like(inputs) * noise_std
    return inputs * (1.0 - mask) + (inputs + noise) * mask


def eval_single_noise_level(model, cy_model, loader, device, args, seg_group_to_cbm_attrs, noise_std):
    """Run evaluation for a single noise level. Returns dict of results."""
    is_independent = getattr(args, "mode", None) == "independent"
    is_xc_only = getattr(args, "mode", None) == "XC"
    n_seg_groups = len(PART_SEG_GROUPS)
    min_part_area = getattr(args, "min_part_area", 100)  # pixels

    baseline_correct = torch.zeros(n_seg_groups, device=device)
    isolated_correct = torch.zeros(n_seg_groups, device=device)
    group_counts = torch.zeros(n_seg_groups, device=device)

    overall_baseline = 0.0
    overall_bg_noise = 0.0
    overall_count = 0.0

    class_acc_baseline = AverageMeter()
    class_acc_bg_noise = AverageMeter()

    n_skipped = 0
    n_evaluated = 0
    n_parts_too_small = 0

    for data in tqdm(loader, desc=f"  noise_std={noise_std:.2f}", leave=False):
        data = [v.to(device) if torch.is_tensor(v) else v for v in data]

        if args.dataset == "waterbirds":
            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts, _ = data
        else:
            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data

        attr_labels = torch.stack(attr_labels, dim=1).float().to(device)
        B = inputs.shape[0]

        has_masks = part_seg_masks.sum(dim=(1, 2, 3)) > 0
        if has_masks.sum() == 0:
            n_skipped += B
            continue

        # --- Baseline ---
        pred_base, scores_base = get_concept_scores(model, inputs, attr_labels, args)
        base_preds = (torch.sigmoid(scores_base) >= 0.5).float()

        if not is_xc_only:
            if is_independent:
                cs = torch.sigmoid(scores_base) if getattr(args, "use_sigmoid_logits", True) else scores_base
                class_pred_base = cy_model.classifier(cs)
            else:
                class_pred_base = pred_base
            valid_labels = labels[has_masks]
            if valid_labels.numel() > 0:
                ca = accuracy(class_pred_base[has_masks], valid_labels, topk=(1,))
                class_acc_baseline.update(ca[0], valid_labels.size(0))

        # --- Background noise ---
        bird_mask = part_seg_masks.max(dim=1, keepdim=True)[0]
        bg_mask = 1.0 - bird_mask

        # Skip bg-noise for samples where bird is too small
        bird_area = bird_mask.sum(dim=(1, 2, 3))  # [B]
        has_large_bird = (bird_area >= min_part_area) & has_masks

        inputs_bg_noise = add_noise_to_region(inputs, bg_mask, noise_std)
        pred_bg, scores_bg = get_concept_scores(model, inputs_bg_noise, attr_labels, args)
        bg_preds = (torch.sigmoid(scores_bg) >= 0.5).float()

        if not is_xc_only:
            if is_independent:
                cs_bg = torch.sigmoid(scores_bg) if getattr(args, "use_sigmoid_logits", True) else scores_bg
                class_pred_bg = cy_model.classifier(cs_bg)
            else:
                class_pred_bg = pred_bg
            valid_labels_bg = labels[has_large_bird]
            if valid_labels_bg.numel() > 0:
                ca_bg = accuracy(class_pred_bg[has_large_bird], valid_labels_bg, topk=(1,))
                class_acc_bg_noise.update(ca_bg[0], valid_labels_bg.size(0))

        for g_idx, group in enumerate(PART_SEG_GROUPS):
            attr_idx = seg_group_to_cbm_attrs[group]
            if len(attr_idx) == 0:
                continue
            for ai in attr_idx:
                gt = attr_labels[has_large_bird, ai]
                bp = base_preds[has_large_bird, ai]
                bgp = bg_preds[has_large_bird, ai]
                overall_baseline += (bp == gt).float().sum().item()
                overall_bg_noise += (bgp == gt).float().sum().item()
                overall_count += gt.shape[0]

        # --- Part isolation ---
        # Per-part area: [B, 8]
        part_areas = part_seg_masks.sum(dim=(2, 3))  # [B, 8]

        for g_idx, group in enumerate(PART_SEG_GROUPS):
            attr_idx = seg_group_to_cbm_attrs[group]
            if len(attr_idx) == 0:
                continue

            target_mask = part_seg_masks[:, g_idx:g_idx+1, :, :]
            # Part must exist AND be large enough
            has_part = (part_areas[:, g_idx] >= min_part_area) & has_masks  # [B]
            n_parts_too_small += ((part_areas[:, g_idx] > 0) & (part_areas[:, g_idx] < min_part_area) & has_masks).sum().item()

            if has_part.sum() == 0:
                continue

            noise_region = 1.0 - target_mask
            inputs_isolated = add_noise_to_region(inputs, noise_region, noise_std)

            _, scores_iso = get_concept_scores(model, inputs_isolated, attr_labels, args)
            iso_preds = (torch.sigmoid(scores_iso) >= 0.5).float()

            for ai in attr_idx:
                gt = attr_labels[has_part, ai]
                bp = base_preds[has_part, ai]
                ip = iso_preds[has_part, ai]

                baseline_correct[g_idx] += (bp == gt).float().sum()
                isolated_correct[g_idx] += (ip == gt).float().sum()
                group_counts[g_idx] += gt.shape[0]

        n_evaluated += has_masks.sum().item()

    return {
        'noise_std': noise_std,
        'n_evaluated': n_evaluated,
        'n_skipped': n_skipped,
        'n_parts_too_small': n_parts_too_small,
        'min_part_area': min_part_area,
        'baseline_correct': baseline_correct,
        'isolated_correct': isolated_correct,
        'group_counts': group_counts,
        'overall_baseline': overall_baseline,
        'overall_bg_noise': overall_bg_noise,
        'overall_count': overall_count,
        'class_acc_baseline': class_acc_baseline,
        'class_acc_bg_noise': class_acc_bg_noise,
    }


def print_results_for_level(results, is_xc_only):
    """Print detailed results for a single noise level."""
    noise_std = results['noise_std']
    n_seg_groups = len(PART_SEG_GROUPS)

    print(f"\n{'─' * 80}")
    print(f"  Noise std = {noise_std:.3f}")
    print(f"  Images evaluated: {results['n_evaluated']}  (skipped {results['n_skipped']} w/o seg masks)")
    print(f"  Min part area: {results['min_part_area']} px  (filtered {results['n_parts_too_small']} too-small part instances)")
    print(f"{'─' * 80}")

    print(f"\n  {'Part Group':<12} {'Baseline':>10} {'Isolated':>10} {'Delta':>10} {'Count':>8}")
    print(f"  {'-' * 54}")

    total_base = 0.0
    total_iso = 0.0
    total_count = 0.0

    for g_idx, group in enumerate(PART_SEG_GROUPS):
        gc = results['group_counts'][g_idx]
        if gc == 0:
            continue
        ba = (results['baseline_correct'][g_idx] / gc).item() * 100
        ia = (results['isolated_correct'][g_idx] / gc).item() * 100
        print(f"  {group:<12} {ba:>9.2f}% {ia:>9.2f}% {ia - ba:>+9.2f}% {int(gc.item()):>8}")
        total_base += results['baseline_correct'][g_idx].item()
        total_iso += results['isolated_correct'][g_idx].item()
        total_count += gc.item()

    print(f"  {'-' * 54}")
    if total_count > 0:
        b = total_base / total_count * 100
        i = total_iso / total_count * 100
        print(f"  {'OVERALL':<12} {b:>9.2f}% {i:>9.2f}% {i - b:>+9.2f}% {int(total_count):>8}")

    if results['overall_count'] > 0:
        bg_base = results['overall_baseline'] / results['overall_count'] * 100
        bg_noise = results['overall_bg_noise'] / results['overall_count'] * 100
        print(f"\n  Background noise:  {bg_base:.2f}% -> {bg_noise:.2f}%  (delta {bg_noise - bg_base:+.2f}%)")

    if not is_xc_only:
        cb = results['class_acc_baseline'].avg.item()
        cn = results['class_acc_bg_noise'].avg.item()
        print(f"  Class acc:         {cb:.4f} -> {cn:.4f}  (delta {cn - cb:+.4f})")


def print_summary_table(all_results, is_xc_only):
    """Print a compact comparison table across all noise levels."""
    n_seg_groups = len(PART_SEG_GROUPS)

    print("\n\n" + "=" * 100)
    print("SUMMARY: CONCEPT ACCURACY ACROSS NOISE LEVELS")
    print("=" * 100)

    # Header
    noise_stds = [r['noise_std'] for r in all_results]
    header = f"{'Part Group':<12} {'Baseline':>10}"
    for ns in noise_stds:
        header += f" {'σ=' + f'{ns:.2f}':>10}"
    print(f"\n--- Per-Part Isolation (only target part visible) ---")
    print(header)
    print("-" * (24 + 11 * len(noise_stds)))

    for g_idx, group in enumerate(PART_SEG_GROUPS):
        gc = all_results[0]['group_counts'][g_idx]
        if gc == 0:
            continue
        ba = (all_results[0]['baseline_correct'][g_idx] / gc).item() * 100
        row = f"{group:<12} {ba:>9.2f}%"
        for r in all_results:
            ia = (r['isolated_correct'][g_idx] / r['group_counts'][g_idx]).item() * 100
            row += f" {ia:>9.2f}%"
        print(row)

    # Overall row
    row = f"{'OVERALL':<12}"
    for i, r in enumerate(all_results):
        total_base = sum(r['baseline_correct'][g].item() for g in range(n_seg_groups))
        total_iso = sum(r['isolated_correct'][g].item() for g in range(n_seg_groups))
        total_count = sum(r['group_counts'][g].item() for g in range(n_seg_groups))
        if total_count > 0:
            if i == 0:
                row = f"{'OVERALL':<12} {total_base / total_count * 100:>9.2f}%"
            val = total_iso / total_count * 100
            row += f" {val:>9.2f}%"
    print("-" * (24 + 11 * len(noise_stds)))
    print(row)

    # Background noise summary
    print(f"\n--- Background Noise (concept acc) ---")
    header_bg = f"{'':>12} {'Baseline':>10}"
    for ns in noise_stds:
        header_bg += f" {'σ=' + f'{ns:.2f}':>10}"
    print(header_bg)
    oc = all_results[0]['overall_count']
    if oc > 0:
        bg_base = all_results[0]['overall_baseline'] / oc * 100
        row_bg = f"{'Concept':>12} {bg_base:>9.2f}%"
        for r in all_results:
            bg_n = r['overall_bg_noise'] / r['overall_count'] * 100
            row_bg += f" {bg_n:>9.2f}%"
        print(row_bg)

    if not is_xc_only:
        cb = all_results[0]['class_acc_baseline'].avg.item() * 100
        row_cls = f"{'Class':>12} {cb:>9.2f}%"
        for r in all_results:
            cn = r['class_acc_bg_noise'].avg.item() * 100
            row_cls += f" {cn:>9.2f}%"
        print(row_cls)

    print("=" * 100)


def eval_segmentation_noise(args):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    is_independent = getattr(args, "mode", None) == "independent"
    is_xc_only = getattr(args, "mode", None) == "XC"

    model, device, cy_model = create_model_for_eval(args)
    model.eval()
    if cy_model is not None:
        cy_model.eval()

    loader, transform_mean, transform_std, img_size = get_localization_loader(
        model, args.data_dir, args.split_dir, args
    )

    # Noise scales to evaluate
    noise_scales = getattr(args, "noise_scales", None)
    if noise_scales is None:
        noise_scales = [0.05, 0.1, 0.2, 0.5, 1.0]

    n_seg_groups = len(PART_SEG_GROUPS)

    # Map seg group -> CBM attribute indices
    seg_group_to_cbm_attrs = {}
    for group in PART_SEG_GROUPS:
        cub_ids = MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS.get(group, [])
        cbm_ids = map_attribute_ids_from_cub_to_cbm(cub_ids)
        seg_group_to_cbm_attrs[group] = torch.tensor(cbm_ids, dtype=torch.long, device=device)

    min_part_area = getattr(args, "min_part_area", 100)
    img_total_px = img_size * img_size

    print(f"Evaluating {len(noise_scales)} noise levels: {noise_scales}")
    print(f"Part segmentation groups: {PART_SEG_GROUPS}")
    print(f"Min part area threshold: {min_part_area} px ({min_part_area / img_total_px * 100:.2f}% of image)")

    # Collect part area statistics (first pass through data)
    print("\n--- Part Segmentation Area Statistics (pixels) ---")
    area_stats = {g: [] for g in PART_SEG_GROUPS}
    for data in tqdm(loader, desc="  Computing area stats", leave=False):
        data = [v.to(device) if torch.is_tensor(v) else v for v in data]
        if args.dataset == "waterbirds":
            part_seg_masks = data[3]
        else:
            part_seg_masks = data[3]
        has_masks = part_seg_masks.sum(dim=(1, 2, 3)) > 0
        for g_idx, group in enumerate(PART_SEG_GROUPS):
            areas = part_seg_masks[has_masks, g_idx].sum(dim=(1, 2))
            nonzero = areas[areas > 0]
            if len(nonzero) > 0:
                area_stats[group].extend(nonzero.cpu().tolist())

    print(f"  {'Part':<10} {'Count':>7} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'<thresh':>8}")
    print(f"  {'-' * 58}")
    for group in PART_SEG_GROUPS:
        vals = area_stats[group]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        n_small = (arr < min_part_area).sum()
        print(f"  {group:<10} {len(vals):>7} {arr.mean():>8.0f} {np.median(arr):>8.0f} {arr.min():>8.0f} {arr.max():>8.0f} {n_small:>8}")

    all_results = []
    for ns in noise_scales:
        print(f"\n>>> Running noise_std = {ns:.3f}")
        results = eval_single_noise_level(
            model, cy_model, loader, device, args, seg_group_to_cbm_attrs, ns
        )
        all_results.append(results)
        print_results_for_level(results, is_xc_only)

    # Print compact summary across all levels
    print_summary_table(all_results, is_xc_only)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = gather_args()

    out_folder_path = os.path.join(args.log_dir, f"seg_noise_occlusion_{args.dataset}")
    os.makedirs(out_folder_path, exist_ok=True)

    path_to_output_txt = os.path.join(out_folder_path, "seg_noise_eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")
    sys.stdout = open(path_to_output_txt, 'a')

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    eval_segmentation_noise(args)
