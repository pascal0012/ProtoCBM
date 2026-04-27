"""
Concept Occlusion Evaluation

For each image, generate a Gaussian mask (sigma=1) centered on the GT keypoint
of each part. Occlude (zero out) the corresponding image region and measure how
concept prediction accuracy drops.

If the model has genuinely learned to use the correct image features for a
concept, occluding the ground-truth region should cause a larger accuracy drop
than for a model relying on spurious features.

Metrics reported per-part and overall:
  - Baseline concept accuracy (no occlusion)
  - Occluded concept accuracy (region masked)
  - Delta (drop in accuracy)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils_protocbm.eval_utils import get_localization_loader, create_model_for_eval
from utils_protocbm.train_utils import AverageMeter, accuracy, gather_args
from utils_protocbm.index_translation import map_attribute_ids_from_cub_to_cbm
from utils_protocbm.mappings import MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS


def get_concept_scores(model, inputs, attr_labels, args):
    """
    Run a forward pass and return (class_pred, concept_logits).
    class_pred: [B, C] or None (for XC-only models).
    concept_logits: [B, A] raw logits.
    """
    is_independent = getattr(args, "mode", None) == "independent"

    if args.concept_mapper == "protomod":
        # ProtoCBM: (class_pred_or_concepts, similarity_scores, attention_maps)
        out = model(inputs, attr_labels)
        return out[0], out[1]
    else:
        # CBM model
        out = model(inputs, attr_labels)
        if is_independent:
            # XC mode: returns list of [B, 1] per-attribute logits
            scores = torch.stack(out, dim=1).squeeze(-1)  # [B, A]
            return None, scores
        else:
            # XCY mode: [class_logits, attr1, ..., attrN]
            class_pred = out[0]
            scores = torch.stack(out[1:], dim=1).squeeze(-1)  # [B, A]
            return class_pred, scores


def build_gaussian_occlusion_mask(part_gts, img_size, sigma=1.0):
    """
    Build per-part Gaussian occlusion masks in image space.

    Args:
        part_gts: [B, K, 2] keypoint coordinates (x, y) in image space
        img_size: int, spatial size of the (square) image
        sigma: std of the Gaussian in feature-map space (scaled to pixel space internally)

    Returns:
        masks: [B, K, H, W] Gaussian masks in [0, 1], 1 at the center
        valid: [B, K] boolean, True if the part is visible
    """
    B, K, _ = part_gts.shape
    device = part_gts.device
    H = W = img_size

    # Coordinate grids
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    centers = part_gts.float()  # [B, K, 2]
    diff = grid.unsqueeze(0).unsqueeze(0) - centers[:, :, None, None, :]  # [B, K, H, W, 2]

    # Sigma in pixel space: 1 feature-map pixel = img_size/8 image pixels
    pixel_sigma = sigma * (img_size / 8.0)
    masks = torch.exp(-diff.pow(2).sum(-1) / (2 * pixel_sigma ** 2))  # [B, K, H, W]

    # Validity: parts with [0,0] are invisible
    valid = (part_gts.abs().sum(dim=-1) > 0)  # [B, K]

    return masks, valid


def eval_concept_occlusion(args):
    """Main evaluation loop with concept occlusion."""

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    is_independent = getattr(args, "mode", None) == "independent"
    is_xc_only = getattr(args, "mode", None) == "XC"

    # Create models
    model, device, cy_model = create_model_for_eval(args)
    model.eval()
    if cy_model is not None:
        cy_model.eval()

    # Get data loader
    loader, transform_mean, transform_std, img_size = get_localization_loader(
        model, args.data_dir, args.split_dir, args
    )

    # Build part-name -> CBM attribute indices mapping
    part_to_attrs = map_attribute_ids_from_cub_to_cbm(MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS)
    part_names = list(loader.dataset.part_dict.values())  # 15 parts, ordered by part_id

    # For each part, which CBM attribute indices does it map to?
    part_attr_indices = []  # list of K tensors
    for pname in part_names:
        if pname in part_to_attrs:
            part_attr_indices.append(torch.tensor(part_to_attrs[pname], dtype=torch.long, device=device))
        else:
            part_attr_indices.append(torch.tensor([], dtype=torch.long, device=device))

    n_attrs = args.n_attributes
    sigma = getattr(args, "occlusion_sigma", 1.0)

    # Accumulators per part: baseline correct, occluded correct, count
    n_parts = len(part_names)
    baseline_correct = torch.zeros(n_parts, device=device)
    occluded_correct = torch.zeros(n_parts, device=device)
    part_counts = torch.zeros(n_parts, device=device)

    # Overall attribute accuracy (baseline vs occluded)
    overall_baseline_correct = 0.0
    overall_occluded_correct = 0.0
    overall_count = 0.0

    # Class accuracy (baseline vs occluded with all parts masked)
    class_acc_baseline = AverageMeter()
    class_acc_occluded = AverageMeter()

    with torch.no_grad():
        for data_idx, data in enumerate(tqdm(loader, desc="Concept Occlusion Eval")):
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            if args.dataset == "waterbirds":
                inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts, _ = data
            else:
                inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = data

            attr_labels = torch.stack(attr_labels, dim=1).float().to(device)
            B = inputs.shape[0]

            # --- Baseline pass (no occlusion) ---
            pred_base, scores_base = get_concept_scores(model, inputs, attr_labels, args)

            # Baseline class accuracy (skip for XC-only models that have no classifier)
            if not is_xc_only:
                if is_independent:
                    if getattr(args, "use_sigmoid_logits", True):
                        concept_scores_base = torch.sigmoid(scores_base)
                    else:
                        concept_scores_base = scores_base
                    class_pred_base = cy_model.classifier(concept_scores_base)
                else:
                    class_pred_base = pred_base
                class_acc_base_val = accuracy(class_pred_base, labels, topk=(1,))
                class_acc_baseline.update(class_acc_base_val[0], B)

            # Baseline binary predictions (threshold at 0.5 on sigmoid)
            base_preds = (torch.sigmoid(scores_base) >= 0.5).float()  # [B, A]

            # --- Build Gaussian occlusion masks ---
            gauss_masks, valid_parts = build_gaussian_occlusion_mask(
                part_gts, img_size, sigma=sigma
            )  # [B, K, H, W], [B, K]

            # --- Per-part occlusion evaluation ---
            for k in range(n_parts):
                attr_idx = part_attr_indices[k]
                if len(attr_idx) == 0:
                    continue

                # Which samples have this part visible?
                vis_mask = valid_parts[:, k]  # [B]
                if vis_mask.sum() == 0:
                    continue

                # Create occluded input: multiply image by (1 - gaussian_mask)
                occ_mask = 1.0 - gauss_masks[:, k].unsqueeze(1)  # [B, 1, H, W]
                occluded_inputs = inputs * occ_mask

                # Forward pass on occluded inputs
                _, scores_occ = get_concept_scores(model, occluded_inputs, attr_labels, args)

                occ_preds = (torch.sigmoid(scores_occ) >= 0.5).float()  # [B, A]

                # Compare for the attributes mapped to this part
                for ai in attr_idx:
                    gt = attr_labels[vis_mask, ai]  # [n_vis]
                    bp = base_preds[vis_mask, ai]
                    op = occ_preds[vis_mask, ai]

                    baseline_correct[k] += (bp == gt).float().sum()
                    occluded_correct[k] += (op == gt).float().sum()
                    part_counts[k] += gt.shape[0]

                    overall_baseline_correct += (bp == gt).float().sum().item()
                    overall_occluded_correct += (op == gt).float().sum().item()
                    overall_count += gt.shape[0]

            # --- Class accuracy with ALL visible parts occluded simultaneously ---
            if not is_xc_only:
                combined_mask = torch.zeros(B, 1, img_size, img_size, device=device)
                for k in range(n_parts):
                    vis = valid_parts[:, k].float().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B,1,1,1]
                    combined_mask += gauss_masks[:, k].unsqueeze(1) * vis
                combined_mask = combined_mask.clamp(0, 1)
                all_occluded_inputs = inputs * (1.0 - combined_mask)

                pred_occ, scores_occ_all = get_concept_scores(model, all_occluded_inputs, attr_labels, args)

                if is_independent:
                    if getattr(args, "use_sigmoid_logits", True):
                        cs = torch.sigmoid(scores_occ_all)
                    else:
                        cs = scores_occ_all
                    class_pred_occ = cy_model.classifier(cs)
                else:
                    class_pred_occ = pred_occ
                class_acc_occ_val = accuracy(class_pred_occ, labels, topk=(1,))
                class_acc_occluded.update(class_acc_occ_val[0], B)

    # --- Print results ---
    print("\n" + "=" * 70)
    print("CONCEPT OCCLUSION EVALUATION RESULTS")
    print(f"Gaussian sigma (feature-map space): {sigma}")
    print(f"Gaussian sigma (pixel space): {sigma * img_size / 8.0:.1f}")
    print("=" * 70)

    print(f"\n{'Part':<20} {'Baseline Acc':>14} {'Occluded Acc':>14} {'Delta':>10} {'Count':>8}")
    print("-" * 70)

    for k in range(n_parts):
        if part_counts[k] == 0:
            continue
        base_acc = (baseline_correct[k] / part_counts[k]).item() * 100
        occ_acc = (occluded_correct[k] / part_counts[k]).item() * 100
        delta = occ_acc - base_acc
        print(f"{part_names[k]:<20} {base_acc:>13.2f}% {occ_acc:>13.2f}% {delta:>+9.2f}% {int(part_counts[k].item()):>8}")

    print("-" * 70)
    if overall_count > 0:
        overall_base = overall_baseline_correct / overall_count * 100
        overall_occ = overall_occluded_correct / overall_count * 100
        overall_delta = overall_occ - overall_base
        print(f"{'OVERALL':<20} {overall_base:>13.2f}% {overall_occ:>13.2f}% {overall_delta:>+9.2f}% {int(overall_count):>8}")

    if not is_xc_only:
        print(f"\n--- Classification Accuracy ---")
        print(f"Baseline (no occlusion):   {class_acc_baseline.avg.item():.4f}")
        print(f"All-parts occluded:        {class_acc_occluded.avg.item():.4f}")
        print(f"Delta:                     {class_acc_occluded.avg.item() - class_acc_baseline.avg.item():+.4f}")
    print("=" * 70)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = gather_args()

    out_folder_path = os.path.join(args.log_dir, f"concept_occlusion_{args.dataset}")
    os.makedirs(out_folder_path, exist_ok=True)

    path_to_output_txt = os.path.join(out_folder_path, "occlusion_eval.txt")
    print(f"Writing outputs into {path_to_output_txt}.")
    sys.stdout = open(path_to_output_txt, 'a')

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    eval_concept_occlusion(args)
