"""
Compactness Evaluation of Prototype Attention Maps

For each image and concept, computes spatial compactness of the attention map
produced by convolving prototype vectors over backbone feature embeddings.

Metrics per attribute (computed only for images where the concept is predicted
active, i.e. sigmoid(score) >= 0.5):

  - PTR   (Peak-to-Total Ratio):      max / sum of attention values (higher = more compact)
  - ESC   (Effective Spatial Coverage): fraction of cells above 50% of peak (lower = more compact)
  - Spread (Weighted Spatial Spread):  mean distance from activation centroid,
                                        weighted by attention (lower = more compact)

Results are reported per-attribute and averaged by CUB body-part group.
Output is printed to stdout and saved as a CSV.

Usage:
    python scripts/analysis/eval_compactness.py --config configs/protocbm/eval/eval_independent.yaml
"""

import os
import sys
import math
import csv

import torch
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from saliency.saliency import get_saliency_map_and_scores_and_prediction
from utils_protocbm.eval_utils import get_localization_loader, create_model_for_eval
from utils_protocbm.train_utils import gather_args
from utils_protocbm.mappings import MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS
from utils_protocbm.index_translation import map_attribute_ids_from_cub_to_cbm, get_attribute_names


# ── Metric computation ────────────────────────────────────────────────────────

def compute_compactness_metrics(attention_maps: torch.Tensor):
    """
    Compute per-sample, per-attribute compactness metrics.

    Args:
        attention_maps: [B, A, H, W] float tensor, values in [0, 1].

    Returns:
        ptr:    [B, A] Peak-to-Total Ratio         (higher = more compact)
        esc:    [B, A] Effective Spatial Coverage  (lower  = more compact)
        spread: [B, A] Weighted Spatial Spread     (lower  = more compact)
    """
    B, A, H, W = attention_maps.shape
    flat = attention_maps.view(B, A, -1)          # [B, A, H*W]

    # ── PTR ──────────────────────────────────────────────────────────────────
    peak_val = flat.max(dim=2).values             # [B, A]
    total = flat.sum(dim=2)                       # [B, A]
    ptr = peak_val / (total + 1e-8)

    # ── ESC ──────────────────────────────────────────────────────────────────
    threshold = 0.5 * peak_val.unsqueeze(2)       # [B, A, 1]
    esc = (flat >= threshold).float().mean(dim=2)  # [B, A]

    # ── Weighted spatial spread ───────────────────────────────────────────────
    device = flat.device
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    grid_y = grid_y.reshape(1, 1, -1)            # [1, 1, H*W]
    grid_x = grid_x.reshape(1, 1, -1)

    weights = flat / (total.unsqueeze(2) + 1e-8)  # normalised attention [B, A, H*W]
    cy = (weights * grid_y).sum(dim=2)            # centroid y  [B, A]
    cx = (weights * grid_x).sum(dim=2)            # centroid x  [B, A]

    dy = grid_y - cy.unsqueeze(2)                 # [B, A, H*W]
    dx = grid_x - cx.unsqueeze(2)
    dist = torch.sqrt(dy ** 2 + dx ** 2)          # Euclidean from centroid
    spread = (weights * dist).sum(dim=2)          # weighted mean distance [B, A]

    return ptr, esc, spread


# ── Part-group mapping ────────────────────────────────────────────────────────

def build_group_to_cbm_attr_ids():
    """
    Returns a dict mapping each part-seg group name to a list of CBM-relative
    attribute indices (0-indexed within the 112 CBM attributes).
    Groups without any CBM attributes are omitted.
    """
    group_to_cbm = {}
    for group, cub_ids in MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS.items():
        cbm_ids = map_attribute_ids_from_cub_to_cbm(cub_ids)
        if cbm_ids:
            group_to_cbm[group] = cbm_ids
    return group_to_cbm


# ── Main evaluation ───────────────────────────────────────────────────────────

def eval_compactness(args):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load model
    model, device, cy_model = create_model_for_eval(args)
    model.eval()
    if cy_model is not None:
        cy_model.eval()

    # Data loader
    loader, _, _, _ = get_localization_loader(model, args.data_dir, args.split_dir, args)
    n_attrs = args.n_attributes  # 112 (all CBM attributes)
    attribute_names = get_attribute_names(args.data_dir, only_cbm_attributes=True)

    # Accumulators: sum and count per attribute (only active concepts)
    ptr_sum    = torch.zeros(n_attrs, device=device)
    esc_sum    = torch.zeros(n_attrs, device=device)
    spread_sum = torch.zeros(n_attrs, device=device)
    count      = torch.zeros(n_attrs, device=device)  # number of active-concept samples

    # CBM uses GradCAM which requires gradients; ProtoCBM attention maps do not.
    use_no_grad = (args.saliency_method != "cam")
    ctx = torch.no_grad() if use_no_grad else torch.enable_grad()

    with ctx:
        for data in tqdm(loader, desc="Evaluating compactness"):
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            if args.dataset == "waterbirds":
                inputs, labels, attr_labels, *_ = data
            else:
                inputs, labels, attr_labels, *_ = data

            attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

            # Forward pass → attention maps [B, A, H, W] normalised to [0, 1]
            _, scores, saliency_maps = get_saliency_map_and_scores_and_prediction(
                model, inputs, args, attr_labels=attr_labels
            )
            saliency_maps = saliency_maps.detach().to(device)  # [B, A, H, W]
            scores = scores.detach()

            # Active-concept mask: only accumulate where concept is predicted on
            active = (torch.sigmoid(scores) >= 0.5)    # [B, A] bool

            ptr, esc, spread = compute_compactness_metrics(saliency_maps)

            # Mask out inactive and accumulate
            ptr_sum    += (ptr    * active.float()).sum(dim=0)
            esc_sum    += (esc    * active.float()).sum(dim=0)
            spread_sum += (spread * active.float()).sum(dim=0)
            count      += active.float().sum(dim=0)

    # Mean over active samples (avoid div/0 for attributes with no active predictions)
    safe_count = count.clamp(min=1)
    ptr_mean    = (ptr_sum    / safe_count).cpu()
    esc_mean    = (esc_sum    / safe_count).cpu()
    spread_mean = (spread_sum / safe_count).cpu()
    count_cpu   = count.cpu()

    # ── Per-attribute table ───────────────────────────────────────────────────
    col_w = max(len(n) for n in attribute_names) + 2
    print(f"\n{'Attribute':<{col_w}} {'PTR':>8} {'ESC':>8} {'Spread':>10} {'N_active':>10}")
    print("-" * (col_w + 40))
    for i, name in enumerate(attribute_names):
        print(f"{name:<{col_w}} {ptr_mean[i]:>8.4f} {esc_mean[i]:>8.4f} "
              f"{spread_mean[i]:>10.4f} {int(count_cpu[i]):>10}")

    # ── Per-part-group summary ────────────────────────────────────────────────
    group_to_cbm = build_group_to_cbm_attr_ids()
    print(f"\n{'Part Group':<16} {'PTR':>8} {'ESC':>8} {'Spread':>10} {'N_attrs':>10}")
    print("-" * 54)
    group_rows = []
    for group, idxs in sorted(group_to_cbm.items()):
        valid = [i for i in idxs if i < n_attrs]
        if not valid:
            continue
        g_ptr    = ptr_mean[valid].mean().item()
        g_esc    = esc_mean[valid].mean().item()
        g_spread = spread_mean[valid].mean().item()
        print(f"{group:<16} {g_ptr:>8.4f} {g_esc:>8.4f} {g_spread:>10.4f} {len(valid):>10}")
        group_rows.append((group, g_ptr, g_esc, g_spread, len(valid)))

    # ── Overall means ─────────────────────────────────────────────────────────
    # Weight by number of active samples so rarely-active attributes don't dominate
    total_active = count_cpu.sum().item()
    if total_active > 0:
        overall_ptr    = ((ptr_mean    * count_cpu).sum() / total_active).item()
        overall_esc    = ((esc_mean    * count_cpu).sum() / total_active).item()
        overall_spread = ((spread_mean * count_cpu).sum() / total_active).item()
    else:
        overall_ptr = overall_esc = overall_spread = float('nan')

    print(f"\n{'OVERALL':<16} {overall_ptr:>8.4f} {overall_esc:>8.4f} {overall_spread:>10.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(getattr(args, "log_dir", "logs"), getattr(args, "model_name", "model"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "compactness.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute", "ptr", "esc", "spread", "n_active"])
        for i, name in enumerate(attribute_names):
            writer.writerow([name,
                             f"{ptr_mean[i].item():.6f}",
                             f"{esc_mean[i].item():.6f}",
                             f"{spread_mean[i].item():.6f}",
                             int(count_cpu[i].item())])
        # Blank separator then group summary
        writer.writerow([])
        writer.writerow(["part_group", "ptr", "esc", "spread", "n_attrs"])
        for row in group_rows:
            writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}",
                             f"{row[3]:.6f}", row[4]])
        writer.writerow(["OVERALL", f"{overall_ptr:.6f}", f"{overall_esc:.6f}",
                         f"{overall_spread:.6f}", ""])

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    args = gather_args()
    eval_compactness(args)
