"""Quick check of raw logit values from a checkpoint — full test set."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from cub.config import BASE_DIR
from cub.dataset import CUBDataset
from eval_sub import get_cbm_attribute_names
from utils_protocbm.train_utils import gather_args, create_model, prepare_model
from utils_protocbm.eval_utils import get_eval_transform_for_model


def main():
    args = gather_args()

    model = create_model(args)
    model, device = prepare_model(model, args, load_weights=True)
    model.eval()

    transform = get_eval_transform_for_model(model, args)[0]
    pkl_path = os.path.join(BASE_DIR, args.split_dir)
    image_dir = os.path.join(BASE_DIR, getattr(args, "image_dir", "data/CUB_200_2011"))
    dataset = CUBDataset([pkl_path], image_dir, transform, getattr(args, "dataset", "cub"))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    cbm_attr_names = get_cbm_attribute_names()

    all_sim = []
    all_gt = []

    with torch.no_grad():
        for images, class_labels, attr_labels in tqdm(loader, desc="Checking logits"):
            images = images.to(device)
            attr_labels_tensor = torch.stack(attr_labels, dim=1).float().to(device)

            outputs = model(images, attr_labels_tensor)

            if isinstance(outputs, tuple) and len(outputs) == 3:
                _, sim_scores, _ = outputs
            elif isinstance(outputs, list):
                sim_scores = torch.cat(outputs[1:], dim=1)
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")

            all_sim.append(sim_scores.cpu().numpy())
            all_gt.append(attr_labels_tensor.cpu().numpy())

    sim = np.concatenate(all_sim, axis=0)   # [N, 112]
    gt = np.concatenate(all_gt, axis=0)     # [N, 112]
    probs = 1 / (1 + np.exp(-sim))          # sigmoid

    print(f"\n{'='*70}")
    print(f"SIMILARITY SCORE DIAGNOSTICS  (N={sim.shape[0]})")
    print(f"{'='*70}")

    print("\nRaw similarity scores:")
    print(f"  Min:    {sim.min():.4f}")
    print(f"  Max:    {sim.max():.4f}")
    print(f"  Mean:   {sim.mean():.4f}")
    print(f"  Median: {np.median(sim):.4f}")
    print(f"  Std:    {sim.std():.4f}")

    preds = (probs >= 0.5).astype(float)
    print("\nPrediction stats (threshold=0.5):")
    print(f"  Fraction predicted active: {preds.mean():.4f}")
    print(f"  GT fraction active:        {gt.mean():.4f}")

    pred_frac = preds.mean(axis=0)  # [112]
    gt_frac = gt.mean(axis=0)       # [112]

    print("\nPer-attribute summary:")
    print(f"  #attrs predicted >90% of time: {(pred_frac > 0.9).sum()}")
    print(f"  #attrs predicted >50% of time: {(pred_frac > 0.5).sum()}")
    print(f"  #attrs predicted <10% of time: {(pred_frac < 0.1).sum()}")
    print(f"  #attrs GT >50% of time:        {(gt_frac > 0.5).sum()}")

    # Histogram of sim scores
    print("\nSimilarity score histogram:")
    bins = [-100, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 100]
    for i in range(len(bins) - 1):
        count = ((sim >= bins[i]) & (sim < bins[i+1])).sum()
        pct = 100 * count / sim.size
        print(f"  [{bins[i]:>7.1f}, {bins[i+1]:>7.1f}): {count:>8} ({pct:>5.1f}%)")
    count = (sim >= bins[-1]).sum()
    pct = 100 * count / sim.size
    print(f"  [{bins[-1]:>7.1f},     inf): {count:>8} ({pct:>5.1f}%)")

    # Top 10 most-predicted and least-predicted attributes
    order = np.argsort(pred_frac)[::-1]
    print("\nTop 15 MOST predicted attributes:")
    for i in range(15):
        idx = order[i]
        print(f"  {cbm_attr_names[idx]:<45} pred={pred_frac[idx]:.3f}  gt={gt_frac[idx]:.3f}  mean_sim={sim[:,idx].mean():.3f}")

    print("\nTop 15 LEAST predicted attributes:")
    for i in range(15):
        idx = order[-(i+1)]
        print(f"  {cbm_attr_names[idx]:<45} pred={pred_frac[idx]:.3f}  gt={gt_frac[idx]:.3f}  mean_sim={sim[:,idx].mean():.3f}")

    # ---- Plotting ----
    gt_flat = gt.flatten().astype(bool)
    sim_flat = sim.flatten()
    probs_flat = probs.flatten()

    sim_pos = sim_flat[gt_flat]
    sim_neg = sim_flat[~gt_flat]
    probs_pos = probs_flat[gt_flat]
    probs_neg = probs_flat[~gt_flat]

    # Compute accuracy stats (threshold=0.5 on raw scores)
    tn_05 = (sim_neg < 0.5).sum()
    fp_05 = sim_neg.size - tn_05
    tp_05 = (sim_pos >= 0.5).sum()
    fn_05 = sim_pos.size - tp_05
    tn_pct_05 = 100 * tn_05 / sim_neg.size
    tp_pct_05 = 100 * tp_05 / sim_pos.size

    # Compute accuracy stats (threshold=0 on raw scores, i.e. sigmoid > 0.5)
    tn_0 = (sim_neg < 0).sum()
    fp_0 = sim_neg.size - tn_0
    tp_0 = (sim_pos >= 0).sum()
    fn_0 = sim_pos.size - tp_0
    tn_pct_0 = 100 * tn_0 / sim_neg.size
    tp_pct_0 = 100 * tp_0 / sim_pos.size

    print(f"\nClassification stats (threshold=0.5 on raw scores):")
    print(f"  GT=0: TN={tn_05} ({tn_pct_05:.1f}%)  FP={fp_05} ({100-tn_pct_05:.1f}%)")
    print(f"  GT=1: TP={tp_05} ({tp_pct_05:.1f}%)  FN={fn_05} ({100-tp_pct_05:.1f}%)")
    print(f"\nClassification stats (threshold=0 on raw scores, sigmoid>0.5):")
    print(f"  GT=0: TN={tn_0} ({tn_pct_0:.1f}%)  FP={fp_0} ({100-tn_pct_0:.1f}%)")
    print(f"  GT=1: TP={tp_0} ({tp_pct_0:.1f}%)  FN={fn_0} ({100-tp_pct_0:.1f}%)")

    # --- Figure 1: Raw concept scores (2 rows) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.hist(sim_flat, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax1.set_title("Raw Concept Scores (all)")
    ax1.set_xlabel("Similarity score")
    ax1.set_ylabel("Count")
    ax1.axvline(0, color="orange", ls="--", lw=1, label="threshold (0)")
    ax1.axvline(0.5, color="red", ls="--", lw=1, label="threshold (0.5)")
    ax1.legend()

    ax2.hist(sim_neg, bins=100, color="salmon", edgecolor="none", alpha=0.6, label=f"GT=0 (n={sim_neg.size})")
    ax2.hist(sim_pos, bins=100, color="seagreen", edgecolor="none", alpha=0.6, label=f"GT=1 (n={sim_pos.size})")
    ax2.set_title("Raw Concept Scores by GT label")
    ax2.set_xlabel("Similarity score")
    ax2.set_ylabel("Count")
    ax2.axvline(0, color="orange", ls="--", lw=1)
    ax2.axvline(0.5, color="red", ls="--", lw=1)
    ax2.annotate(f"GT=0 correct (TN): {tn_05} ({tn_pct_05:.1f}%)\nGT=0 wrong (FP): {fp_05} ({100-tn_pct_05:.1f}%)",
                 xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8))
    ax2.annotate(f"GT=1 correct (TP): {tp_05} ({tp_pct_05:.1f}%)\nGT=1 wrong (FN): {fn_05} ({100-tp_pct_05:.1f}%)",
                 xy=(0.98, 0.98), xycoords="axes fraction", va="top", ha="right", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.8))
    ax2.legend(loc="center right")

    fig1.suptitle("Raw Concept Score Distributions", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig("logit_distributions_raw.png", dpi=150)
    print("\nPlot saved to logit_distributions_raw.png")
    plt.close(fig1)

    # --- Figure 2: Sigmoid probabilities (2 rows) ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

    ax3.hist(probs_flat, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax3.set_title("Sigmoid Probabilities (all)")
    ax3.set_xlabel("σ(score)")
    ax3.set_ylabel("Count")
    ax3.axvline(0.5, color="red", ls="--", lw=1, label="threshold (0.5)")
    ax3.legend()

    ax4.hist(probs_neg, bins=100, color="salmon", edgecolor="none", alpha=0.6, label=f"GT=0 (n={probs_neg.size})")
    ax4.hist(probs_pos, bins=100, color="seagreen", edgecolor="none", alpha=0.6, label=f"GT=1 (n={probs_pos.size})")
    ax4.set_title("Sigmoid Probabilities by GT label")
    ax4.set_xlabel("σ(score)")
    ax4.set_ylabel("Count")
    ax4.axvline(0.5, color="red", ls="--", lw=1)
    ax4.legend()

    fig2.suptitle("Sigmoid Probability Distributions", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig("logit_distributions_sigmoid.png", dpi=150)
    print("Plot saved to logit_distributions_sigmoid.png")
    plt.close(fig2)

    # --- Figure 3: Combined raw + sigmoid by GT label, shared x-axis (2 rows) ---
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x_min = min(sim_flat.min(), 0) - 0.1
    x_max = max(sim_flat.max(), 1) + 0.1

    ax5.hist(sim_neg, bins=100, color="salmon", edgecolor="none", alpha=0.6, label=f"GT=0 (n={sim_neg.size})")
    ax5.hist(sim_pos, bins=100, color="seagreen", edgecolor="none", alpha=0.6, label=f"GT=1 (n={sim_pos.size})")
    ax5.set_title("Raw Concept Scores by GT label")
    ax5.set_ylabel("Count")
    ax5.axvline(0, color="orange", ls="--", lw=1, label="threshold (0)")
    ax5.axvline(0.5, color="red", ls="--", lw=1, label="threshold (0.5)")
    ax5.set_xlim(x_min, x_max)
    ax5.legend(fontsize=7)

    ax6.hist(probs_neg, bins=100, color="salmon", edgecolor="none", alpha=0.6, label=f"GT=0 (n={probs_neg.size})")
    ax6.hist(probs_pos, bins=100, color="seagreen", edgecolor="none", alpha=0.6, label=f"GT=1 (n={probs_pos.size})")
    ax6.set_title("Sigmoid Probabilities by GT label")
    ax6.set_xlabel("Score")
    ax6.set_ylabel("Count")
    ax6.axvline(0.5, color="red", ls="--", lw=1, label="threshold (0.5)")
    ax6.set_xlim(x_min, x_max)
    ax6.legend(fontsize=7)

    fig3.suptitle("Raw Scores vs Sigmoid Probabilities (shared x-axis)", fontsize=14, fontweight="bold")
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig("logit_distributions_combined.png", dpi=150)
    print("Plot saved to logit_distributions_combined.png")
    plt.close(fig3)


if __name__ == "__main__":
    main()
