import matplotlib.pyplot as plt
import numpy as np

# Data from Table
models = ["CBM", "ProtoCBM", "LocoCBM"]

# (a) CUB -> Waterbirds
cub_wb = {
    "Class Acc (%)":   ([48.4, 62.9, 64.1], [2.0, 0.5, 1.2]),
    "Attr Acc (%)":    ([93.8, 93.9, 94.0], [0.2, 0.2, 0.1]),
    "Loc Dist (px)":   ([100.7, 70.6, 41.9], [3.1, 0.2, 2.5]),
    "Loc mIoU (%)":    ([5.4, 5.7, 7.0], [0.1, 0.0, 0.0]),
}

# (b) Waterbirds unbalanced -> balanced
wb_wb = {
    "Class Acc (%)":   ([59.9, 70.8, 71.1], [0.7, 0.3, 1.0]),
    "Attr Acc (%)":    ([95.8, 95.7, 95.4], [0.1, 0.0, 0.1]),
    "Loc Dist (px)":   ([118.6, 71.2, 39.0], [0.9, 0.4, 1.5]),
    "Loc mIoU (%)":    ([4.9, 5.8, 7.06], [0.0, 0.1, 0.0]),
}

colors = ["#4C72B0", "#DD8452", "#55A868"]
x = np.arange(len(models))
width = 0.6

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for col, (metric, (vals, errs)) in enumerate(cub_wb.items()):
    ax = axes[0, col]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=15, ha="right")
    ax.set_title(metric, fontsize=10, fontweight="bold")
    ax.set_ylim(min(vals) * 0.85, max(vals) * 1.08)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 1.1,
                f"{v}", ha="center", va="bottom", fontsize=8, fontweight="bold")

for col, (metric, (vals, errs)) in enumerate(wb_wb.items()):
    ax = axes[1, col]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=15, ha="right")
    ax.set_title(metric, fontsize=10, fontweight="bold")
    ax.set_ylim(min(vals) * 0.85, max(vals) * 1.08)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 1.1,
                f"{v}", ha="center", va="bottom", fontsize=8, fontweight="bold")

axes[0, 0].set_ylabel("(a) CUB → Waterbirds", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("(b) WB unbal. → WB bal.", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("waterbirds_results.png", dpi=200, bbox_inches="tight")
print("Saved waterbirds_results.png and waterbirds_results.pdf")
