import matplotlib.pyplot as plt
import numpy as np

labels = [
    r"$\mathcal{L}_{cls}$",
    r"$\mathcal{L}_{cls}$ + $\mathcal{L}_{con}$",
    r"$\mathcal{L}_{cls}$ + $\mathcal{L}_{con}$ + $\mathcal{L}_{cmp}$",
    r"$\mathcal{L}_{cls}$ + $\mathcal{L}_{con}$ + $\mathcal{L}_{cmp}$ + $\mathcal{L}_{dec}$",
]
short_labels = [
    r"$\mathcal{L}_{cls}$",
    r"+ $\mathcal{L}_{con}$",
    r"+ $\mathcal{L}_{cmp}$",
    r"+ $\mathcal{L}_{dec}$",
]

metrics = {
    "Class Acc (%)":  ([73.2, 73.2, 73.7, 74.2], [1.2, 0.8, 0.7, 0.6]),
    "Attr Acc (%)":   ([50.9, 95.6, 95.8, 95.9], [1.1, 0.1, 0.2, 0.0]),
    "Loc Dist (px)":  ([92.2, 72.4, 72.4, 72.4], [1.7, 0.5, 0.4, 0.4]),
    "Loc mIoU (%)":   ([6.1, 6.1, 6.1, 6.2],     [0.2, 0.1, 0.1, 0.1]),
}

x = np.arange(len(labels))
width = 0.55
# Progressive color: each added loss darkens the bar
colors = ["#a6cee3", "#6baed6", "#3182bd", "#08519c"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (metric, (vals, errs)) in zip(axes, metrics.items()):
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylim(min(vals) * 0.92, max(vals) * 1.06)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 1.1,
                f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.suptitle("Loss Ablation (ProtoCBM Joint, CUB)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("loss_ablation_results.png", dpi=200, bbox_inches="tight")
plt.savefig("loss_ablation_results.pdf", bbox_inches="tight")
print("Saved loss_ablation_results.png and loss_ablation_results.pdf")
