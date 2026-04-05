import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

models = ["CBM", "ProtoCBM", "ProtoCBM+Loco"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

# Data per mode: (vals, errs)
data = {
    "Sequential": {
        "Class Acc (%)":  ([75.7, 76.3, 75.2], [0.3, 0.3, 0.3]),
        "Attr Acc (%)":   ([96.6, 96.6, 96.3], [0.1, 0.0, 0.0]),
        "Loc Dist (px)":  ([98.7, 71.4, 36.8], [3.3, 0.2, 1.4]),
        "Loc mIoU (%)":   ([5.5, 5.8, 7.05],   [0.1, 0.1, 0.0]),
    },
    "Independent": {
        "Class Acc (%)":  ([76.0, 74.4, 72.6], [0.6, 0.8, 1.0]),
        "Attr Acc (%)":   ([96.6, 96.6, 96.3], [0.1, 0.0, 0.0]),
        "Loc Dist (px)":  ([98.7, 71.4, 36.8], [3.3, 0.2, 1.4]),
        "Loc mIoU (%)":   ([5.5, 5.8, 7.05],   [0.1, 0.1, 0.0]),
    },
    "Joint": {
        "Class Acc (%)":  ([80.1, 74.3, 75.4], [0.3, 0.6, 0.6]),
        "Attr Acc (%)":   ([96.9, 95.9, 96.1], [0.0, 0.0, 0.2]),
        "Loc Dist (px)":  ([117.6, 72.4, 39.9], [24.9, 0.0, 0.7]),
        "Loc mIoU (%)":   ([5.4, 6.2, 7.07],    [0.1, 0.1, 0.0]),
    },
}

modes = list(data.keys())
metrics = list(data["Sequential"].keys())
hatches = ["///", "", ".."]  # Sequential=hatched, Independent=full, Joint=dotted

x = np.arange(len(models))
n_modes = len(modes)
total_width = 0.75
bar_w = total_width / n_modes

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for col, metric in enumerate(metrics):
    ax = axes[col]
    all_vals = []
    all_tops = []  # track bar top + error for ylim
    for i, mode in enumerate(modes):
        vals, errs = data[mode][metric]
        all_vals.extend(vals)
        all_tops.extend([v + e for v, e in zip(vals, errs)])
        offset = (i - (n_modes - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9, yerr=errs, capsize=3,
                      color=colors, edgecolor="black", linewidth=0.5,
                      hatch=hatches[i])
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.5,
                    f"{v}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    # Give enough headroom for annotations + error bars
    y_min = min(all_vals) * 0.88
    y_max = max(all_tops) * 1.12
    ax.set_ylim(y_min, y_max)

# Legend for modes (hatching)
legend_elements = [
    Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="Sequential"),
    Patch(facecolor="lightgray", edgecolor="black", label="Independent"),
    Patch(facecolor="lightgray", edgecolor="black", hatch="..", label="Joint"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, 1.01))

fig.suptitle("Main Results — Comparison to CBMs (CUB)", fontsize=14,
             fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig("main_results.png", dpi=200, bbox_inches="tight")
plt.savefig("main_results.pdf", bbox_inches="tight")
print("Saved main_results.png and main_results.pdf")
