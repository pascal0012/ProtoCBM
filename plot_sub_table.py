import matplotlib.pyplot as plt
import numpy as np

models = ["CBM", "ProtoCBM", "ProtoCBM+Loco"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

# Sequential / Independent
seq_new = ([35.1, 37.8, 41.9], [3.1, 2.7, 1.4])
seq_old = ([52.0, 50.4, 52.3], [2.3, 1.6, 3.0])

# Joint
jnt_new = ([34.3, 38.9, 43.7], [1.1, 0.7, 2.7])
jnt_old = ([56.6, 48.7, 52.1], [2.3, 1.8, 1.0])

x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, (metric, seq_data, jnt_data) in zip(axes, [
    ("New Detected (%)", seq_new, jnt_new),
    ("Old Detected (%)", seq_old, jnt_old),
]):
    vals_s, errs_s = seq_data
    vals_j, errs_j = jnt_data

    bars_s = ax.bar(x - width/2, vals_s, width, yerr=errs_s, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6,
                    hatch="///", label="Seq. / Indep.")
    bars_j = ax.bar(x + width/2, vals_j, width, yerr=errs_j, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6, label="Joint")

    for bars, vals in [(bars_s, vals_s), (bars_j, vals_j)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{v}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylim(min(vals_s + vals_j) * 0.82, max(vals_s + vals_j) * 1.12)

# Custom legend: hatched = Seq/Indep, full = Joint
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="gray", edgecolor="black", hatch="///", label="Seq. / Indep."),
    Patch(facecolor="gray", edgecolor="black", label="Joint"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Robustness to Concept Substitutions (SUB)", fontsize=13,
             fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig("sub_results.png", dpi=200, bbox_inches="tight")
plt.savefig("sub_results.pdf", bbox_inches="tight")
print("Saved sub_results.png and sub_results.pdf")
