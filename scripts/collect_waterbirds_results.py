"""Collect waterbirds evaluation results across seeds and compute mean ± std."""
import os
import re
import numpy as np
from collections import defaultdict

# Define where each mode's results live
RESULT_DIRS = {
    "Joint": [
        "outputs/Inspection/CBM-Baseline/waterbirds/seed1/waterbirds_visualization_cam/eval.txt",
        "outputs/Inspection/CBM-Baseline/waterbirds/seed2/waterbirds_visualization_cam/eval.txt",
    ],
    "Sequential": [
        "outputs/Inspection/CBM-Sequential/waterbirds/seed1/waterbirds_visualization_cam/eval.txt",
        "outputs/Inspection/CBM-Sequential/waterbirds/seed2/waterbirds_visualization_cam/eval.txt",
        "outputs/Inspection/CBM-Sequential/waterbirds/seed3/waterbirds_visualization_cam/eval.txt",
    ],
    "Independent": [
        "outputs/Inspection/CBM-Independent/waterbirds/seed1/waterbirds_visualization_cam/eval.txt",
        "outputs/Inspection/CBM-Independent/waterbirds/seed2/waterbirds_visualization_cam/eval.txt",
        "outputs/Inspection/CBM-Independent/waterbirds/seed3/waterbirds_visualization_cam/eval.txt",
    ],
}

# Metrics to extract (regex pattern, display name)
METRICS = [
    (r"Mean Classification Accuracy:\s+([\d.]+)", "Classification Acc"),
    (r"Mean Attribute Accuracy:\s+([\d.]+)", "Attribute Acc"),
    (r"Mean Attribute Cross-Entropy:\s+([\d.]+)", "Attribute CE"),
    (r"Macro Precision:\s+([\d.]+)", "Macro Precision"),
    (r"Macro Recall:\s+([\d.]+)", "Macro Recall"),
    (r"Macro F1:\s+([\d.]+)", "Macro F1"),
    (r"Mean Waterbirds Classification Accuracy:\s+([\d.]+)", "Waterbird Class Acc"),
    (r"Mean Waterbirds Attribute Accuracy:\s+([\d.]+)", "Waterbird Attr Acc"),
    (r"Mean Landbirds Classification Accuracy:\s+([\d.]+)", "Landbird Class Acc"),
    (r"Mean Landbirds Attribute Accuracy:\s+([\d.]+)", "Landbird Attr Acc"),
]


def parse_eval_file(path):
    """Extract metrics from an eval.txt file."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        content = f.read()
    results = {}
    for pattern, name in METRICS:
        match = re.search(pattern, content)
        if match:
            results[name] = float(match.group(1))
    return results


def main():
    for mode, paths in RESULT_DIRS.items():
        print(f"\n{'='*60}")
        print(f"  {mode} CBM — Waterbirds ({len(paths)} seeds)")
        print(f"{'='*60}")

        all_results = []
        for i, p in enumerate(paths):
            r = parse_eval_file(p)
            if r is not None:
                all_results.append(r)
                print(f"\n  Seed {i+1}: {p}")
                for name, val in r.items():
                    print(f"    {name}: {val:.4f}")

        if len(all_results) < 2:
            print(f"\n  Not enough seeds to compute std (found {len(all_results)})")
            continue

        print(f"\n  --- Mean ± Std ({len(all_results)} seeds) ---")
        metric_names = [name for _, name in METRICS]
        for name in metric_names:
            vals = [r[name] for r in all_results if name in r]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                print(f"    {name:30s}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
