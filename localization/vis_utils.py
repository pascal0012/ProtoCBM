from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import os 

@dataclass
class PredictionStyle:
    """Configuration for visualizing a single prediction type."""
    color: str
    marker: str
    size: int
    label: str
    edgecolor: Optional[str] = None
    linewidth: int = 2
    line_style: str = "--"
    line_alpha: float = 0.7
    zorder: int = 3


def extract_image_name(img_path: str) -> str:
    """Extract and sanitize image name from path."""
    img_name = "_".join(img_path.split(os.sep)[-2:]) if os.sep in img_path else img_path
    img_name = img_name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    return img_name.replace("/", "_").replace("\\", "_")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array on CPU."""
    return tensor.cpu().numpy()


def denormalize_image(img: torch.Tensor, t_mean: Tuple, t_std: Tuple) -> np.ndarray:
    """Denormalize and convert image tensor to numpy."""
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    return np.clip(img_np, 0, 1)


def plot_prediction(ax, x: float, y: float, style: PredictionStyle, show_label: bool = False):
    """Plot a single prediction point with given style."""
    label = style.label if show_label else None
    ax.scatter(
        x, y,
        c=style.color,
        s=style.size,
        marker=style.marker,
        edgecolors=style.edgecolor,
        linewidths=style.linewidth,
        label=label,
        zorder=style.zorder,
    )


def plot_line_to_gt(ax, gt_x: float, gt_y: float, pred_x: float, pred_y: float, style: PredictionStyle):
    """Plot dashed line from GT to prediction."""
    ax.plot(
        [gt_x, pred_x], [gt_y, pred_y],
        color=style.color,
        linestyle=style.line_style,
        linewidth=1.5,
        alpha=style.line_alpha,
    )


def build_title(part_name: str, distances: Dict[str, float], has_gt: bool) -> str:
    """Build subplot title with distances."""
    dist_strs = [f"{name}: {dist:.1f}px" for name, dist in distances.items()]
    if has_gt:
        return f"{part_name}\n" + " | ".join(dist_strs)
    else:
        return f"{part_name}\nNo GT | " + " | ".join(dist_strs)

# Default prediction styles
DEFAULT_STYLES = {
    "argmax": PredictionStyle(color="red", marker="x", size=120, label="Argmax", linewidth=3),
    "agg": PredictionStyle(color="blue", marker="^", size=100, label="Aggregated", edgecolor="white", linewidth=1),
    "center": PredictionStyle(color="orange", marker="s", size=100, label="Weighted", edgecolor="white", linewidth=1),
}

GT_STYLE = PredictionStyle(color="lime", marker="o", size=120, label="GT", edgecolor="black", linewidth=2, zorder=4)
