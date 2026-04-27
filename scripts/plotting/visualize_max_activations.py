"""
Activation Maximization for Concept Nodes.

Generates synthetic images that maximally activate specific concept neurons
for ProtoCBM, Oracle-ProtoCBM, and CBM models. Uses gradient ascent on
the input image to maximize concept activation.

Usage:
    python visualize_max_activations.py --concepts 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
from argparse import Namespace
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))

from cub.config import N_ATTRIBUTES_CBM
from models.models import ModelXtoC, ModelXtoCtoY, backbone_by_name, concept_mapper_by_name
from models.model_connector import ModelConnector
from models.components import MLP
from cub.config import N_CLASSES
from utils_protocbm.train_utils import prepare_model, _clean_state_dict


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Create a minimal Namespace with defaults for model construction."""
    defaults = dict(
        backbone="inception",
        backbone_pretrained=True,
        backbone_freeze=False,
        use_aux=False,
        expand_dim=0,
        n_attributes=N_ATTRIBUTES_CBM,
        concept_activation="none",
        proto_n_vectors=1,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def load_protocbm_xc(checkpoint_path, device="cuda"):
    args = _make_args(
        model_name="protocbm",
        concept_mapper="protomod",
        mode="XC",
        checkpoint=checkpoint_path,
    )
    model = ModelXtoC(args)
    model, device = prepare_model(model, args, load_weights=True, compile=False)
    model.eval()
    return model, device


def load_cbm_xc(checkpoint_path, device="cuda"):
    args = _make_args(
        model_name="cbm",
        concept_mapper="cbm",
        mode="XC",
        checkpoint=checkpoint_path,
    )
    model = ModelXtoC(args)
    model, device = prepare_model(model, args, load_weights=True, compile=False)
    model.eval()
    return model, device


def load_oracle_protocbm(checkpoint_path, device="cuda"):
    """Oracle is trained as XCY with protomod mapper + classifier."""
    args = _make_args(
        model_name="protocbm",
        concept_mapper="protomod",
        mode="XCY",
        checkpoint=checkpoint_path,
    )
    model = ModelXtoCtoY(args)
    model, device = prepare_model(model, args, load_weights=True, compile=False)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Concept score extraction (unified for both architectures)
# ---------------------------------------------------------------------------

def get_concept_scores(model, img_tensor):
    """
    Run forward pass and return concept scores [B, 112].
    Works for both ProtoMod and CBM mappers.
    """
    mapper_name = model.concept_mapper._get_name()

    if mapper_name == "ProtoMod":
        # forward returns (output, sim_scores, maps) for PROTO
        result = model(img_tensor, attr_labels=None)
        if isinstance(result, tuple) and len(result) == 3:
            # XCY: (class_logits_or_concepts, sim_scores, maps)
            _, sim_scores, _ = result
            return sim_scores  # raw similarity scores [B, 112]
        elif isinstance(result, tuple) and len(result) == 2:
            sim_scores, _ = result
            return sim_scores
        return result

    elif mapper_name == "CBMMapper":
        # forward returns list of [B, 1] tensors for XC mode
        result = model(img_tensor, attr_labels=None)
        if isinstance(result, list):
            return torch.cat(result, dim=1)  # [B, 112]
        # XCY mode: [class_logits, attr1, ..., attrN]
        if isinstance(result, list) and len(result) > N_ATTRIBUTES_CBM:
            return torch.cat(result[1:], dim=1)
        return result

    raise ValueError(f"Unknown mapper: {mapper_name}")


# ---------------------------------------------------------------------------
# Activation maximization
# ---------------------------------------------------------------------------

def activation_maximization(
    model,
    concept_idx,
    device="cuda",
    img_size=299,
    n_steps=512,
    lr=0.05,
    l2_weight=1e-3,
    tv_weight=1e-4,
    blur_every=4,
    blur_sigma=0.5,
    jitter=8,
):
    """
    Generate an image that maximally activates concept `concept_idx`.

    Uses gradient ascent with regularization:
      - L2 penalty on pixel values (keeps image bounded)
      - Total variation penalty (spatial smoothness)
      - Gaussian blur every few steps (removes high-freq noise)
      - Random spatial jitter (avoids fixed-position artifacts)
    """
    # Start from random noise with small magnitude
    img = torch.randn(1, 3, img_size, img_size, device=device) * 0.01
    img.requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Random jitter for translation invariance
        ox, oy = torch.randint(-jitter, jitter + 1, (2,)).tolist()
        img_shifted = torch.roll(torch.roll(img, ox, -1), oy, -2)

        scores = get_concept_scores(model, img_shifted)  # [1, 112]
        activation = scores[0, concept_idx]

        # Total variation regularization
        tv_loss = (
            torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        )

        # Maximize activation, minimize regularizers
        loss = -activation + l2_weight * torch.norm(img) + tv_weight * tv_loss
        loss.backward()
        optimizer.step()

        # Periodic Gaussian blur to suppress high-frequency noise
        if blur_every > 0 and step % blur_every == 0:
            with torch.no_grad():
                img.data = _gaussian_blur(img.data, sigma=blur_sigma)

    return img.detach()


def _gaussian_blur(img, sigma=1.0, kernel_size=5):
    """Apply Gaussian blur to a [1, C, H, W] tensor."""
    channels = img.shape[1]
    x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(img, kernel, padding=pad, groups=channels)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def tensor_to_displayable(img_tensor, mean, std):
    """Convert a normalized [1, 3, H, W] tensor back to displayable [H, W, 3] numpy in [0, 1]."""
    img = img_tensor[0].cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def load_attribute_names(data_dir="data/CUB_200_2011"):
    """Load the 112 CBM attribute names from the CUB dataset."""
    from utils_protocbm.index_translation import get_attribute_names
    return get_attribute_names(data_dir, only_cbm_attributes=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Activation Maximization for Concept Nodes")
    parser.add_argument("--concepts", type=int, nargs="+", default=list(range(10)),
                        help="Concept indices to visualize (default: 0-9)")
    parser.add_argument("--n_steps", type=int, default=2048, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--l2_weight", type=float, default=1e-3, help="L2 regularization weight")
    parser.add_argument("--tv_weight", type=float, default=1e-4, help="Total variation weight")
    parser.add_argument("--output_dir", type=str, default="outputs/activation_maximization",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data/CUB_200_2011",
                        help="CUB data directory (for attribute names)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Model definitions: (name, loader_func, checkpoint_path, normalization_mean, normalization_std)
    # Inception: mean=[0.5,0.5,0.5], std=[2,2,2] — normalization is done INSIDE the backbone
    # So the input image is in [0, 1] range (after ToTensor), then normalized internally.
    # For activation maximization we optimize in the raw input space that the model expects.
    model_configs = [
        ("protocbm", load_protocbm_xc, "weights/protoCBM-models/independent/xc_sigmoid.pth"),
        ("cbm", load_cbm_xc, "weights/baseline_CUB/concept-mapper-xc/best_model_1-xc.pth"),
        ("oracle-protocbm", load_oracle_protocbm, "outputs/retrain_run25_variance/seed_2/best_model_2.pth"),
    ]

    # Inception normalization (applied inside backbone, but we need it for display)
    display_mean = (0.5, 0.5, 0.5)
    display_std = (2, 2, 2)

    # Load attribute names
    try:
        attr_names = load_attribute_names(args.data_dir)
    except Exception as e:
        print(f"Warning: Could not load attribute names ({e}), using indices.")
        attr_names = [f"concept_{i}" for i in range(N_ATTRIBUTES_CBM)]

    concept_indices = args.concepts
    n_concepts = len(concept_indices)
    n_models = len(model_configs)

    # Create figure: rows = concepts, columns = models
    fig, axes = plt.subplots(n_concepts, n_models, figsize=(5 * n_models, 5 * n_concepts))
    if n_concepts == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for col, (model_name, loader_fn, ckpt_path) in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model, device = loader_fn(ckpt_path)

        for row, cidx in enumerate(concept_indices):
            concept_name = attr_names[cidx] if cidx < len(attr_names) else f"concept_{cidx}"
            print(f"  Optimizing concept {cidx}: {concept_name} ...", end=" ", flush=True)

            img = activation_maximization(
                model,
                concept_idx=cidx,
                device=device,
                img_size=299,
                n_steps=args.n_steps,
                lr=args.lr,
                l2_weight=args.l2_weight,
                tv_weight=args.tv_weight,
            )

            # Get final activation score
            with torch.no_grad():
                final_score = get_concept_scores(model, img)[0, cidx].item()
            print(f"activation={final_score:.3f}")

            # Convert for display
            display_img = tensor_to_displayable(img, display_mean, display_std)

            ax = axes[row, col]
            ax.imshow(display_img)
            ax.set_title(f"{model_name}\n{concept_name}\nscore={final_score:.2f}", fontsize=9)
            ax.axis("off")

            # Also save individual images
            individual_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(individual_dir, exist_ok=True)
            plt.imsave(
                os.path.join(individual_dir, f"concept_{cidx}_{concept_name.replace(' ', '_').replace('/', '_')}.png"),
                display_img,
            )

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "activation_maximization_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrid saved to: {grid_path}")
    print(f"Individual images saved to: {args.output_dir}/{{model_name}}/")


if __name__ == "__main__":
    main()
