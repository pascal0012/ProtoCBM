from argparse import Namespace
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, zeros_like

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils_protocbm.mappings import (
    MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS,
    PART_SEG_GROUPS,
)
from utils_protocbm.index_translation import map_attribute_ids_from_cub_to_cbm
from utils_protocbm.protomod_utils import add_glasso, get_middle_graph
from models.concept_mapper import ProtoMod
from localization.gaussian_targets import GaussianTargetGenerator


class ProtoModLoss(nn.Module):
    def __init__(self, protomod: ProtoMod, kernel_size: int, args: Namespace):
        super(ProtoModLoss, self).__init__()

        self.protomod = protomod
        self.reg_weights = {
            "attribute_reg": args.loss_weight_attribute_reg,
            "cpt": args.loss_weight_map_compactness,
            "decorrelation": args.loss_weight_attribute_decorrelation,
        }
        self.use_groups = args.loss_decorrelation_per_group

        # Compute per-attribute pos_weight for class imbalance
        from cub.dataset import find_class_imbalance
        train_data_path = os.path.join(args.data_dir, "train.pkl")
        imbalance_ratios = find_class_imbalance(train_data_path, multiple_attr=True)
        self.register_buffer(
            "pos_weight", torch.FloatTensor(imbalance_ratios[:args.n_attributes])
        )

        self.middle_graph = get_middle_graph(kernel_size)

        # To calculate regularization among groups
        self.groups = PART_SEG_GROUPS
        self.attributes_per_group = MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS

        # Precompute group attribute indices as tensors for faster indexing
        # TODO: Unify this mapping with group_ids below
        if self.use_groups:
            self.group_attr_indices = [
                torch.tensor(
                    map_attribute_ids_from_cub_to_cbm(self.attributes_per_group[group]),
                    dtype=torch.long,
                )
                for group in self.groups[:-1]
            ]

        # One hot encoding for each group
        self.group_ids = torch.zeros(
            [args.n_attributes, len(PART_SEG_GROUPS) - 1],
            dtype=torch.bool,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )  # [num_attributes, num_groups]
        for i, group in enumerate(PART_SEG_GROUPS[:-1]):
            attributes = map_attribute_ids_from_cub_to_cbm(
                MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS[group]
            )

            self.group_ids[:, i].scatter_(
                0, torch.tensor(attributes, device=self.group_ids.device), 1
            )

    def forward(
        self,
        similarity_scores: torch.Tensor,
        attention_maps: torch.Tensor,
        attribute_labels: torch.Tensor,
        aux_forward: bool = False,
    ):
        # L_reg: BCE with logits + per-attribute pos_weight for class imbalance
        # pos_weight = self.pos_weight.to(similarity_scores.device)
        attribute_reg_loss = self.reg_weights["attribute_reg"] * F.binary_cross_entropy_with_logits(
            similarity_scores, attribute_labels, #pos_weight=pos_weight
        )
        if aux_forward:
            return (attribute_reg_loss, attribute_reg_loss, None, None)
        
        loss = attribute_reg_loss

        batch_size, num_attributes, map_dim, _ = attention_maps.size()

        # L_cpt from the APN paper: Enforces compactness of the attention maps
        peak_id = torch.argmax(
            attention_maps.reshape(batch_size * num_attributes, -1), dim=1
        )
        peak_mask = self.middle_graph[peak_id, :, :].view(
            batch_size, num_attributes, map_dim, map_dim
        )
        cpt_loss = self.reg_weights["cpt"] * torch.mean(
            F.sigmoid(attention_maps) * peak_mask
        )
        loss += cpt_loss

        num_vectors = self.protomod.prototype_vectors.shape[0] // num_attributes
        prototypes = self.protomod.prototype_vectors.squeeze().reshape(
            num_attributes, num_vectors, -1
        )  # [num_attributes, num_vectors, channel_dim]

        if self.use_groups:
            # L_AD in the APN paper: Attribute decorrelation loss
            decorrelation_loss = zeros_like(cpt_loss)
            for group_attr_idx in self.group_attr_indices:
                # Only selects prototypes that are relevant for this group - Enforces competition per group
                decorrelation_loss += self.reg_weights["decorrelation"] * add_glasso(
                    prototypes, group_attr_idx
                )
            loss += decorrelation_loss
        else:
            # Enforces competition between attributes
            decorrelation_loss = self.reg_weights["decorrelation"] * prototypes.norm(2)
            loss += decorrelation_loss

        # DEACTIVATED: Drastically increased localization distance
        # Experimental loss to enforce low variance across activated attention maps per group
        # consistency_loss = self.compute_consistency_loss(
        #     attention_maps, similarity_scores
        # )
        # loss += consistency_loss

        return (
            loss,
            attribute_reg_loss,
            cpt_loss,
            decorrelation_loss,
        )

    def compute_consistency_loss(self, attention_maps, similarity_scores):
        """
        Checks whether all attention maps with high activations are consistent per group

        :param attention_maps: [batch_size, num_attributes, H, W]
        :param similarity_scores: [batch_size, num_attributes]
        :return: scalar consistency loss
        """

        batch_size, num_attributes, H, W = attention_maps.shape

        # similarity mask
        similarity_mask = (similarity_scores > 0.5)[
            :, :, None
        ]  # [batch_size, num_attributes, 1]

        # group mask
        group_mask = self.group_ids[None, :, :]  # [1, num_attributes, num_groups]

        # combined validity mask
        mask = similarity_mask * group_mask  # [batch_size, num_attributes, num_groups]

        activated_maps_per_group = (
            attention_maps.unsqueeze(2) * mask[:, :, :, None, None]
        )  # [batch_size, num_attributes, num_groups, H, W]

        # Compute mean per group
        sum_per_group = activated_maps_per_group.sum(dim=[0, 1])  # [num_groups, H, W]
        count_per_group = mask.sum(dim=[0, 1])  # [num_groups]

        mean_per_group = sum_per_group / (
            count_per_group[:, None, None] + 1e-8
        )  # [num_groups, H, W]

        # Compute variance per group using mean
        differences = (
            attention_maps.unsqueeze(2) - mean_per_group.unsqueeze(0)
        ) * mask[:, :, :, None, None]  # [batch_size, num_attributes, num_groups, H, W]

        variance_per_feature = (differences**2).sum(dim=[0, 1]) / (
            count_per_group[:, None, None] + 1e-8
        )  # [num_groups, H, W]
        return variance_per_feature.mean()


class LocalizationDistanceLoss(nn.Module):
    """
    Loss function that supervises attention maps using Gaussian target heatmaps
    centered on ground truth keypoints.

    For each attribute, a 2D Gaussian is generated at the attention map's native
    resolution, centered on the GT keypoint of the attribute's corresponding part.
    The model's attention maps (after sigmoid) are compared against these targets
    via MSE loss.
    """

    def __init__(self, part_dict: Dict[int, str], part_attribute_mapping: Dict[str, torch.IntTensor], img_size: int = 299, sigma: float = 1.0, loss_type: str = "mse"):
        """
        Args:
            part_dict: Mapping from part ID to part name
            part_attribute_mapping: Mapping from part name to attribute indices
            img_size: Size of the input images (assumes square images)
            sigma: Standard deviation of the Gaussian targets in feature map space
            loss_type: Loss function to compare predictions and targets ("mse" or "kl")
        """
        super(LocalizationDistanceLoss, self).__init__()
        self.part_dict = part_dict
        self.part_attribute_mapping = part_attribute_mapping
        if loss_type not in ("mse", "kl"):
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'mse' or 'kl'.")
        self.loss_type = loss_type

        # Create lookup table: attr_idx -> part_idx
        num_attributes = max(
            idx.max().item() for indices in part_attribute_mapping.values() for idx in [indices]
        ) + 1
        self.attr_to_part = torch.full((num_attributes,), -1, dtype=torch.long)

        for part_idx, (part_id, part_name) in enumerate(part_dict.items()):
            if part_name in part_attribute_mapping:
                attrs = part_attribute_mapping[part_name]
                self.attr_to_part[attrs] = part_idx

        # Pre-compute which attributes are mapped (avoid per-forward work)
        self.mapped_attrs = (self.attr_to_part != -1).nonzero(as_tuple=True)[0]  # [M]
        self.mapped_part_indices = self.attr_to_part[self.mapped_attrs]  # [M]

        self.target_generator = GaussianTargetGenerator(img_size=img_size, sigma=sigma)

    def forward(
        self,
        attention_maps: torch.Tensor,
        part_gts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian heatmap supervision loss.

        Args:
            attention_maps: [B, A, H, W] attention maps from the model
            part_gts: [B, K, 2] ground truth keypoint coordinates (x, y) in image space

        Returns:
            loss: Scalar MSE loss between sigmoid(attention_maps) and Gaussian targets
        """
        B, A, H, W = attention_maps.shape
        M = len(self.mapped_attrs)
        device = attention_maps.device

        # Move lookup tables to device if needed
        if self.attr_to_part.device != device:
            self.attr_to_part = self.attr_to_part.to(device)
            self.mapped_attrs = self.mapped_attrs.to(device)
            self.mapped_part_indices = self.mapped_part_indices.to(device)

        # Generate Gaussian targets and validity mask
        target, valid_mask = self.target_generator.generate(
            part_gts, self.mapped_part_indices, H, W
        )

        mask_expanded = valid_mask[:, :, None, None]  # [B, M, 1, 1]
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if self.loss_type == "mse":
            # Normalize attention maps to [0, 1]
            pred = torch.sigmoid(attention_maps[:, self.mapped_attrs, :, :])  # [B, M, H, W]
            pred_masked = pred * mask_expanded
            target_masked = target * mask_expanded
            loss = (pred_masked - target_masked).pow(2).sum() / (n_valid * H * W)
        else:
            # KL divergence: treat both as spatial distributions per attribute
            # Normalize target to a probability distribution over H*W
            target_flat = (target * mask_expanded).view(B, M, -1)  # [B, M, H*W]
            target_dist = target_flat / (target_flat.sum(dim=-1, keepdim=True) + 1e-8)

            # Normalize predictions via log-softmax over spatial dimensions
            pred_logits = attention_maps[:, self.mapped_attrs, :, :]  # [B, M, H, W]
            pred_flat = (pred_logits * mask_expanded).view(B, M, -1)  # [B, M, H*W]
            pred_log_dist = F.log_softmax(pred_flat, dim=-1)

            # KL(target || pred) per valid attribute, averaged over valid count
            kl = F.kl_div(pred_log_dist, target_dist, reduction='none').sum(dim=-1)  # [B, M]
            loss = (kl * valid_mask).sum() / n_valid

        return loss
