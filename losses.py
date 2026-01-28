from argparse import Namespace

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
    ):
        # L_reg from the APN paper
        attribute_reg_loss = self.reg_weights["attribute_reg"] * F.mse_loss(
            similarity_scores, attribute_labels
        )
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

        # Experimental loss to enforce low variance across activated attention maps per group
        consistency_loss = self.compute_consistency_loss(
            attention_maps, similarity_scores
        )
        loss += consistency_loss

        return (
            loss,
            attribute_reg_loss,
            cpt_loss,
            decorrelation_loss,
            consistency_loss,
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
