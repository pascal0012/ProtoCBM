import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from models.components import FC

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cub.config import N_ATTRIBUTES_CBM


class ProtoMod(nn.Module):
    def __init__(self, channel_dim: int = 2048, num_vectors: int = 1):
        super(ProtoMod, self).__init__()

        prototype_shape = [N_ATTRIBUTES_CBM * num_vectors, channel_dim, 1, 1]
        self.prototype_vectors = nn.Parameter(
            2e-4 * torch.rand(prototype_shape), requires_grad=True
        )
        self.num_vectors = num_vectors

    def forward(self, x):
        batch_size = x.shape[0]

        attention_map = F.conv2d(
            input=x, weight=self.prototype_vectors
        )  # [64, num_attributes x num_vectors, H, W]
        similarity_score = F.max_pool2d(
            attention_map, kernel_size=attention_map.size(-1)
        ).view(batch_size, -1)

        similarity_score = similarity_score.reshape(
            batch_size, N_ATTRIBUTES_CBM, -1
        )  # [batch_size, num_attributes, num_vectors]

        # Gets max scores and indices as tuple
        max_similarity_score, max_indices = similarity_score.max(dim=2)

        # For each attribute, get the attention map of that prototype vector that had the maximum activation
        attr_offsets = torch.arange(N_ATTRIBUTES_CBM, device=x.device).view(1, -1) * self.num_vectors
        channel_indices = max_indices + attr_offsets
        attention_map_max = attention_map[torch.arange(batch_size).unsqueeze(1), channel_indices]  # [batch_size, num_attributes, H, W]
        return max_similarity_score, attention_map_max


class CBMMapper(nn.Module):
    def __init__(self, channel_dim, expand_dim, is_aux):
        """
            Args:
                expand_dim: The dimensionality of the hidden layer MLP. If = 0, no extra hidden layer is inserted, but a direct mapping is cretated.
        """
        super(CBMMapper, self).__init__()

        self.all_fc = nn.ModuleList()
        for _ in range(N_ATTRIBUTES_CBM):
            self.all_fc.append(FC(channel_dim, 1, expand_dim))

        self.is_aux = is_aux


    def forward(self, x):
        """Given a feature map of shape [B, C, H , W], creates concepts from it."""
        # Adaptive average pooling
        
        #! Wir verwenden für aux und normal die selben operationen
        #! also im originalcode gibt es kein dropout für aux

        # N x C x 1 x 1
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        if not self.is_aux:
            x = F.dropout(x, training=self.training)

        # N x C
        x = x.view(x.size(0), -1)
        
        out = []
        for fc in self.all_fc:
            out.append(fc(x))

        return out
 
