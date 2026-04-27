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
        max_similarity_score, max_indices = similarity_score.max(dim=2) # [batch_size, num_attributes]

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



    def forward(self, x):
        """Given a feature map of shape [B, C, H , W], creates concepts from it."""
        # Adaptive average pooling
        # N x C x 1 x 1
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = F.dropout(x, training=self.training)

        # N x C
        x = x.view(x.size(0), -1)
        
        out = []
        for fc in self.all_fc:
            out.append(fc(x))

        return out
 


class DebugAuxMapperCBM(nn.Module):
    def __init__(self, channel_dim, expand_dim):
        """
            Args:
                expand_dim: The dimensionality of the hidden layer MLP. If = 0, no extra hidden layer is inserted, but a direct mapping is cretated.
        """
        super(DebugAuxMapperCBM, self).__init__()

        self.all_fc = nn.ModuleList()
        for _ in range(N_ATTRIBUTES_CBM):
            self.all_fc.append(FC(channel_dim, 1, expand_dim, stddev=0.001))


        for m in self.modules():
            if isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, "stddev") else 0.1

                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)


    def forward(self, x):
        """Given a feature map of shape [B, C, H , W], creates concepts from it."""
        # Adaptive average pooling
        # N x C x 1 x 1
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1) # N x C
        
        out = []
        for fc in self.all_fc:
            out.append(fc(x))

        return out


class LCBMMapper(nn.Module):
    """Locality-aware Concept Bottleneck Model Mapper.
    
    Implements prototype learning with CLIP guidance for concept localization.
    """
    
    def __init__(self, channel_dim: int = 2048, num_concepts: int = 112, 
                 expand_dim: int = 0, k1: int = 3, k2: int = 2, num_classes: int = 200):
        super(LCBMMapper, self).__init__()
        
        self.channel_dim = channel_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.k1 = k1  # Top-K1 concepts for prototype selection
        self.k2 = k2  # Top-K2 prototypes per patch
        
        # Learnable prototypes: [K, D]
        self.prototypes = nn.Parameter(
            torch.randn(num_concepts, channel_dim) * 0.01,
            requires_grad=True
        )
        
        # Concept prediction linear layer: [D] -> [K]
        self.concept_predictor = nn.Linear(channel_dim, num_concepts)
        
        # Auxiliary classifier: [D] -> [num_classes]
        # For auxiliary loss on class prediction
        self.auxiliary_classifier = nn.Linear(channel_dim, num_classes)
        
    def forward(self, features, clip_scores=None, labels=None):
        """Paper-aligned forward pass.

        Args:
            features: [B, C, H, W] backbone feature map F.
            clip_scores: [B, H*W, K] CLIP concept-patch similarities S. If
                provided, top-K1 concepts per patch are kept (mask); otherwise
                all K concepts are kept.
            labels: unused (kept for API compatibility).

        Returns:
            dict with:
              concept_logits : [B, K]
              concept_scores : [B, K]     sigmoid of concept_logits
              aux_logits     : [B, num_classes]
              M0             : [B, H*W, K]   cosine(F, P) (requires_grad)
              M0_prime       : [B, H*W, K]   M0 with non-top-K2-per-concept set to -inf
              attention_maps : [B, K, H, W]  reshape of M0
              features_flat  : [B, H*W, C]
        """
        batch_size, channel_dim, height, width = features.shape
        hw = height * width

        features_flat = features.permute(0, 2, 3, 1).reshape(batch_size, hw, channel_dim)

        # --- Prototype-patch similarity M0: cosine(F, P) ---
        f_norm = F.normalize(features_flat, dim=-1)        # [B, H*W, C]
        p_norm = F.normalize(self.prototypes, dim=-1)      # [K, C]
        M0 = torch.einsum('bhc,kc->bhk', f_norm, p_norm)   # [B, H*W, K]

        # --- Top-K1 CLIP masking (per patch, over concept axis) ---
        # Reduces M0 -> M by keeping only the K1 concepts per location that
        # CLIP considers most relevant. Without CLIP we fall back to all K.
        if clip_scores is not None:
            clip_scores = clip_scores.reshape(batch_size, hw, -1)
            _, top_k1_indices = torch.topk(clip_scores, self.k1, dim=2)  # [B, H*W, K1]
            k1_mask = torch.zeros_like(M0, dtype=torch.bool)
            k1_mask.scatter_(2, top_k1_indices, True)
        else:
            k1_mask = torch.ones_like(M0, dtype=torch.bool)

        # --- Top-K2 masking per concept (over spatial axis) -> M0' ---
        # For each concept k, keep the K2 spatial locations with highest
        # similarity, mask others to -inf. This is the target distribution for
        # L_local (after softmax over space).
        M0_for_rank = M0.masked_fill(~k1_mask, float('-inf'))        # [B, H*W, K]
        M0_rank = M0_for_rank.permute(0, 2, 1)                        # [B, K, H*W]
        k2 = min(self.k2, hw)
        _, topk2_idx = torch.topk(M0_rank, k2, dim=2)                 # [B, K, K2]
        k2_mask = torch.zeros_like(M0_rank, dtype=torch.bool)
        k2_mask.scatter_(2, topk2_idx, True)
        M0_prime = M0_rank.masked_fill(~k2_mask, float('-inf')).permute(0, 2, 1)  # [B, H*W, K]

        # --- Concept prediction: max-pool similarity over space ---
        concept_logits = M0.max(dim=1).values                         # [B, K]
        concept_scores = torch.sigmoid(concept_logits)

        # --- Auxiliary classifier: softmax-weighted prototype pooling ---
        # Softmax over the (active) concept dimension per patch, then pool
        # prototypes to get a feature vector used to predict the class label.
        M_for_soft = M0.masked_fill(~k1_mask, float('-inf'))
        M_soft = F.softmax(M_for_soft, dim=2)                         # [B, H*W, K]
        weighted_protos = torch.einsum('bhk,kd->bhd', M_soft, self.prototypes)
        weighted_protos_avg = weighted_protos.mean(dim=1)             # [B, D]
        aux_logits = self.auxiliary_classifier(weighted_protos_avg)   # [B, num_classes]

        attention_maps = M0.view(batch_size, height, width, self.num_concepts).permute(0, 3, 1, 2)

        return {
            'concept_logits': concept_logits,
            'concept_scores': concept_scores,
            'aux_logits': aux_logits,
            'M0': M0,
            'M0_prime': M0_prime,
            'attention_maps': attention_maps,
            'features_flat': features_flat,
        }
 
