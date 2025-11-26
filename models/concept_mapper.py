from models.components import FC
import torch.nn.functional as F
from torch import nn
import torch

from utils.apn_consts import NUM_ATTRIBUTES


class ProtoMod(nn.Module):
    def __init__(self, channel_dim: int = 2048, num_vectors: int = 1):
        super(ProtoMod, self).__init__()

        prototype_shape = [NUM_ATTRIBUTES * num_vectors, channel_dim, 1, 1]
        self.prototype_vectors = nn.Parameter(
            2e-4 * torch.rand(prototype_shape), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        attention_map = F.conv2d(
            input=x, weight=self.prototype_vectors
        )  # [64, num_attributes x num_vectors, H, W]
        similarity_score = F.max_pool2d(
            attention_map, kernel_size=attention_map.size(-1)
        ).view(batch_size, -1)

        similarity_score = similarity_score.reshape(
            batch_size, NUM_ATTRIBUTES, -1
        )  # [batch_size, num_attributes, num_vectors]

        max_similarity_score = similarity_score.max(dim=2)[
            0
        ]  # [batch_size, num_attributes]

        return max_similarity_score, attention_map


class CBMMapper(nn.Module):
    def init(self, expand_dim):
        """
            Args:
                expand_dim: The dimensionality of the hidden layer MLP. If = 0, no extra hidden layer is inserted, but a direct mapping is cretated.
        """
        super(CBMMapper, self).__init__()
        for _ in range(self.n_attributes):
            self.all_fc.append(FC(2048, 1, expand_dim))


    def forward(self, x):
        """
            Given a feature map of shape [B, C, H , W], creates concepts from it.
        """
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        return out
 
