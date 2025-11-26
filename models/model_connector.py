"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer option for freezing lower layers except fc
"""

from typing import Literal, Optional
import torch.nn as nn
import torch.nn.functional as F


class ModelConnector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        concept_mapper: Optional[nn.Module],
        classifier: Optional[nn.Module],
        use_aux: bool = False,
        concept_activation: Optional[Literal["sigmoid", "relu"]] = None,
    ):
        super(ModelConnector, self).__init__()
        self.models = nn.ModuleList()
        self.backbone = backbone
        self.concept_mapper = concept_mapper
        self.classifier = classifier
        self.use_aux = use_aux
        self.concept_activation = concept_activation

    def forward_features(self, features, return_maps=True):
        maps=None
        if self.concept_mapper is not None:
            features, maps = self.concept_mapper(features)

        if self.concept_activation == "sigmoid":
            features = F.sigmoid(features)
        elif self.concept_activation == "relu":
            features = F.relu(features)

        if self.classifier is not None:
            features = self.classifier(features)

        if return_maps:
            return features, maps
        return features

    def forward(self, x):
        if self.use_aux:
            assert self.backbone is not None, "Backbone must be defined when using auxiliary outputs."
            output, aux_output = self.backbone(x)
            return self.forward_features(output), self.forward_features(aux_output, return_maps=False)
        else:
            if self.backbone is not None:
                x = self.backbone(x)

            return self.forward_features(x)
