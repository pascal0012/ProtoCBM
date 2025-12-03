"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer option for freezing lower layers except fc
"""

from typing import Literal, Optional
import torch.nn as nn
import torch.nn.functional as F

# We need this instead of a lamba since pytorch can't serialize lambdas
def identity(x):
    return x

class ModelConnector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        concept_mapper: Optional[nn.Module],
        classifier: Optional[nn.Module],
        use_aux: bool = False,
        concept_activation: Optional[Literal["sigmoid", "relu"]] = None,
        aux_concept_mapper: Optional[nn.Module] = None,
    ):
        super(ModelConnector, self).__init__()
        self.backbone = backbone
        self.concept_mapper = concept_mapper
        self.aux_concept_mapper = aux_concept_mapper
        self.classifier = classifier
        self.use_aux = use_aux

        # Create activation function of concepts, if any
        if concept_activation == "sigmoid":
            self.concept_activation = F.sigmoid
        elif concept_activation == "relu":
            self.concept_activation = F.relu
        else:
            self.concept_activation = identity

    def forward_features(self, features, aux_forward=False):
        # Saves concept mapper outputs, if any
        maps, sim_scores = None, None

        # Map features to concepts, if we have a mapper
        if self.concept_mapper is not None:
            mapper = self.aux_concept_mapper if aux_forward else self.concept_mapper
            sim_scores, maps = mapper(features)
            raw = sim_scores
        else:
            raw = features

        # Apply activation
        output = self.concept_activation(raw)

        # C -> Y, if we have a classifier
        if self.classifier is not None:
            output = self.classifier(output)

        return (output, sim_scores, maps) if not aux_forward else output

    def forward(self, x):
        if self.training and self.use_aux:
            assert self.backbone is not None, "Backbone must be defined when using auxiliary outputs."
            output, aux_output = self.backbone(x)
            # Unpack main feature tuple, construct one tuple
            return (*self.forward_features(output), self.forward_features(aux_output, aux_forward=True))

        if self.backbone is not None:
            x = self.backbone(x)
        return self.forward_features(x)
