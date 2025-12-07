"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer option for freezing lower layers except fc
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# We need this instead of a lamba since pytorch can't serialize lambdas
def identity(x):
    return x

class ModelConnector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        concept_mapper: Optional[nn.Module], # wie ist das bitte optional?
        classifier: Optional[nn.Module] = None,
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

        self.forward_fn = self._forward_func_by_name(self.concept_mapper._get_name())

    def _forward_func_by_name(self, name: str):
        if name == "CBMMapper":
            return self.forward_featuresCBM
        elif name == "ProtoMod":
            return self.forward_featuresPROTO
        else:
            raise ValueError(f"Unknown forward function name: {name}")


    def forward_featuresCBM(self, features, aux_forward=False):
        # features [N, 768, 1, 1] -> 768 == backbone.aux_final_channel_dim

        # if self.use_relu:
        #     attr_outputs = [nn.ReLU()(o) for o in features]
        # elif self.use_sigmoid:
        #     attr_outputs = [torch.nn.Sigmoid()(o) for o in features]
        # else:
        #     attr_outputs = features

        # take the Backbone featuremaps and map to feature vector 
        mapper = self.aux_concept_mapper if aux_forward else self.concept_mapper
        mapped_input = mapper(features)

        if self.classifier is None:
            return mapped_input

        # take the feature vector and map to class logits
        cls_input = torch.cat(mapped_input, dim=1)
        
        # [class_logits, attr1, attr2, ..., attrN]
        all_out = [self.classifier(cls_input)]
        all_out.extend(mapped_input)  
        
        return all_out


    def forward_featuresPROTO(self, features, aux_forward=False):
        # Saves concept mapper outputs, if any
        maps, sim_scores = None, None

        # Map features to concepts, if we have a mapper
        if self.concept_mapper is not None:
            mapper = self.aux_concept_mapper if aux_forward else self.concept_mapper
            # sim_scores, maps
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
        #? XCY
        if self.training and self.use_aux:
            assert self.backbone is not None, "Backbone must be defined when using auxiliary outputs."
            # Unpack main feature tuple, construct one tuple
            output, aux_output = self.backbone(x)
            
            # old code  self.forward_stage2(outputs), self.forward_stage2(aux_outputs)
            return self.forward_fn(output), self.forward_fn(aux_output, aux_forward=True)

        if self.backbone is not None:
            x = self.backbone(x)

        #? either ATTR (CY) directly or without aux (eval)    
        return self.forward_fn(x)
