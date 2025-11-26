from argparse import Namespace

from torch import nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.backbones import Inception3
from models.components import MLP
from models.concept_mapper import CBMMapper, ProtoMod
from models.model_connector import ModelConnector

from cub.config import N_CLASSES


def ModelXtoC(args: Namespace):
    backbone = backbone_by_name(args)

    concept_mapper = concept_mapper_by_name(args, backbone.final_channel_dim)

    classifier = MLP(
        input_dim=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    return ModelConnector(
        backbone, concept_mapper, classifier, args.use_aux, args.concept_activation
    )

def ModelCtoY(args: Namespace):
    classifier = MLP(
        input_dim=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    
    return classifier


def ModelXtoCtoY(args: Namespace):
    backbone = backbone_by_name(args)

    concept_mapper = concept_mapper_by_name(args, backbone.final_channel_dim)

    classifier = MLP(
        input_dim=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    return ModelConnector(
        backbone, concept_mapper, classifier, args.use_aux, args.concept_activation
    )


def ModelXtoY(args: Namespace):
    backbone = backbone_by_name(args)

    classifier = MLP(
        input_dim=backbone.final_channel_dim,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    return ModelConnector(
        backbone, None, classifier, args.use_aux, args.concept_activation
    )


def backbone_by_name(args: Namespace) -> Inception3:
    if args.backbone == "inception":
        return Inception3(
            args.use_aux,
            args.n_attributes,
            args.expand_dim,
            args.backbone_pretrained,
            args.backbone_freeze,
        )
    else:
        raise ValueError(f"Unknown backbone name: {args.backbone}")


def concept_mapper_by_name(args: Namespace, input_channel_dim: int) -> nn.Module:
    if args.concept_mapper == "protomod":
        return ProtoMod(channel_dim=input_channel_dim, num_vectors=args.proto_n_vectors)
    elif args.concept_mapper == "cbm":
        return CBMMapper(args.expand_dim)
    else:
        raise ValueError(f"Unknown concept mapper name: {args.concept_mapper}")
