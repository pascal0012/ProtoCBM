from argparse import Namespace

from torch import nn
import torch

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.backbones import Inception3, DINO
from models.components import MLP
from models.concept_mapper import CBMMapper, ProtoMod
from models.model_connector import ModelConnector

from cub.config import N_CLASSES


def ModelXtoC(args: Namespace):
    backbone = backbone_by_name(args)

    concept_mapper = concept_mapper_by_name(args, backbone.final_channel_dim)

    # The auxiliary logits need a separate concept mapper, as they are of different channel dimensionality + feature map shape
    concept_mapper_aux = None
    if args.use_aux:
        concept_mapper_aux = concept_mapper_by_name(
            args, backbone.aux_final_channel_dim, is_aux=True
        )

    classifier = None
    return ModelConnector(
        backbone,
        concept_mapper,
        classifier,
        args.use_aux,
        args.concept_activation,
        concept_mapper_aux,
    )


def ModelCtoY(args: Namespace):
    # TODO: Load pretrained concept mapper here
    classifier = MLP(
        input_dim=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )

    return ModelConnector(
        backbone,
        concept_mapper,
        classifier,
        args.use_aux,
        args.concept_activation,
        concept_mapper_aux,
    )


def ModelXtoCtoY(args: Namespace):
    backbone = backbone_by_name(args)

    concept_mapper = concept_mapper_by_name(args, backbone.final_channel_dim)

    # The auxiliary logits need a separate concept mapper, as they are of different channel dimensionality + feature map shape
    concept_mapper_aux = None
    if args.use_aux:
        concept_mapper_aux = concept_mapper_by_name(
            args, backbone.aux_final_channel_dim, is_aux=True
        )

    classifier = MLP(
        input_dim=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    return ModelConnector(
        backbone,
        concept_mapper,
        classifier,
        args.use_aux,
        args.concept_activation,
        concept_mapper_aux,
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
            args.expand_dim,
            args.backbone_pretrained,
            args.backbone_freeze,
            299
        )
    if "dino" in args.backbone:
        return DINO(
            args.use_aux,
            args.backbone_pretrained,
            args.backbone_freeze,
            224,
            args.backbone,
        )
    else:
        raise ValueError(f"Unknown backbone name: {args.backbone}")


def concept_mapper_by_name(args: Namespace, input_channel_dim: int, is_aux: bool = False) -> nn.Module:
    if args.concept_mapper == "protomod":
        return ProtoMod(channel_dim=input_channel_dim, num_vectors=args.proto_n_vectors)
    elif args.concept_mapper == "cbm":
        return CBMMapper(input_channel_dim, args.expand_dim, is_aux)
    else:
        raise ValueError(f"Unknown concept mapper name: {args.concept_mapper}")
