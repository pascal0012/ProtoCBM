import argparse
import os
import sys
from argparse import Namespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from losses import ProtoModLoss
from models.concept_mapper import ProtoMod
from models.models import ModelXtoC, ModelXtoCtoY, ModelXtoY


def prepare_model(
    model: nn.Module,
    args: Namespace,
    load_weights: bool = False,
    training: bool = False,
):
    # Load in weights, if any
    if load_weights:
        path_to_weights = (
            args.apn_weights_dir
            if args.model_name == "apn"
            else os.path.join(args.log_dir, f"best_model_{args.seed}.pth")
        )
        state_dict = torch.load(path_to_weights)

        # Remove auxiliary logits and concept mapper, as it is not needed for inference
        if not args.model_name == "apn" and not training:
            print(
                "Deleting all weights for auxiliary logits and auxiliary mappers from loaded model weights..."
            )
            keys_to_remove = [
                k
                for k in state_dict.keys()
                if "AuxLogits" in k or "aux_concept_mapper" in k
            ]
            for k in keys_to_remove:
                del state_dict[k]
        model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.compile()

    return model, device


def logger_and_summarywriter(args: Namespace):
    os.makedirs(args.log_dir, exist_ok=True)

    write_console = args.write_console if hasattr(args, "write_console") else True
    logger = Logger(os.path.join(args.log_dir, "log.txt"), write_console=write_console)
    for k, v in vars(args).items():
        logger.write(f"{k}: {v}")
    logger.flush()

    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))

    return logger, tb_writer


def optimizer_and_scheduler_by_name(model: nn.Module, args: Namespace):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.1
    )

    return optimizer, scheduler


def model_by_mode(args: Namespace) -> nn.Module:
    if args.mode == "XCY":
        model = ModelXtoCtoY(args)
    elif args.mode == "XY":
        model = ModelXtoY(args)
    elif args.mode == "XC":
        model = ModelXtoC(args)
    elif args.mode == "CY":
        model = ModelXtoCtoY(args)
        raise NotImplementedError("CY mode not implemented yet")

    else:
        raise ValueError(f"Unknown mode {args.mode}")

    return model


def create_criterions(model: nn.Module, args: Namespace):
    assert (
        model.concept_mapper is not None and type(model.concept_mapper) is ProtoMod
    ), "Model does not have a concept mapper for ProtoModLoss"

    cross_entropy = nn.CrossEntropyLoss()
    protomod_criterion = ProtoModLoss(
        model.concept_mapper, model.backbone.output_map_size, args
    )

    return cross_entropy, protomod_criterion

def build_attr_criterion(
    args: Namespace,
    imbalance: Optional[List[float]],
    device: torch.device,
) -> Optional[List[nn.Module]]:
    """Build attribute criterion based on configuration."""
    if not args.use_attr or args.no_img:
        return None

    attr_criterion = []
    if args.weighted_loss and imbalance is not None:
        for ratio in imbalance:
            attr_criterion.append(
                nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device))
            )
    else:
        attr_criterion = [nn.CrossEntropyLoss() for _ in range(args.n_attributes)]
    
    return attr_criterion


class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None, write_console=True):
        self.write_console = write_console
        if self.write_console:
            self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.write_console:
            self.console.write(msg + "\n")
        if self.file is not None:
            self.file.write(msg + "\n")

    def flush(self):
        if self.write_console:
            self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.write_console:
            self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeter(object):
    """
    Computes and stores the average and current value for multiple losses.
    """

    def __init__(self, loss_labels):
        self.loss_labels = loss_labels
        self.n_losses = len(loss_labels)
        self.reset()

    def reset(self):
        self.val = np.array([0.0 for _ in range(self.n_losses)])
        self.avg = np.array([0.0 for _ in range(self.n_losses)])
        self.sum = np.array([0.0 for _ in range(self.n_losses)])
        self.count = 0

    def update(self, val, n=1):
        assert len(val) == len(self.loss_labels), (
            "Loss labels and loss values must be of same length."
        )
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to("cuda" if torch.cuda.is_available() else "cpu")
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_accuracy(similarity_scores, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = similarity_scores >= 0.5

    acc = (pred.int()).eq(target.int()).sum()
    acc = acc * 100 / np.prod(np.array(target.size()))
    return acc

def compute_attr_accuracy(attributes, attr_labels_var):
    """Compute binary accuracy over all attributes."""
    sigmoid_outputs = torch.nn.Sigmoid()(attributes)
    return binary_accuracy(sigmoid_outputs, attr_labels_var), attributes.size(0)

def compute_accuracies(
    outputs: torch.Tensor,
    similarity_scores: torch.Tensor,
    attribute_labels: torch.Tensor,
    labels,
    epoch: int,
    class_acc_meter: AverageMeter,
    attr_acc_meter: AverageMeter,
    tb_writer: SummaryWriter,
) -> Tuple[AverageMeter, AverageMeter]:
    """Helper function that combines accuracy computation and logging."""

    # Calculate classification accuracy
    class_acc = accuracy(
        outputs, labels, topk=(1,)
    )  # only care about class prediction accuracy
    class_acc_meter.update(class_acc[0], outputs.size(0))

    tb_writer.add_scalar("Class Accuracy/train", class_acc_meter.avg.item(), epoch)

    # Calculate attribute accuracy
    attr_acc = binary_accuracy(similarity_scores, attribute_labels)
    attr_acc_meter.update(attr_acc, outputs.size(0))

    tb_writer.add_scalar("Attribute Accuracy/train", attr_acc_meter.avg.item(), epoch)

    return class_acc_meter, attr_acc_meter


def normalize_scientific_floats(cfg):
    """
    Recursively convert strings that contain 'e' and look like floats into floats to parse
    yaml arguments like 1e9 into their proper float.
    """
    if isinstance(cfg, dict):
        return {k: normalize_scientific_floats(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [normalize_scientific_floats(v) for v in cfg]
    elif isinstance(cfg, str) and "e" in cfg.lower():
        try:
            return float(cfg)
        except ValueError:
            return cfg
    else:
        return cfg

def process_cbm_config(args: Namespace) -> Namespace:
    # if mode = 'XC', create a argument bottleneck
    if args.mode == "XC":
        args.use_attr = True
        args.no_img = False
        args.bottleneck = True
    
    return args

def gather_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/debug.yaml",
        help="Path to config file (YAML)",
    )
    cli_args = parser.parse_args()

    # Load the config yaml
    with open(cli_args.config) as f:
        args = yaml.safe_load(f)
    args = normalize_scientific_floats(args)

    # Add run name, keep as namespace to be able to access like args.param
    args = Namespace(**args, config_path=cli_args.config)

    return args
