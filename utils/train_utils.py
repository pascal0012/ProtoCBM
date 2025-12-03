from argparse import Namespace
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from losses import ProtoModLoss
from models.concept_mapper import ProtoMod
from models.models import ModelXtoC, ModelXtoCtoY, ModelXtoY


def prepare_model(model: nn.Module, args: Namespace, load_weights: bool = False):
    # Load in weights, if any
    if load_weights:
        path_to_weights = args.apn_weights_dir if args.model_name == "apn" else os.path.join(args.log_dir, "best_model_1.pth")
        model.load_state_dict(torch.load(path_to_weights))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.compile()

    return model, device


def logger_and_summarywriter(args: Namespace):
    os.makedirs(args.log_dir, exist_ok=True)

    logger = Logger(os.path.join(args.log_dir, "log.txt"))
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
    protomod_criterion = ProtoModLoss(model.concept_mapper, model.backbone.output_map_size, args)

    return cross_entropy, protomod_criterion


class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg + "\n")
        if self.file is not None:
            self.file.write(msg + "\n")

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
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
        self.val = np.array([0. for _ in range(self.n_losses)])
        self.avg = np.array([0. for _ in range(self.n_losses)])
        self.sum = np.array([0. for _ in range(self.n_losses)])
        self.count = 0

    def update(self, val, n=1):
        assert len(val) == len(self.loss_labels), "Loss labels and loss values must be of same length."
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


def compute_accuracies(
    outputs: torch.Tensor,
    labels,
    epoch: int,
    class_acc_meter: AverageMeter,
    tb_writer: SummaryWriter,
) -> AverageMeter:
    """Helper function that combines accuracy computation and logging."""

    # Calculate classification accuracy
    class_acc = accuracy(
        outputs, labels, topk=(1,)
    )  # only care about class prediction accuracy
    class_acc_meter.update(class_acc[0], outputs.size(0))

    tb_writer.add_scalar("Class Accuracy/train", class_acc_meter.avg.item(), epoch)

    return class_acc_meter


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