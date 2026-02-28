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

# Add ConceptBottleneck to path for loading legacy checkpoints that use CUB.template_model
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "ConceptBottleneck")
)

from losses import ProtoModLoss
from models.concept_mapper import ProtoMod
from models.models import ModelCtoY, ModelXtoC, ModelXtoCtoY, ModelXtoY
from models.apn_baseline import load_apn_baseline


def _resolve_weights_path(args: Namespace) -> str:
    """Return the checkpoint path from args, falling back to ``log_dir/best_model_{seed}.pth``."""
    if "weight_dir" in vars(args):
        return args.weight_dir
    if hasattr(args, "checkpoint") and args.checkpoint:
        return args.checkpoint
    if args.model_name == "apn":
        return args.apn_weights_dir
    return os.path.join(args.log_dir, f"best_model_{args.seed}.pth")


def _clean_state_dict(
    state_dict: dict, 
    args: Namespace, 
    training: bool
) -> dict:
    """Unwrap, strip aux weights, and remap legacy keys in a loaded state dict."""
    # Unwrap full model saves
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    # Drop aux weights not needed for inference
    if args.model_name != "apn" and not training:
        print("Deleting auxiliary logits and mapper weights from checkpoint...")
        aux_keys = [
            k for k in state_dict if "AuxLogits" in k or "aux_concept_mapper" in k
        ]
        for k in aux_keys:
            del state_dict[k]

    # Remap legacy End2EndModel keys → ModelConnector keys
    # Legacy:  first_model (Inception3 + all_fc), sec_model (MLP classifier)
    # Current: backbone, concept_mapper.all_fc, classifier
    if any(k.startswith("first_model.") for k in state_dict):
        print("Remapping legacy End2EndModel keys to ModelConnector format...")
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("first_model.all_fc."):
                remap[k.replace("first_model.all_fc.", "concept_mapper.all_fc.")] = v
            elif k.startswith("first_model."):
                remap[k.replace("first_model.", "backbone.")] = v
            elif k.startswith("sec_model."):
                remap[k.replace("sec_model.", "classifier.")] = v
            else:
                remap[k] = v
        state_dict = remap

    # Remap flat legacy XC-only checkpoint keys → ModelConnector keys
    # Legacy:  Conv2d_*/Mixed_*/fc.* (flat Inception3), all_fc.* (concept head)
    # Current: backbone.*, concept_mapper.all_fc.*
    elif any(k.startswith("all_fc.") for k in state_dict):
        print("Remapping flat legacy XC keys to ModelConnector format...")
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("all_fc."):
                remap["concept_mapper." + k] = v
            else:
                remap["backbone." + k] = v
        state_dict = remap

    # Remap flat legacy CY-only checkpoint keys → ModelConnector keys
    # Legacy:  linear.weight / linear.bias
    # Current: classifier.linear.weight / classifier.linear.bias
    elif list(state_dict.keys()) == ["linear.weight", "linear.bias"]:
        print("Remapping flat legacy CY keys to ModelConnector format...")
        state_dict = {"classifier." + k: v for k, v in state_dict.items()}

    return state_dict


def prepare_model(
    model: nn.Module,
    args: Namespace,
    load_weights: bool = False,
    training: bool = False,
    compile: bool = True,
) -> Tuple[nn.Module, str]:
    """Load weights (optional), move model to device, and optionally compile it.

    Returns the model on the target device and the device string (``"cuda"`` or ``"cpu"``).
    """
    if load_weights:
        path = _resolve_weights_path(args)
        state_dict = torch.load(path, weights_only=False)
        state_dict = _clean_state_dict(state_dict, args, training)
        model.load_state_dict(state_dict, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if compile:
        model.compile()

    return model, device


def logger_and_summarywriter(
    args: Namespace, 
    close_console: bool = True
) -> Tuple["Logger", SummaryWriter]:
    """Create a file logger and a TensorBoard writer, both rooted at ``log_dir/model_name/``."""
    os.makedirs(os.path.join(args.log_dir, args.model_name), exist_ok=True)

    write_console = getattr(args, "write_console", True)
    logger = Logger(
        os.path.join(args.log_dir, args.model_name, "log.txt"),
        write_console=write_console,
        close_console=close_console,
    )
    for k, v in vars(args).items():
        logger.write(f"{k}: {v}")
    logger.flush()

    tb_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, args.model_name, "tensorboard")
    )

    return logger, tb_writer


def optimizer_and_scheduler_by_name(
    model: nn.Module, 
    args: Namespace
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    """Build optimizer (Adam/AdamW/RMSprop/SGD) and a StepLR scheduler from args."""
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
    """Instantiate the model class matching ``args.mode`` (XCY, XY, XC, CY)."""
    if args.mode == "XCY" or args.mode == "XCCY" or args.mode == "XC->CY":
        model = ModelXtoCtoY(args)
    elif args.mode == "XY":
        model = ModelXtoY(args)
    elif args.mode == "XC":
        model = ModelXtoC(args)
    elif args.mode == "CY" or args.mode == "C*Y":
        model = ModelCtoY(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    if bool(getattr(args, "checkpoint", False)):
        loaded = torch.load(
            args.checkpoint,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Handle case where full model was saved instead of just state_dict
        if hasattr(loaded, "state_dict"):
            loaded = loaded.state_dict()
        model.load_state_dict(loaded, strict=False)
        print("Continuing with checkpoint:", args.checkpoint)

    return model


def create_model(args: Namespace) -> nn.Module:
    """Create and return a model based on the model_name in args.

    Supports: protocbm, cbm, apn
    """
    if args.model_name == "protocbm":
        model = model_by_mode(args)
    elif args.model_name == "cbm":
        model = model_by_mode(args)
    elif args.model_name == "apn":
        model = load_apn_baseline(args)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    return model


def create_criterions(model: nn.Module, args: Namespace) -> Tuple[nn.CrossEntropyLoss, ProtoModLoss]:
    """Return ``(cross_entropy, protomod_criterion)`` for a model with a ProtoMod concept mapper."""
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
    """Build one loss per attribute: weighted BCEWithLogitsLoss or CrossEntropyLoss.

    Returns ``None`` if attributes are disabled (``use_attr=False`` or ``no_img=True``).
    """
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

    def __init__(self, fpath=None, write_console=True, close_console=True):
        self.write_console = write_console
        self.close_console = close_console

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
        if self.write_console and self.close_console:
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
    Computes the accuracy for multiple binary predictions.
    Applies sigmoid to raw similarity scores before thresholding at 0.5.
    output and target are Torch tensors
    """
    similarity_scores = torch.nn.Sigmoid()(similarity_scores)
    pred = similarity_scores >= 0.5

    acc = (pred.int()).eq(target.int()).sum()
    acc = acc * 100 / np.prod(np.array(target.size()))
    return acc


def compute_attr_accuracy(attributes, attr_labels_var):
    """Compute binary accuracy over all attributes."""
    return binary_accuracy(attributes, attr_labels_var), attributes.size(0)


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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config file)",
    )
    cli_args = parser.parse_args()

    # Load the config yaml
    with open(cli_args.config) as f:
        args = yaml.safe_load(f)

    # Recursively load base config if provided
    base_config = args.get("base_config", None)
    i = 0
    while base_config is not None:
        print(f"Loading base config from '{base_config}'.")
        with open(base_config) as f:
            base_args = yaml.safe_load(f)

        base_config = base_args.get("base_config", None)

        args = base_args | args

        if i > 10:
            raise ValueError(
                "Too many levels of base config. Possible circular reference."
            )

    args = normalize_scientific_floats(args)

    # If val_metric is only one entry
    if "val_metric" in args.keys():
        val_metrics = args["val_metric"]
        if isinstance(val_metrics, str):
            args["val_metric"] = [val_metrics]

        # Ensure only valid metrics are provided (🥲)
        assert all(
            m in ["class_acc", "attr_acc", "seg_iou", "dist_loc"] for m in args["val_metric"]
        ), "val_metrics must be chosen from 'class_acc', 'attr_acc', 'seg_iou', or 'dist_loc'"

    args = Namespace(**args, config_path=cli_args.config)

    # Override checkpoint from CLI if provided
    if cli_args.checkpoint is not None:
        args.checkpoint = cli_args.checkpoint

    return args
