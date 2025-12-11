"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""

import argparse
import math
import os
import time
from argparse import Namespace
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

import nanoid

from cub.config import (
    BASE_DIR,
    LR_DECAY_SIZE,
    MIN_LR,
)

# from CUB import gen_cub_synthetic
from cub.dataset import find_class_imbalance, load_data
from utils_protocbm.misc import Colors
from utils_protocbm.train_utils import (
    AverageMeter,
    LossMeter,
    accuracy,
    build_attr_criterion,
    compute_accuracies,
    compute_attr_accuracy,
    logger_and_summarywriter,
    model_by_mode,
    normalize_scientific_floats,
    optimizer_and_scheduler_by_name,
    prepare_model,
)


def is_wandb_available():
    """Check if wandb is installed and can be imported."""
    try:
        import wandb

        return True
    except ImportError:
        return False


def compute_auxiliary_losses(
    outputs: tuple,
    aux_outputs: tuple,
    labels: torch.Tensor,
    attr_labels: torch.Tensor,
    attr_criterion: List[nn.Module],
    args: Namespace,
    cross_entropy: nn.Module,
) -> List[torch.Tensor]:
    """Compute losses when using auxiliary outputs."""
    losses = []
    out_start = 0

    if args.mode in ["XY", "XCY"]:
        # call this when XY or XCY
        loss_class = 1.0 * cross_entropy(outputs[0], labels) + 0.4 * cross_entropy(
            aux_outputs[0], labels
        )
        losses.append(loss_class)
        out_start = 1

    if attr_criterion is not None and args.attr_loss_weight > 0:
        for i, crit in enumerate(attr_criterion):
            losses.append(
                args.attr_loss_weight
                * (
                    1.0 * crit(outputs[i + out_start].squeeze(), attr_labels[:, i])
                    + 0.4
                    * crit(aux_outputs[i + out_start].squeeze(), attr_labels[:, i])
                )
            )
    return losses


def compute_standard_losses(
    outputs: tuple,
    labels: torch.Tensor,
    attr_labels: torch.Tensor,
    criterion: nn.Module,
    attr_criterion: List[nn.Module],
    args: Namespace,
) -> List[torch.Tensor]:
    """Compute losses without auxiliary outputs."""
    losses = []
    out_start = 0

    if args.mode in ["XY", "XCY"]:
        losses.append(criterion(outputs[0], labels))
        out_start = 1

    if attr_criterion is not None and args.attr_loss_weight > 0:
        for i, attr_crit in enumerate(attr_criterion):
            losses.append(
                args.attr_loss_weight
                * attr_crit(outputs[i + out_start].squeeze(), attr_labels[:, i])
            )
    return losses


def run_epoch_simple(
    model: nn.Module,
    optimizer,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    criterion,
    args: Namespace,
    is_training: bool,
    tb_writer,
):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """

    raise NotImplementedError("TODO: implement run_epoch_simple for A->Y models")
    if is_training:
        model.train()
    else:
        model.eval()

    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _, data in enumerate(dataloader):
        inputs, labels = data
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t().float()

        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = inputs.to(device)
        labels_var = labels.to(device)

        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        loss_meter.update(loss.item(), inputs.size(0))

        class_acc_meter, attr_acc_meter = compute_accuracies(
            outputs, labels, epoch, class_acc_meter, attr_acc_meter, tb_writer
        )

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for i in range(loss_meter.n_losses):
        tb_writer.add_scalar(
            f"{'Train' if is_training else 'Val'}/{loss_labels[i]}",
            loss_meter.avg[i],
            epoch,
        )

    return loss_meter, class_acc_meter


def run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    criterion: nn.Module,
    attr_criterion: Optional[List[nn.Module]],
    args: Namespace,
    is_training: bool,
    tb_writer,
) -> tuple[LossMeter, AverageMeter, AverageMeter]:
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    loss_labels = []
    if args.mode == "XC":
        loss_labels = ["attr_loss"]
    else:
        if attr_criterion is not None:
            loss_labels = ["class_loss", "attr_loss"]
        else:
            loss_labels = ["class_loss"]

    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_entropy = nn.CrossEntropyLoss()

    for batch in tqdm(
        dataloader,
        desc=f"{'Train' if is_training else 'Val'} Epoch {epoch}",
        leave=False,
        mininterval=1.0,  # Update at most once per second
    ):
        inputs, labels, attr_labels = batch

                
        if attr_criterion is not None:
            attr_labels = torch.stack(attr_labels, dim=1).float()
            attr_labels_var = attr_labels.to(device)

        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs)
            losses = compute_auxiliary_losses(
                outputs,
                aux_outputs,
                labels,
                attr_labels_var,
                attr_criterion,
                args,
                cross_entropy,
            )

        # testing or no aux logits
        else:
            outputs = model(inputs)
            losses = compute_standard_losses(
                outputs, labels, attr_labels_var, criterion, attr_criterion, args
            )

        if not args.bottleneck:
            softmax_outputs = torch.nn.Softmax(dim=1)(outputs[0])
            class_acc = accuracy(softmax_outputs, labels, topk=(1,))
            class_acc_meter.update(class_acc[0], softmax_outputs.size(0))

        ## Accuracy calculation
        # Compute attribute accuracy (bottleneck always, otherwise when using images)
        if args.mode not in ["XY", "CY"]:
            pred_attr = outputs
            if args.mode != "XC":
                pred_attr = pred_attr[1:]

            # list of [N_ATTR, batch_size] tensors -> (B, N)
            reshaped_preds = torch.concat(pred_attr, dim=1).to(device)
            attr_acc, batch_size = compute_attr_accuracy(
                reshaped_preds, attr_labels_var
            )
            attr_acc_meter.update(attr_acc, batch_size)



        total_loss = None
        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = [(sum(losses) / args.n_attributes).detach()]
                back_loss = sum(losses) / args.n_attributes

            else:
                # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                attr_loss = sum(losses[1:])
                if args.normalize_loss:
                    attr_loss = attr_loss / (args.attr_loss_weight * args.n_attributes)

                back_loss = losses[0] + attr_loss
                
                total_loss = (back_loss.detach(), attr_loss.detach())
        else:
            # finetune
            back_loss = sum(losses)
            total_loss = [back_loss.detach()]

        loss_meter.update(np.array([x.cpu() for x in total_loss]), len(total_loss))

        if is_training:
            optimizer.zero_grad()
            back_loss.backward()
            optimizer.step()

    train_mode = "train" if is_training else "val"
    if args.mode != "XC":
        tb_writer.add_scalar(
            f"Class Accuracy/{train_mode}", class_acc_meter.avg.item(), epoch
        )

    if args.mode not in ["XY", "CY"]:
        tb_writer.add_scalar(
            f"Attribute Accuracy/{train_mode}", attr_acc_meter.avg.item(), epoch
        )

    return loss_meter, class_acc_meter, attr_acc_meter


def train(model: nn.Module, args: Namespace) -> float:
    """Train function for CUB models.

    Args:
        model (nn.Module): Model given a specific mode (XY, XCY, etc)
        args (Namespace): Config arguments

    Returns:
        float: best validation accuracy
    """
    model, device = prepare_model(model, args)
    logger, tb_writer = logger_and_summarywriter(args)

    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")

        # When arg multiple finds the imbalance ratio for each attr
        multiple_loss = args.weighted_loss == "multiple"
        imbalance = find_class_imbalance(train_data_path, multiple_loss)

    logger.write(str(args) + "\n")
    logger.write("Class Imbalances\n" + str(imbalance) + "\n")
    logger.flush()

    criterion = torch.nn.CrossEntropyLoss()

    attr_criterion = build_attr_criterion(args, imbalance, device)
    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

    loss_labels = []
    if args.bottleneck:
        loss_labels = ["attr_loss"]
    else:
        if attr_criterion is not None:
            loss_labels = ["class_loss", "attr_loss"]
        else:
            loss_labels = ["class_loss"]

    best_val_epoch, best_val_acc = -1, 0
    scheduler_stop_epoch = (
        int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    )

    # load data
    train_loader = load_data(args, "train")
    val_loader = load_data(args, "val")

    for epoch in range(0, args.epochs):
        start_time = time.time()

        # ----- Training -----
        if args.no_img:
            raise NotImplementedError("TODO")
            train_loss_meter, train_acc_meter = run_epoch_simple(
                model,
                optimizer,
                train_loader,
                epoch,
                criterion,
                args,
                is_training=True,
                tb_writer=tb_writer,
            )
        else:
            train_loss_meter, train_acc_meter, train_attr_acc_meter = run_epoch(
                model,
                optimizer,
                train_loader,
                epoch,
                criterion,
                attr_criterion,
                args,
                is_training=True,
                tb_writer=tb_writer,
            )

        # ----- Validation -----
        with torch.no_grad():
            if args.no_img:
                raise NotImplementedError("TODO")
                val_loss_meter, val_acc_meter = run_epoch_simple(
                    model,
                    optimizer,
                    val_loader,
                    epoch,
                    criterion,
                    args,
                    is_training=False,
                    tb_writer=tb_writer,
                )
            else:
                val_loss_meter, val_acc_meter, val_attr_acc_meter = run_epoch(
                    model,
                    optimizer,
                    val_loader,
                    epoch,
                    criterion,
                    attr_criterion,
                    args,
                    is_training=False,
                    tb_writer=tb_writer,
                )

        if (best_val_acc < val_acc_meter.avg and not args.bottleneck) or (
            best_val_acc < val_attr_acc_meter.avg and args.bottleneck
        ):
            best_val_epoch = epoch
            best_val_acc = (
                val_attr_acc_meter.avg if args.bottleneck else val_acc_meter.avg
            )

            logger.write("New model best model at epoch %d\n" % epoch)
            torch.save(
                model.state_dict(),
                os.path.join(args.log_dir, args.model_name, f"{model.name}_best_model_seed=%d.pth" % args.seed),
            )

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        log_dict = {}

        for key, value in zip(loss_labels, train_loss_avg):
            log_dict["Train/" + key] = value

        for key, value in zip(loss_labels, val_loss_avg):
            log_dict["Val/" + key] = value

        # Except XC always log class accuracy
        if args.mode != "XC":
            log_dict["Train/class_acc"] = f"{train_acc_meter.avg.item():.4f}"
            log_dict["Val/class_acc"] = f"{val_acc_meter.avg.item():.4f}"

        # do not log attribute accuracy
        if args.mode not in ["XY", "CY"]:
            log_dict["Train/attr_acc"] = f"{train_attr_acc_meter.avg.item():.4f}"
            log_dict["Val/attr_acc"] = f"{val_attr_acc_meter.avg.item():.4f}"

        if args.use_wandb:
            wandb.log(
                {
                    **log_dict,
                    "epoch": epoch,
                }
            )

        log_str = " - ".join([f"{key}: {value}" for key, value in log_dict.items()])

        time_duration = time.time() - start_time
        logger.write(
            " - ".join(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    log_str,
                    "\n",
                    f"Best val epoch: {best_val_epoch}",
                    f"Time: {time_duration:.2f} sec",
                ]
            )
            + "\n"
        )

        logger.flush()

        if epoch <= scheduler_stop_epoch:
            scheduler.step(epoch)

        if epoch % 10 == 0:
            print("Current lr:", scheduler.get_last_lr())

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

    # At the end of training:
    if args.use_wandb:
        wandb.finish()

    return best_val_acc


if __name__ == "__main__":
    print("Training CUB model")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cbm.yaml",
        help="Path to config file (YAML)",
    )
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = yaml.safe_load(f)
    args = normalize_scientific_floats(args)

    args = argparse.Namespace(**args, config_path=cli_args.config)
    args.model_name = nanoid.generate()


    # Initialize wandb if requested in config and available
    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb:
        if is_wandb_available():
            import wandb

            wandb.init(
                project=getattr(args, "wandb_project", "proto-CBM"),
                name=getattr(args, "wandb_run_name", None),
                config=vars(args),
            )
            print("Weights & Biases logging enabled")
        else:
            print("Warning: wandb requested but not installed. Skipping wandb logging.")
            args.use_wandb = False

    print("creating model...")
    model = model_by_mode(args)
    model.name = args.model_name
    print("Running model with name:", model.name)

    print("starting training...")
    train(model, args)
