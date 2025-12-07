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

from cub.config import (
    BASE_DIR,
    LR_DECAY_SIZE,
    MIN_LR,
)

# from CUB import gen_cub_synthetic
from cub.dataset import find_class_imbalance, load_data
from utils.misc import Colors
from utils.train_utils import (
    AverageMeter,
    LossMeter,
    accuracy,
    compute_attr_accuracy,
    build_attr_criterion,
    compute_accuracies,
    logger_and_summarywriter,
    model_by_mode,
    normalize_scientific_floats,
    optimizer_and_scheduler_by_name,
    prepare_model,
)


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

    if not args.bottleneck:
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

    if not args.bottleneck:
        losses.append(criterion(outputs[0], labels))
        out_start = 1

    if attr_criterion is not None and args.attr_loss_weight > 0:
        for i, crit in enumerate(attr_criterion):
            losses.append(
                args.attr_loss_weight
                * crit(outputs[i + out_start].squeeze(), attr_labels[:, i])
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
    if args.bottleneck:
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
    ):
        inputs, labels, attr_labels = batch

        if attr_criterion is not None:
            attr_labels = torch.stack(attr_labels, dim=1).float()
            attr_labels_var = attr_labels.to(device)

        inputs, labels = (
            inputs.to(device),
            labels.to(device),
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
            if is_training:
                tb_writer.add_scalar("Class Accuracy/train", class_acc_meter.avg.item(), epoch)
            else:
                tb_writer.add_scalar("Class Accuracy/val", class_acc_meter.avg.item(), epoch)

        # Compute attribute accuracy (bottleneck always, otherwise when using images)
        if args.bottleneck or not args.no_img:

            pred_attr = outputs 
            if args.bottleneck:
                pred_attr = pred_attr[1:]

            # list of [N_ATTR, batch_size] tensors -> (B, N)
            reshaped_preds = torch.concat(pred_attr, dim=1).to(device)
            attr_acc, batch_size = compute_attr_accuracy(reshaped_preds, attr_labels_var)
            attr_acc_meter.update(attr_acc, batch_size)
            if is_training:
                tb_writer.add_scalar("Attribute Accuracy/train", attr_acc_meter.avg.item(), epoch)
            else:
                tb_writer.add_scalar("Attribute Accuracy/val", attr_acc_meter.avg.item(), epoch)

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
                total_loss = (losses[0].detach(), attr_loss.detach())
        else:
            # finetune
            back_loss = losses[0]
            total_loss = [losses[0].detach()]

        loss_meter.update(np.array([x.cpu() for x in total_loss]), len(total_loss))

        if is_training:
            optimizer.zero_grad()
            back_loss.backward()
            optimizer.step()

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

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg

            logger.write("New model best model at epoch %d\n" % epoch)
            torch.save(
                model, os.path.join(args.log_dir, "best_model_%d.pth" % args.seed)
            )

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        if isinstance(train_loss_avg, np.ndarray):
            train_loss_string = " - ".join(
                [
                    Colors.CYAN + f"(T){key}/loss: {value:.4f}" + Colors.ENDC
                    for (key, value) in zip(loss_labels, train_loss_avg)
                ]
            )
        else:
            train_loss_string = f"Train/loss: {train_loss_avg:.4f}"

        if isinstance(val_loss_avg, np.ndarray):
            val_loss_string = " - ".join(
                [
                    Colors.MAGENTA + f"(V){key}/loss: {value:.4f}" + Colors.ENDC
                    for (key, value) in zip(loss_labels, val_loss_avg)
                ]
            )
        else:
            val_loss_string = f"Val/loss: {val_loss_avg:.4f}"

        time_duration = time.time() - start_time
        logger.write(
            " - ".join(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    f"Epoch [{epoch}]",
                    train_loss_string,
                    Colors.GREEN + f"Train/acc: {train_acc_meter.avg.item():.4f}" + Colors.ENDC,
                    Colors.GREEN + f"Train/attr_acc: {train_attr_acc_meter.avg.item():.4f}" + Colors.ENDC,
                    val_loss_string,
                    Colors.YELLOW + f"Val/acc: {val_acc_meter.avg.item():.4f}" + Colors.ENDC,
                    Colors.YELLOW + f"Val/attr_acc: {val_attr_acc_meter.avg.item():.4f}" + Colors.ENDC,
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

    print("creating model...")
    model = model_by_mode(args)

    print("starting training...")
    train(model, args)
