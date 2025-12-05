"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import os
import sys
import time
from datetime import datetime

import yaml

from utils.train_utils import model_by_mode, normalize_scientific_floats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import os
from argparse import Namespace

import torch
from torch import nn

from cub.config import (
    BASE_DIR,
    LR_DECAY_SIZE,
    MIN_LR,
)

# from CUB import gen_cub_synthetic
from cub.dataset import find_class_imbalance, load_data
from utils.train_utils import (
    AverageMeter,
    LossMeter,
    compute_accuracies,
    logger_and_summarywriter,
    optimizer_and_scheduler_by_name,
    prepare_model,
)

# Define loss labels at module level
loss_labels = [
    "total_loss",
    "concept_loss",
]


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
    if is_training:
        model.train()
    else:
        model.eval()

    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()

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

        class_acc_meter = compute_accuracies(
            outputs, labels, epoch, class_acc_meter, tb_writer
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
    criterion,
    attr_criterion,
    args: Namespace,
    is_training: bool,
    tb_writer,
):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_entropy = nn.CrossEntropyLoss()

    for batch in dataloader:
        inputs, labels, attr_labels = batch

        if attr_criterion is not None:
            attr_labels = torch.stack(attr_labels, dim=1).float()
            attr_labels_var = attr_labels.to(device)

        inputs, labels = (
            inputs.to(device),
            labels.to(device),
        )

        losses = []

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs)
            out_start = 0

            # loss main is for the main task label (always the first output)
            if not args.bottleneck:
                loss_class = (
                    1.0 * cross_entropy(outputs[0], labels) + 
                    0.4 * cross_entropy(aux_outputs[0], labels)
                )
                losses.append(loss_class)
                out_start = 1

            # X -> A, cotraining, end2end
            if attr_criterion is not None and args.attr_loss_weight > 0:
                for i in range(len(attr_criterion)):
                    losses.append(
                        args.attr_loss_weight
                        * (
                            1.0
                            * attr_criterion[i](
                                outputs[i + out_start].squeeze(),
                                attr_labels_var[:, i],
                            )
                            + 0.4
                            * attr_criterion[i](
                                aux_outputs[i + out_start].squeeze(),
                                attr_labels_var[:, i],
                            )
                        )
                    )
        # testing or no aux logits
        else:
            outputs = model(inputs)
            losses = []
            out_start = 0

            if not args.bottleneck:
                loss_class = criterion(outputs[0], labels)
                losses.append(loss_class)
                out_start = 1

            # X -> A, cotraining, end2end
            if attr_criterion is not None and args.attr_loss_weight > 0:
                for i in range(len(attr_criterion)):
                    losses.append(
                        args.attr_loss_weight
                        * attr_criterion[i](
                            outputs[i + out_start].squeeze(),
                            attr_labels_var[:, i],
                        )
                    )

        if args.bottleneck:
            # computes the binary accuracy over all attributes
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            class_acc_meter, _ = compute_accuracies(
                sigmoid_outputs, labels, epoch, class_acc_meter, tb_writer
            )
        else:
            # only use the first output
            class_acc_meter = compute_accuracies(
                outputs[0], labels, epoch, class_acc_meter, tb_writer
            )

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses) / args.n_attributes
            else:
                # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (
                        1 + args.attr_loss_weight * args.n_attributes
                    )
        else:
            # finetune
            total_loss = sum(losses)

        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return loss_meter, class_acc_meter


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

    attr_criterion = None
    if args.use_attr and not args.no_img:
        attr_criterion = []
        if args.weighted_loss:
            assert imbalance is not None
            for ratio in imbalance:
                attr_criterion.append(
                    torch.nn.BCEWithLogitsLoss(
                        weight=torch.FloatTensor([ratio]).to(device)
                    )
                )
        else:
            # unweighed CE loss
            for _ in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())

    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

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
            train_loss_meter, train_acc_meter = run_epoch(
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
                val_loss_meter, val_acc_meter = run_epoch(
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

        time_duration = time.time() - start_time
        logger.write(
            " - ".join(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    f"Epoch [{epoch}]",
                    f"Train/loss: {train_loss_avg:.4f}",
                    f"Train/acc: {train_acc_meter.avg.item():.4f}",
                    f"Val/loss: {val_loss_avg:.4f}",
                    f"Val/acc: {val_acc_meter.avg.item():.4f}",
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
    
    print('creating model...')
    model = model_by_mode(args)

    print('starting training...')
    train(model, args)