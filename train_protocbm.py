import argparse
import math
import os
import time
from argparse import Namespace
from datetime import datetime

from numpy import cross
import torch
import yaml
from analysis import AverageMeter, LossMeter
from torch import nn

from cub.config import LR_DECAY_SIZE, MIN_LR
from cub.dataset import load_data
from losses import ProtoModLoss
from utils.train_utils import (
    create_criterions,
    logger_and_summarywriter,
    model_by_mode,
    optimizer_and_scheduler_by_name,
    prepare_model,
)

# TODO
loss_labels = [
    "total_loss",
    "classification_loss",
    "attribute_reg_loss",
    "cpt_loss",
    "decorrelation_loss",
]


def epoch_wrapper(
    model,
    optimizer,
    dataloader,
    epoch: int,
    is_training: bool,
    args: Namespace,
    tb_writer,
    cross_entropy: nn.CrossEntropyLoss,
    protomod_criterion: ProtoModLoss,
):
    loss_meter = LossMeter(loss_labels)
    acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in dataloader:
        inputs, labels, attr_labels = batch

        # TODO: Move to collate_fn
        attr_labels = torch.stack(attr_labels, dim=1).float()

        inputs, labels, attr_labels = (
            inputs.to(device),
            labels.to(device),
            attr_labels.to(device),
        )

        losses = []
        log_losses = []

        if args.use_aux:
            outputs, similarity_scores, attention_maps, aux_outputs = model(inputs)

            classification_loss = cross_entropy(outputs, labels) + 0.4 * cross_entropy(aux_outputs, labels)
        else:
            outputs, similarity_scores, attention_maps = model(inputs)

            classification_loss = cross_entropy(outputs, labels)


        losses.append(classification_loss)

        loss, attribute_reg_loss, cpt_loss, decorrelation_loss = protomod_criterion(
                similarity_scores, attention_maps, attr_labels
            )
        
        losses.append(attribute_reg_loss)
        losses.append(cpt_loss)
        losses.append(decorrelation_loss)

        if args.bottleneck:  # attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))

        else:
            acc = accuracy(
                outputs[0], labels, topk=(1,)
            )  # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses) / args.n_attributes
            else:  # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (
                        1 + args.attr_loss_weight * args.n_attributes
                    )
        else:  # finetune
            total_loss = sum(losses)

        loss_meter.update(
            np.array(
                [
                    total_loss.item(),
                    log_losses[0],
                    log_losses[1],
                    log_losses[2],
                    log_losses[3],
                ]
            ),
            inputs.size(0),
        )

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    for i in range(loss_meter.n_losses):
        tb_writer.add_scalar(f"Train/{loss_labels[i]}", loss_meter.avg[i], epoch)
    tb_writer.add_scalar("Accuracy/train", acc_meter.avg.item(), epoch)

    return loss_meter, acc_meter

def train(model: nn.Module, args: Namespace) -> float:
    model = prepare_model(model)

    logger, tb_writer = logger_and_summarywriter(args)

    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

    # TODO: Add checkpoints
    # TODO: Distinguish mode
    train_loader = load_data(
        args.data_dir,
        "train",
        args.use_attr,
        args.no_img,
        args.batch_size,
        args.uncertain_labels,
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr,
        resampling=args.resampling,
    )
    val_loader = load_data(
        args.data_dir,
        "val",
        args.use_attr,
        args.no_img,
        args.batch_size,
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr,
    )

    cross_entropy, protomod_criterion = create_criterions(model, args)

    best_val_epoch, best_val_acc = -1, 0

    scheduler_stop_epoch = (
        int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    )
    for epoch in range(0, args.epochs):
        start_time = time.time()

        # ----- Training -----
        train_loss_meter, train_acc_meter = epoch_wrapper(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            epoch=epoch,
            is_training=True,
            args=args,
            tb_writer=tb_writer,
            cross_entropy=cross_entropy,
            protomod_criterion=protomod_criterion,
        )

        # ----- Validation -----
        with torch.no_grad():
            val_loss_meter, val_acc_meter = epoch_wrapper(
                model=model,
                optimizer=optimizer,
                dataloader=val_loader,
                epoch=epoch,
                is_training=False,
                args=args,
                tb_writer=tb_writer,
                cross_entropy=cross_entropy,
                protomod_criterion=protomod_criterion,
            )

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg

            logger.write("New model best model at epoch %d\n" % epoch)
            torch.save(
                model, os.path.join(args.log_dir, "best_model_%d.pth" % args.seed)
            )

        logger.write(
            " - ".join(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    f"Epoch [{epoch}]",
                    f"Train/loss: {train_loss_meter.avg:.4f}",
                    f"Train/acc: {train_acc_meter.avg.item():.4f}",
                    f"Val/loss: {val_loss_meter.avg:.4f}",
                    f"Val/acc: {val_acc_meter.avg.item():.4f}"
                    f"Best val epoch: {best_val_epoch}",
                    f"Time: {time.time() - start_time:.2f} sec",
                    f"LR: {scheduler.get_lr()[0]:.6f}",
                ]
            )
        )

        logger.flush()

        if epoch <= scheduler_stop_epoch:
            scheduler.step(epoch)

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

    return best_val_acc

if __name__ == "__main__":
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

    # Add run name, keep as namespace to be able to access like args.param
    args = Namespace(**args, run_name=cli_args.run_name, config_path=cli_args.config)

    model = model_by_mode(args)

    train(model, args)
