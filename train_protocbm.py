import math
import os
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from torch import nn

from cub.config import BASE_DIR, LR_DECAY_SIZE, MIN_LR
from cub.dataset import load_data
from localization.part_seg_iou import create_mapping_attributes_to_part_seg_group
from losses import ProtoModLoss
from utils_protocbm.eval_utils import (
    eval_part_segmentation_iou,
    get_localization_loader,
)
from utils_protocbm.train_utils import (
    AverageMeter,
    LossMeter,
    compute_accuracies,
    create_criterions,
    gather_args,
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
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

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

        if model.training and args.use_aux:
            (outputs, similarity_scores, attention_maps), aux_outputs = model(inputs)

            # For X->C we do not train a classifier
            if model.classifier is None:
                classification_loss = torch.tensor(-1.0, device=device)
            else:
                classification_loss = cross_entropy(
                    outputs, labels
                ) + 0.4 * cross_entropy(aux_outputs, labels)
        else:
            (outputs, similarity_scores, attention_maps) = model(inputs)


            if model.classifier is None:
                classification_loss = torch.tensor(-1.0, device=device)
            else:
                classification_loss = cross_entropy(outputs, labels)

        losses.append(classification_loss)

        # Ignore protomod criterion if concepts were already provided
        if args.mode != "CY":
            loss, attribute_reg_loss, cpt_loss, decorrelation_loss = protomod_criterion(
                similarity_scores, attention_maps, attr_labels
            )
        else:
            attribute_reg_loss = torch.tensor(-1, device=device)
            cpt_loss = torch.tensor(-1, device=device)
            decorrelation_loss = torch.tensor(-1, device=device)

        losses.append(attribute_reg_loss)
        losses.append(cpt_loss)
        losses.append(decorrelation_loss)

        # Calculate attribute accuracy
        class_acc_meter, attr_acc_meter = compute_accuracies(
            outputs,
            similarity_scores,
            attr_labels,
            labels,
            epoch,
            class_acc_meter,
            attr_acc_meter,
            tb_writer,
        )

        total_loss = torch.stack(losses).sum()

        loss_meter.update(
            np.array([total_loss.item()] + [loss.item() for loss in losses]),
            inputs.size(0),
        )

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7.0)
            optimizer.step()

    for i in range(loss_meter.n_losses):
        tb_writer.add_scalar(
            f"{'Train' if is_training else 'Val'}/{loss_labels[i]}",
            loss_meter.avg[i],
            epoch,
        )

    # Using a joint criterion
    return (
        loss_meter,
        class_acc_meter,
        attr_acc_meter,
        -attr_acc_meter.avg - class_acc_meter.avg,
    )


def train(model: nn.Module, args: Namespace) -> float:
    model, device = prepare_model(model, args)

    logger, tb_writer = logger_and_summarywriter(args)

    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

    # TODO: Add checkpoints
    train_loader = load_data(args, "train")
    val_loader = load_data(args, "val")

    cross_entropy, protomod_criterion = create_criterions(model, args)

    best_val_epoch, best_val_metric = -1, math.inf
    best_val_acc = 0

    scheduler_stop_epoch = (
        int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    )
    for epoch in range(0, args.epochs):
        start_time = time.time()

        # ----- Training -----
        train_loss_meter, train_class_acc_meter, train_attr_acc_meter, _ = (
            epoch_wrapper(
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
        )

        # ----- Validation -----
        with torch.no_grad():
            val_loss_meter, val_class_acc_meter, val_attr_acc_meter, val_metric = (
                epoch_wrapper(
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
            )

        # NOTE: We are minimizing val_metric (ACCURACY NEEDS TO BE INVERTED)
        metric_criterion = val_metric < best_val_metric
        acc_criterion = val_class_acc_meter.avg > best_val_acc
        if metric_criterion or acc_criterion:
            best_val_epoch = epoch
            if metric_criterion:
                best_val_metric = val_metric
            if acc_criterion:
                best_val_acc = val_class_acc_meter.avg

            logger.write("New model best model at epoch %d" % epoch)
            torch.save(
                model.state_dict(),
                os.path.join(args.log_dir, f"best_model_{args.seed}.pth"),
            )

        logger_lst = [
            datetime.now().strftime("%H:%M:%S"),
            f"Epoch [{epoch}]",
            f"Train/loss: {[f'{type}: {loss:.4f}' for type, loss in zip(loss_labels, train_loss_meter.avg)]}",
            f"Train/Class acc: {train_class_acc_meter.avg.item():.4f}",
            f"Train/Attr acc: {train_attr_acc_meter.avg.item():.4f}",
            f"Val/loss: {[f'{type}: {loss:.4f}' for type, loss in zip(loss_labels, val_loss_meter.avg)]}",
            f"Val/Class acc: {val_class_acc_meter.avg.item():.4f}",
            f"Val/Attr acc: {val_attr_acc_meter.avg.item():.4f}",
            f"Best val epoch: {best_val_epoch}",
            f"Time: {time.time() - start_time:.2f} sec",
            f"LR: {scheduler.get_lr()[0]:.6f}",
        ]
        # if args.compute_localization:
        #    logger_lst.append(f"Val/PartSegLocalizationIoU: {part_seg_iou}")
        logger.write(" - ".join(logger_lst))
        logger.flush()

        if epoch <= scheduler_stop_epoch:
            scheduler.step()

        if epoch >= 100 and val_class_acc_meter.avg < 3:
            logger.write("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            logger.write("Early stopping because acc hasn't improved for a long time")
            break

    logger.close()

    # Compute the part segmentation IoU, globally across all groups / attributes.
    # TODO: Make data dirs uniform across all datasets / configs, hacky fix here
    tmp_data_dir = os.path.join(BASE_DIR, "/".join(args.image_dir.split("/")[:-1]))
    tmp_split_dir = os.path.join(BASE_DIR, args.data_dir, "val.pkl")

    # Get necessary data for localization, see eval.py or documentation for details
    loader, _, _, _ = get_localization_loader(model, tmp_data_dir, tmp_split_dir, args)
    map_attr_id_to_part_seg_group, attribute_names, unmatched_attr_mask = (
        create_mapping_attributes_to_part_seg_group(
            tmp_data_dir, device, only_cbm_attributes=True
        )
    )

    # Compute part segmentation IoU
    part_seg_iou = eval_part_segmentation_iou(
        model,
        loader,
        attribute_names,
        map_attr_id_to_part_seg_group,
        unmatched_attr_mask,
        args,
    )
    return val_metric, part_seg_iou


if __name__ == "__main__":
    args = gather_args()

    model = model_by_mode(args)

    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint))
        print("Continuing with checkpoint:", args.checkpoint)

    train(model, args)
