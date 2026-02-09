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
from localization.localization_accuracy import (
    compute_localization_distance,
    calculate_average_partwise_localization_distance
)
from losses import ProtoModLoss, LocalizationDistanceLoss
from utils_protocbm.eval_utils import (
    LocalizationMeter,
    eval_part_segmentation_iou,
    get_localization_loader,
)
from utils_protocbm.mappings import MAP_RESULT_GROUPS_TO_CUB_GROUPS
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
    "consistency_loss",
    "localization_distance_loss",
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
    localization_criterion: LocalizationDistanceLoss = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    # get arg for distance_loss, distance_loss_weight
    dist_loss_active = getattr(args, "distance_loss", False)
    dist_loss_weight = getattr(args, "distance_loss_weight", 1.0)


    if is_training:
        model.train()
    else:
        model.eval()
        if any(req in args.val_metric for req in ["seg_iou", "dist_loc"]):
            loc_meter = LocalizationMeter(dataloader.dataset.attribute_names, device)
            loc_acc_meter = []

    for batch in dataloader:

        batch = [v.to(device) if torch.is_tensor(v) else v for v in batch]

        if not is_training and any(req in args.val_metric for req in ["seg_iou", "dist_loc"]):
            inputs, labels, attr_labels, part_seg_masks, part_bbs, source_paths, part_gts = batch
        elif dist_loss_active:
            # During training with distance loss, CUBKeypointDataset returns 4-tuple
            inputs, labels, attr_labels, part_gts = batch
        else:
            inputs, labels, attr_labels = batch

        # TODO: Move to collate_fn
        attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

        losses = []

        if model.training and args.use_aux:
            (outputs, similarity_scores, attention_maps), aux_outputs = model(inputs, attr_labels)

            # For X->C we do not train a classifier
            if model.classifier is None:
                classification_loss = torch.tensor(-1.0, device=device)
            else:
                classification_loss = cross_entropy(
                    outputs, labels
                ) + 0.4 * cross_entropy(aux_outputs, labels)
        else:
            (outputs, similarity_scores, attention_maps) = model(inputs, attr_labels)

            if model.classifier is None:
                classification_loss = torch.tensor(-1.0, device=device)
            else:
                classification_loss = cross_entropy(outputs, labels)

        losses.append(classification_loss)

        # Ignore protomod criterion if concepts were already provided
        if args.mode != "CY":
            loss, attribute_reg_loss, cpt_loss, decorrelation_loss, consistency_loss = protomod_criterion(
                similarity_scores, attention_maps, attr_labels
            )
        else:
            attribute_reg_loss = torch.tensor(-1, device=device)
            cpt_loss = torch.tensor(-1, device=device)
            decorrelation_loss = torch.tensor(-1, device=device)
            consistency_loss = torch.tensor(-1, device=device)



        losses.append(attribute_reg_loss)
        losses.append(cpt_loss)
        losses.append(decorrelation_loss)
        losses.append(consistency_loss)

        # Compute localization distance loss if active
        if dist_loss_active and localization_criterion is not None:
            localization_distance_loss = dist_loss_weight * localization_criterion(
                attention_maps, part_gts
            )
        else:
            localization_distance_loss = torch.tensor(0.0, device=device)

        losses.append(localization_distance_loss)

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
        else:
            # Compute metrics based on provided string(s)
            if 'seg_loc' in args.val_metric:
                loc_meter.update(
                    eval_part_segmentation_iou(
                        attention_maps,
                        part_seg_masks,
                        dataloader.dataset.map_attr_id_to_part_seg_group,
                        dataloader.dataset.unmatched_attr_mask
                    )
                )
            if 'dist_loc' in args.val_metric: 
                compute_localization_distance(
                    similarity_scores, attention_maps, part_bbs, part_gts, dataloader.dataset.part_dict,
                    dataloader.dataset.map_part_to_attr_loc_acc, loc_acc_meter, img_size=inputs.shape[-1]
                )

    for i in range(loss_meter.n_losses):
        tb_writer.add_scalar(
            f"{'Train' if is_training else 'Val'}/{loss_labels[i]}",
            loss_meter.avg[i],
            epoch,
        )
    
    # Metric(s) used to evaluate overall quality of model during validation
    # IMPORTANT: Must be maximized! So if any metric is minimize->better, invert it.
    val_metric = torch.tensor(0.0)
    if not is_training:
        for val_metric_str in args.val_metric:
            if val_metric_str == "class_acc":      # --- Class accuracy
                val_metric += class_acc_meter.avg.squeeze().cpu()
            if val_metric_str == "concept_acc":  # --- Concept accuracy
                val_metric += attr_acc_meter.avg.squeeze().cpu()
            if val_metric_str == "seg_iou":  # --- Segmentation IoU
                val_metric += loc_meter.compute(dataloader.dataset.map_attr_id_to_part_seg_group)
            if val_metric_str == "dist_loc": # --- Localization distance
                val_metric += torch.tensor(
                    -calculate_average_partwise_localization_distance(loc_acc_meter, MAP_RESULT_GROUPS_TO_CUB_GROUPS, verbose=False)[1]
                )
    
        tb_writer.add_scalar(
            "Val/Metric",
            val_metric,
            epoch,
        )
    
    return (
        loss_meter,
        class_acc_meter,
        attr_acc_meter,
        val_metric,
    )


def train(model: nn.Module, args: Namespace) -> float:
    model, device = prepare_model(model, args)

    logger, tb_writer = logger_and_summarywriter(args)

    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

    train_loader = load_data(args, "train")
    
    if any(req in args.val_metric for req in ["seg_iou", "dist_loc"]):
        # TODO: Make data dirs uniform across all datasets / configs, hacky fix here
        tmp_data_dir = os.path.join(BASE_DIR, args.image_dir)
        tmp_split_dir = os.path.join(BASE_DIR, args.data_dir, "val.pkl")

        # Get necessary data for localization, see eval.py or documentation for details
        val_loader, _, _, _ = get_localization_loader(model, tmp_data_dir, tmp_split_dir, args)
    else:
        val_loader = load_data(args, "val")

    cross_entropy, protomod_criterion = create_criterions(model, args)

    # Create localization distance loss criterion if needed
    localization_criterion = None
    if getattr(args, "distance_loss", False):
        # Get part dictionary and attribute mapping from the train loader dataset
        if hasattr(train_loader.dataset, 'part_dict') and hasattr(train_loader.dataset, 'map_part_to_attr_loc_acc'):
            localization_criterion = LocalizationDistanceLoss(
                part_dict=train_loader.dataset.part_dict,
                part_attribute_mapping=train_loader.dataset.map_part_to_attr_loc_acc,
                img_size=train_loader.dataset.img_size
            ).to(device)
        else:
            raise ValueError("distance_loss is enabled but train_loader.dataset does not have required attributes")

    best_val_epoch, best_val_metric = -1, -math.inf

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
                localization_criterion=localization_criterion,
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
                    localization_criterion=localization_criterion,
                )
            )

        # We are maximizing metric_criterion!
        if val_metric > best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric

            logger.write("New model best model at epoch %d" % epoch)

            # Save model
            if getattr(args, "save_model", True):
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
            f"Val/Metric: {val_metric.item():.4f}",
            f"Best val epoch: {best_val_epoch}",
            f"Time: {time.time() - start_time:.2f} sec",
            f"LR: {scheduler.get_lr()[0]:.6f}",
        ]
        logger.write(" - ".join(logger_lst))
        logger.flush()

        if epoch <= scheduler_stop_epoch:
            scheduler.step()
        if epoch - best_val_epoch >= 100:
            logger.write("Early stopping because validation metric hasn't improved for a long time")
            break

    logger.close()
    return best_val_metric, val_class_acc_meter.avg, val_attr_acc_meter.avg


if __name__ == "__main__":
    args = gather_args()

    model = model_by_mode(args)

    train(model, args)
