import math
import os
import time
from argparse import Namespace
from datetime import datetime

import matplotlib
import numpy as np
import torch
from torch import nn

from cub.config import BASE_DIR, LR_DECAY_SIZE, MIN_LR
from cub.dataset import load_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from localization.localization_accuracy import (
    calculate_average_partwise_localization_distance,
    compute_localization_distance,
)
from localization.visualise import visualize_keypoints_to_figure
from losses import LocalizationDistanceLoss, ProtoModLoss
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

torch.set_float32_matmul_precision("high")

# TODO
loss_labels = [
    "total_loss",
    "classification_loss",
    "attribute_reg_loss",
    "cpt_loss",
    "decorrelation_loss",
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

        if not is_training and any(
            req in args.val_metric for req in ["seg_iou", "dist_loc"]
        ):
            (
                inputs,
                labels,
                attr_labels,
                part_seg_masks,
                part_bbs,
                source_paths,
                part_gts,
            ) = batch
        elif dist_loss_active:
            # During training with distance loss, CUBKeypointDataset returns 4-tuple
            inputs, labels, attr_labels, part_gts = batch
        else:
            inputs, labels, attr_labels = batch

        # TODO: Move to collate_fn
        attr_labels = torch.stack(attr_labels, dim=1).float().to(device)

        losses = []

        if model.training and args.use_aux:
            base_outputs, aux_outputs = model(
                inputs, attr_labels
            )

            (outputs, similarity_scores, attention_maps) = base_outputs
            (out_aux, sim_scores_aux, attn_maps_aux) = aux_outputs

            # For X->C we do not train a classifier
            if args.mode == "XC":
                classification_loss = torch.tensor(0.0, device=device)
            else:
                classification_loss = cross_entropy(
                    outputs, labels
                ) + 0.4 * cross_entropy(out_aux, labels)
        else:
            (outputs, similarity_scores, attention_maps) = model(inputs, attr_labels)

            if args.mode == "XC":
                classification_loss = torch.tensor(0.0, device=device)
            else:
                classification_loss = cross_entropy(outputs, labels)

        losses.append(classification_loss)

        # Ignore protomod criterion if concepts were already provided
        if args.mode != "CY":
            _, attribute_reg_loss, cpt_loss, decorrelation_loss = protomod_criterion(
                similarity_scores, attention_maps, attr_labels
            )   
        
            if args.use_aux and model.training:

                _, aux_attribute_reg_loss, _, _ = protomod_criterion(
                    sim_scores_aux, attn_maps_aux, attr_labels, aux_forward=True
                )

                attribute_reg_loss = (1.0 * attribute_reg_loss) + (0.4 * aux_attribute_reg_loss)

        else:
            attribute_reg_loss = torch.tensor(0.0, device=device)
            cpt_loss = torch.tensor(0.0, device=device)
            decorrelation_loss = torch.tensor(0.0, device=device)

        losses.append(attribute_reg_loss)
        losses.append(cpt_loss)
        losses.append(decorrelation_loss)

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
            if "seg_loc" in args.val_metric:
                loc_meter.update(
                    eval_part_segmentation_iou(
                        attention_maps,
                        part_seg_masks,
                        dataloader.dataset.map_attr_id_to_part_seg_group,
                        dataloader.dataset.unmatched_attr_mask,
                    )
                )
            if "dist_loc" in args.val_metric:
                compute_localization_distance(
                    similarity_scores,
                    attention_maps,
                    part_bbs,
                    part_gts,
                    dataloader.dataset.part_dict,
                    dataloader.dataset.map_part_to_attr_loc_acc,
                    loc_acc_meter,
                    img_size=inputs.shape[-1],
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
            if val_metric_str == "class_acc":  # --- Class accuracy
                val_metric += class_acc_meter.avg.squeeze().cpu()
            if val_metric_str == "concept_acc":  # --- Concept accuracy
                val_metric += attr_acc_meter.avg.squeeze().cpu()
            if val_metric_str == "seg_iou":  # --- Segmentation IoU
                val_metric += loc_meter.compute(
                    dataloader.dataset.map_attr_id_to_part_seg_group
                )
            if val_metric_str == "dist_loc":  # --- Localization distance
                val_metric += 0.1 * torch.tensor(
                    -calculate_average_partwise_localization_distance(
                        loc_acc_meter, MAP_RESULT_GROUPS_TO_CUB_GROUPS, verbose=False
                    )[1]
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


def train(model: nn.Module, args: Namespace, close_console: bool = True) -> float:
    model, device = prepare_model(model, args)

    logger, tb_writer = logger_and_summarywriter(args, close_console=close_console)

    optimizer, scheduler = optimizer_and_scheduler_by_name(model, args)

    train_loader = load_data(args, "train")

    if any(req in args.val_metric for req in ["seg_iou", "dist_loc"]):
        # TODO: Make data dirs uniform across all datasets / configs, hacky fix here
        tmp_data_dir = os.path.join(BASE_DIR, args.image_dir)
        tmp_split_dir = os.path.join(BASE_DIR, args.data_dir, "val.pkl")

        # Get necessary data for localization, see eval.py or documentation for details
        val_loader, _, _, _ = get_localization_loader(
            model, tmp_data_dir, tmp_split_dir, args
        )
    else:
        val_loader = load_data(args, "val")

    cross_entropy, protomod_criterion = create_criterions(model, args)

    # Create localization distance loss criterion if needed
    localization_criterion = None
    if getattr(args, "distance_loss", False):
        # Get part dictionary and attribute mapping from the train loader dataset
        if hasattr(train_loader.dataset, "part_dict") and hasattr(
            train_loader.dataset, "map_part_to_attr_loc_acc"
        ):
            localization_criterion = LocalizationDistanceLoss(
                part_dict=train_loader.dataset.part_dict,
                part_attribute_mapping=train_loader.dataset.map_part_to_attr_loc_acc,
                img_size=train_loader.dataset.img_size,
                sigma=getattr(args, "distance_loss_sigma", 1.0),
                loss_type=getattr(args, "distance_loss_type", "mse"),
            ).to(device)
        else:
            raise ValueError(
                "distance_loss is enabled but train_loader.dataset does not have required attributes"
            )

    # ----- Sample fixed images for periodic keypoint visualization -----
    viz_every = getattr(args, "viz_every", 50)
    fixed_viz_data = None
    if getattr(args, "distance_loss", False):
        # Determine denormalization constants from backbone
        if args.backbone == "inception":
            viz_mean, viz_std = (0.5, 0.5, 0.5), (2, 2, 2)
        else:
            viz_mean, viz_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        # Grab first batch from train_loader, pick up to 4 images
        for batch in train_loader:
            batch = [v if torch.is_tensor(v) else v for v in batch]
            viz_inputs, _, viz_attr_labels, viz_part_gts = batch
            n_viz = min(4, viz_inputs.shape[0])
            # attr_labels comes as list of tensors from default collation
            if isinstance(viz_attr_labels, (list, tuple)):
                viz_attr_stacked = torch.stack(viz_attr_labels, dim=1).float()[:n_viz]
            else:
                viz_attr_stacked = viz_attr_labels[:n_viz].float()
            fixed_viz_data = {
                "imgs": viz_inputs[:n_viz].cpu(),
                "attr_labels": viz_attr_stacked.cpu(),
                "part_gts": viz_part_gts[:n_viz].cpu(),
            }
            break

        viz_dir = os.path.join(args.log_dir, args.model_name, "keypoint_viz")
        os.makedirs(viz_dir, exist_ok=True)
        logger.write(f"Keypoint visualization enabled every {viz_every} epochs ({n_viz} fixed images)")

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

        # ----- Periodic keypoint visualization -----
        if fixed_viz_data is not None and epoch % viz_every == 0:
            model.eval()
            with torch.no_grad():
                viz_imgs_gpu = fixed_viz_data["imgs"].to(device)
                viz_attr_gpu = fixed_viz_data["attr_labels"].to(device)
                (_, viz_sim_scores, viz_attn_maps) = model(viz_imgs_gpu, viz_attr_gpu)

                fig = visualize_keypoints_to_figure(
                    imgs=fixed_viz_data["imgs"],
                    part_gts=fixed_viz_data["part_gts"],
                    attention_maps=viz_attn_maps.cpu(),
                    similarity_scores=viz_sim_scores.cpu(),
                    part_dict=train_loader.dataset.part_dict,
                    part_attribute_mapping=train_loader.dataset.map_part_to_attr_loc_acc,
                    img_size=train_loader.dataset.img_size,
                    t_mean=viz_mean,
                    t_std=viz_std,
                )
                tb_writer.add_figure("Keypoint_Visualization", fig, epoch)
                fig.savefig(
                    os.path.join(viz_dir, f"keypoints_epoch_{epoch}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

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
            f"LR: {scheduler.get_last_lr()[0]:.6f}",
        ]
        logger.write(" - ".join(logger_lst))
        logger.flush()

        if epoch <= scheduler_stop_epoch:
            scheduler.step()
        if epoch - best_val_epoch >= 100:
            logger.write(
                "Early stopping because validation metric hasn't improved for a long time"
            )
            break

    logger.close()
    return best_val_metric, val_class_acc_meter.avg, val_attr_acc_meter.avg


if __name__ == "__main__":
    args = gather_args()

    model = model_by_mode(args)

    # For sequential training, first train the model with X->C, then C->Y
    if args.mode == "XC->CY":
        print("--- Training sequentially with modes 'XC' then 'CY' ---")
        args.mode = "XC"
        print("Training with mode 'XC' (predicting concepts from images)...")
        train(model, args, close_console=False)

        args.mode = "CY"
        print("Training with mode 'CY' (predicting classes from concepts)...")
        train(model, args)
    else:
        train(model, args)
