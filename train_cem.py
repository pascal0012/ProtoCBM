"""
Training script for the Concept Embedding Model (CEM).

CEM trains jointly end-to-end with:
  - Weighted BCE loss on concept probabilities (to handle class imbalance)
  - CrossEntropy loss on class predictions
  - Random concept interventions during training (default 25%)

Paper hyperparameters (CUB):
  - concept_loss_weight (alpha) = 5
  - SGD optimizer, momentum=0.9, lr=0.01, weight_decay=4e-5
  - ReduceLROnPlateau: factor=0.1, patience=10 on val loss
  - Early stopping: patience=15 on val loss
  - batch_size=128, epochs=300
"""

import os
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import nanoid

from cub.config import BASE_DIR
from cub.dataset import find_class_imbalance, load_data
from models.cem import build_cem
from utils_protocbm.train_utils import (
    AverageMeter,
    LossMeter,
    accuracy,
    binary_accuracy,
    gather_args,
    logger_and_summarywriter,
    prepare_model,
)


def is_wandb_available():
    try:
        import wandb
        return True
    except ImportError:
        return False


def build_concept_loss(args, device):
    """Build per-attribute weighted BCE loss to handle concept label imbalance."""
    use_weighted = getattr(args, "weighted_loss", "multiple")
    if use_weighted:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
        imbalance = find_class_imbalance(train_data_path, multiple_attr=True)
        # imbalance[i] = (n_total / n_positive) - 1, i.e. neg/pos ratio
        # BCEWithLogitsLoss pos_weight = neg_count / pos_count
        pos_weights = torch.tensor(imbalance, dtype=torch.float32, device=device)
        print(f"Using weighted BCE with pos_weight range: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        return nn.BCEWithLogitsLoss()


def run_epoch_cem(
    model,
    optimizer,
    dataloader,
    epoch,
    args,
    is_training,
    tb_writer,
    concept_loss_fn,
    class_loss_fn,
    device,
):
    loss_labels = ["total_loss", "concept_loss", "class_loss"]
    loss_meter = LossMeter(loss_labels)
    class_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    concept_loss_weight = getattr(args, "concept_loss_weight", 5.0)
    task_loss_weight = getattr(args, "task_loss_weight", 1.0)

    for batch in tqdm(
        dataloader,
        desc=f"{'Train' if is_training else 'Val'} Epoch {epoch}",
        leave=False,
        mininterval=1.0,
    ):
        inputs, labels, attr_labels = batch

        attr_labels = torch.stack(attr_labels, dim=1).float()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attr_labels = attr_labels.to(device, non_blocking=True)

        # Forward pass
        class_logits, concept_logits, _ = model(inputs, attr_labels)

        # Losses
        c_loss = concept_loss_fn(concept_logits, attr_labels)
        t_loss = class_loss_fn(class_logits, labels)
        total_loss = concept_loss_weight * c_loss + task_loss_weight * t_loss

        # Logging
        loss_meter.update(
            np.array([total_loss.item(), c_loss.item(), t_loss.item()]),
            inputs.size(0),
        )

        # Accuracy
        class_acc = accuracy(class_logits, labels, topk=(1,))
        class_acc_meter.update(class_acc[0], class_logits.size(0))

        attr_acc = binary_accuracy(concept_logits, attr_labels)
        attr_acc_meter.update(attr_acc, inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    train_mode = "train" if is_training else "val"
    tb_writer.add_scalar(f"Class Accuracy/{train_mode}", class_acc_meter.avg, epoch)
    tb_writer.add_scalar(f"Attribute Accuracy/{train_mode}", attr_acc_meter.avg, epoch)

    return loss_meter, class_acc_meter, attr_acc_meter


def train_cem(model, args):
    model, device = prepare_model(model, args)
    logger, tb_writer = logger_and_summarywriter(args)

    logger.write(str(args) + "\n")
    logger.flush()

    # Build loss functions
    concept_loss_fn = build_concept_loss(args, device)
    class_loss_fn = nn.CrossEntropyLoss()

    # SGD with momentum (paper default)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    # ReduceLROnPlateau: reduce lr by 0.1 if val loss doesn't improve for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=getattr(args, "lr_patience", 10),
    )

    train_loader = load_data(args, ["train", "val"])
    val_loader = load_data(args, "val")

    best_val_epoch = -1
    best_val_loss = float("inf")

    # Early stopping patience on val loss
    early_stop_patience = getattr(args, "early_stop_patience", 15)

    loss_labels = ["total_loss", "concept_loss", "class_loss"]

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss_meter, train_acc_meter, train_attr_acc_meter = run_epoch_cem(
            model, optimizer, train_loader, epoch, args,
            is_training=True, tb_writer=tb_writer,
            concept_loss_fn=concept_loss_fn, class_loss_fn=class_loss_fn,
            device=device,
        )

        with torch.no_grad():
            val_loss_meter, val_acc_meter, val_attr_acc_meter = run_epoch_cem(
                model, optimizer, val_loader, epoch, args,
                is_training=False, tb_writer=tb_writer,
                concept_loss_fn=concept_loss_fn, class_loss_fn=class_loss_fn,
                device=device,
            )

        # Monitor val total loss for LR scheduling and early stopping
        val_total_loss = val_loss_meter.avg[0]
        scheduler.step(val_total_loss)

        # Track best model by val loss
        if val_total_loss < best_val_loss:
            best_val_epoch = epoch
            best_val_loss = val_total_loss
            logger.write(f"New best model at epoch {epoch} (val_loss={val_total_loss:.4f})\n")
            torch.save(
                model.state_dict(),
                os.path.join(args.log_dir, args.model_name, f"best_model_{args.seed}.pth"),
            )

        # Logging
        log_dict = {}
        for key, value in zip(loss_labels, train_loss_meter.avg):
            log_dict["Train/" + key] = f"{value:.4f}"
        for key, value in zip(loss_labels, val_loss_meter.avg):
            log_dict["Val/" + key] = f"{value:.4f}"

        train_class_acc = train_acc_meter.avg.item() if torch.is_tensor(train_acc_meter.avg) else train_acc_meter.avg
        val_class_acc = val_acc_meter.avg.item() if torch.is_tensor(val_acc_meter.avg) else val_acc_meter.avg
        log_dict["Train/class_acc"] = f"{train_class_acc:.4f}"
        log_dict["Val/class_acc"] = f"{val_class_acc:.4f}"
        log_dict["Train/attr_acc"] = f"{train_attr_acc_meter.avg:.4f}"
        log_dict["Val/attr_acc"] = f"{val_attr_acc_meter.avg:.4f}"
        log_dict["lr"] = f"{optimizer.param_groups[0]['lr']:.6f}"

        if getattr(args, "use_wandb", False):
            wandb.log({**log_dict, "epoch": epoch})

        log_str = " - ".join([f"{key}: {value}" for key, value in log_dict.items()])
        time_duration = time.time() - start_time
        logger.write(
            " - ".join([
                datetime.now().strftime("%H:%M:%S"),
                log_str, "\n",
                f"Best val epoch: {best_val_epoch}",
                f"Time: {time_duration:.2f} sec",
            ]) + "\n"
        )
        logger.flush()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}, val_loss: {val_total_loss:.4f}")

        # Early stopping on val loss
        if epoch - best_val_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val loss improvement for {early_stop_patience} epochs)")
            break

    if getattr(args, "use_wandb", False):
        wandb.finish()

    return best_val_loss


if __name__ == "__main__":
    print("Training CEM model")
    args = gather_args()
    args.model_name = nanoid.generate()

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
            print("Warning: wandb requested but not installed. Skipping.")
            args.use_wandb = False

    print("Creating CEM model...")
    model = build_cem(args)
    model.name = args.model_name
    print(f"Running model with name: {model.name}")

    # Create log directory
    os.makedirs(os.path.join(args.log_dir, args.model_name), exist_ok=True)

    print("Starting training...")
    train_cem(model, args)
