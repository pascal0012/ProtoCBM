"""Training entry for SEG-MIL-CBM on CUB.

Usage:
    python train_seg_mil_cbm.py --config configs/seg_mil_cbm/seg_mil_cbm.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from cub.seg_mil_dataset import CUBSegMILDataset
from losses import SegMILLoss
from models.backbones import ResNet50, DINO
from models.seg_mil_cbm import SegMILCBM


def build_backbone(name: str, pretrained: bool, freeze: bool):
    if name == "resnet50":
        bb = ResNet50(pretrained=pretrained, freeze=freeze, input_img_size=224)
        return bb, bb.final_channel_dim
    if name in {"dino_vitb16", "dino_vits8", "dino_vitb8"}:
        bb = DINO(pretrained=True, freeze=freeze, input_img_size=224, backbone=name)
        return bb, bb.final_channel_dim
    raise ValueError(f"Unsupported backbone {name}")


def set_backbone_grad(model: SegMILCBM, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    # set train or eval
    model.train(train)

    total = correct = 0
    sum_loss = sum_cls = sum_concept = 0.0
    n_batches = 0
    for crops, valid, clip_concepts, label in loader:
        # crop and check if valid
        crops = crops.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        
        # load concept scores and labels
        clip_concepts = clip_concepts.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, seg_concepts, _alpha = model(crops, valid)
            loss, parts = criterion(logits, seg_concepts, clip_concepts, label, valid)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()
        sum_cls += parts["cls"].item()
        sum_concept += parts["concept"].item()
        n_batches += 1

        # compute accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    return {
        "loss": sum_loss / max(n_batches, 1),
        "cls": sum_cls / max(n_batches, 1),
        "concept": sum_concept / max(n_batches, 1),
        "acc": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("log_dir", "outputs/SEG-MIL-CBM"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CUBSegMILDataset(
        [os.path.join(cfg["pkl_dir"], "train.pkl")],
        cache_dir=os.path.join(cfg["cache_dir"], "train"),
    )
    val_ds = CUBSegMILDataset(
        [os.path.join(cfg["pkl_dir"], "val.pkl")],
        cache_dir=os.path.join(cfg["cache_dir"], "val"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    backbone, feat_dim = build_backbone(
        cfg["backbone"],
        pretrained=cfg.get("backbone_pretrained", True),
        freeze=False,  # we manage freezing manually via warm-up schedule
    )
    model = SegMILCBM(
        backbone=backbone,
        feature_dim=feat_dim,
        concept_dim=cfg["concept_dim"],
        num_classes=cfg.get("num_classes", 200),
        attn_hidden=cfg.get("attn_hidden", 128),
        attn_temperature=cfg.get("attn_temperature", 1.0),
    ).to(device)

    criterion = SegMILLoss(lambda_concept=cfg.get("lambda_concept", 0.1)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    epochs = cfg["epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 5)
    best_acc = 0.0
    log_path = out_dir / "log.txt"
    log_f = open(log_path, "a", buffering=1)

    for epoch in range(epochs):
        # Warm-up: backbone trainable. After warm-up: freeze backbone and
        # rebuild optimizer over the remaining trainable params.
        if epoch == warmup_epochs:
            set_backbone_grad(model, False)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.get("lr", 1e-4),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
            log_f.write(f"[epoch {epoch}] froze backbone\n")

        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0
        msg = (
            f"epoch {epoch:03d}  ({dt:.1f}s)  "
            f"train loss={tr['loss']:.4f} cls={tr['cls']:.4f} con={tr['concept']:.4f} acc={tr['acc']:.4f}  "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f}"
        )
        print(msg)
        log_f.write(msg + "\n")

        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc},
                out_dir / "best.pth",
            )

    log_f.write(f"best val acc: {best_acc:.4f}\n")
    log_f.close()
    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
