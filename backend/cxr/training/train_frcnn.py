"""Fine-tune Faster R-CNN on object-CXR.

Equivalent to notebook cells 11-14, but config driven, device agnostic, and it
writes a run manifest next to the checkpoint so a served model can be traced
back to the run that produced it.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cxr.config import Settings
from cxr.data.dataset import ForeignObjectDataset, collate_fn, labels_to_dict, load_labels
from cxr.data.transforms import build_transforms
from cxr.device import device_description, resolve_device
from cxr.logging_conf import get_logger
from cxr.models.faster_rcnn import build_faster_rcnn
from cxr.training.engine import evaluate_classification, train_one_epoch

logger = get_logger(__name__)


@dataclass
class TrainResult:
    checkpoint: Path
    best_auc: float
    best_accuracy: float
    best_epoch: int
    history: list[dict]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(settings: Settings, limit: int | None = None):
    data = settings.data
    if not data.is_available():
        raise FileNotFoundError(
            f"dataset not found under {data.root}. Expected {data.train_csv}, {data.dev_csv} "
            f"and the {data.train_dir}/ and {data.dev_dir}/ image folders."
        )

    transform = build_transforms(data.image_size)
    train_labels = labels_to_dict(
        load_labels(data.train_labels, positives_only=data.train_on_positives_only)
    )
    dev_labels = labels_to_dict(load_labels(data.dev_labels))

    train_set = ForeignObjectDataset(
        data.train_images, train_labels, mode="train", transform=transform,
        image_size=data.image_size,
    )
    dev_set = ForeignObjectDataset(
        data.dev_images, dev_labels, mode="dev", transform=transform,
        image_size=data.image_size,
    )
    if limit:
        train_set = Subset(train_set, range(min(limit, len(train_set))))
        dev_set = Subset(dev_set, range(min(limit, len(dev_set))))

    train_loader = DataLoader(
        train_set, batch_size=data.batch_size, shuffle=True,
        num_workers=data.num_workers, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=1, shuffle=False,
        num_workers=data.num_workers, collate_fn=collate_fn,
    )
    return train_loader, dev_loader


def train(settings: Settings, epochs: int | None = None, limit: int | None = None) -> TrainResult:
    cfg = settings.training
    set_seed(cfg.seed)
    device = resolve_device(settings.device)
    epochs = epochs or cfg.epochs
    logger.info("training Faster R-CNN on %s for %d epoch(s)", device_description(device), epochs)

    train_loader, dev_loader = build_dataloaders(settings, limit=limit)
    logger.info("train batches: %d | dev images: %d", len(train_loader), len(dev_loader.dataset))

    model = build_faster_rcnn(pretrained_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )

    checkpoint = Path(settings.frcnn.weights)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_auc, best_acc, best_epoch = -1.0, 0.0, -1
    history: list[dict] = []
    started = time.time()

    for epoch in range(epochs):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=cfg.print_freq
        )
        lr_scheduler.step()

        val = evaluate_classification(
            model, dev_loader, device, threshold=settings.frcnn.conf_threshold
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val["accuracy"],
            "val_auc": val["auc"],
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(entry)
        logger.info(
            "epoch %d | loss %.4f | val acc %.4f | val auc %.4f",
            epoch, train_loss, val["accuracy"], val["auc"],
        )

        if val["auc"] > best_auc:
            best_auc, best_acc, best_epoch = val["auc"], val["accuracy"], epoch
            torch.save(model.state_dict(), checkpoint)
            logger.info("new best AUC %.4f - checkpoint saved to %s", best_auc, checkpoint)

    manifest = {
        "model": "frcnn",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_accuracy": best_acc,
        "total_minutes": round((time.time() - started) / 60, 2),
        "image_size": settings.data.image_size,
        "train_on_positives_only": settings.data.train_on_positives_only,
        "hyperparameters": {
            "lr": cfg.lr, "momentum": cfg.momentum, "weight_decay": cfg.weight_decay,
            "batch_size": settings.data.batch_size, "lr_step_size": cfg.lr_step_size,
            "lr_gamma": cfg.lr_gamma, "seed": cfg.seed,
        },
        "history": history,
    }
    (checkpoint.parent / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainResult(checkpoint, best_auc, best_acc, best_epoch, history)
