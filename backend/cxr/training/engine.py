"""Train / validate loops for the torchvision detector."""

from __future__ import annotations

import math

import torch
from sklearn.metrics import roc_auc_score

from cxr.logging_conf import get_logger
from cxr.training.metrics import MetricLogger, SmoothedValue

logger = get_logger(__name__)


class TrainingDivergedError(RuntimeError):
    """Raised instead of ``sys.exit(1)`` when the loss goes non-finite."""


def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float):
    def f(step: int) -> float:
        if step >= warmup_iters:
            return 1.0
        alpha = step / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, print_freq: int = 50):
    """One pass over the training set. Returns the epoch's mean loss."""

    model.train()
    metric_logger = MetricLogger()
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, len(data_loader) - 1)
        if warmup_iters > 0:
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, 1.0 / 1000)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logger.error("loss is %s, stopping training: %s", loss_value, loss_dict)
            raise TrainingDivergedError(f"loss became {loss_value}")

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger.meters["loss"].global_avg


@torch.inference_mode()
def evaluate_classification(model, data_loader, device, threshold: float = 0.5):
    """Image level accuracy/AUC on the dev split.

    The image score is the highest box score, exactly as in the notebook, but the
    loop is batch-size agnostic instead of relying on ``[-1]`` indexing.
    """

    model.eval()
    scores: list[float] = []
    labels: list[int] = []

    for batch in data_loader:
        images, batch_labels = batch[0], batch[1]
        outputs = model([image.to(device) for image in images])
        for output, label in zip(outputs, batch_labels, strict=True):
            box_scores = output["scores"]
            scores.append(float(box_scores.max().item()) if box_scores.numel() else 0.0)
            labels.append(int(label))

    predictions = [1 if s >= threshold else 0 for s in scores]
    correct = sum(
        int(prediction == truth)
        for prediction, truth in zip(predictions, labels, strict=True)
    )
    accuracy = correct / max(len(labels), 1)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    return {"accuracy": accuracy, "auc": float(auc), "scores": scores, "labels": labels}
