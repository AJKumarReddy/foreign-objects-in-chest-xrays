"""Image level (classification) metrics: accuracy, AUC, ROC, threshold sweep."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


def accuracy_at(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> float:
    preds = np.asarray(scores, dtype=float) >= threshold
    return float((preds == np.asarray(labels, dtype=bool)).mean())


def confusion_at(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> dict:
    preds = np.asarray(scores, dtype=float) >= threshold
    truth = np.asarray(labels, dtype=bool)
    tp = int(np.sum(preds & truth))
    tn = int(np.sum(~preds & ~truth))
    fp = int(np.sum(preds & ~truth))
    fn = int(np.sum(~preds & truth))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    denominator = precision + sensitivity
    f1 = 2 * precision * sensitivity / denominator if denominator else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1,
    }


def roc_points(scores: Sequence[float], labels: Sequence[int], max_points: int = 300) -> dict:
    """ROC curve, AUC, and the Youden-optimal operating point.

    The curve is subsampled so the JSON the dashboard downloads stays small.
    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = float(sk_auc(fpr, tpr))

    youden = np.argmax(tpr - fpr)
    best_threshold = float(thresholds[youden])
    if not np.isfinite(best_threshold):
        best_threshold = 1.0

    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]

    return {
        "auc": roc_auc,
        "points": [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr, strict=True)],
        "best_threshold": best_threshold,
        "best_tpr": float(tpr[min(youden, len(tpr) - 1)]),
        "best_fpr": float(fpr[min(youden, len(fpr) - 1)]),
    }


def threshold_sweep(
    scores: Sequence[float], labels: Sequence[int], steps: int = 21
) -> list[dict]:
    """Accuracy / sensitivity / specificity across candidate thresholds."""

    sweep = []
    for threshold in np.linspace(0.0, 1.0, steps):
        stats = confusion_at(scores, labels, float(threshold))
        sweep.append(
            {
                "threshold": round(float(threshold), 3),
                "accuracy": accuracy_at(scores, labels, float(threshold)),
                "sensitivity": stats["sensitivity"],
                "specificity": stats["specificity"],
                "f1": stats["f1"],
            }
        )
    return sweep


def classification_report(
    scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5
) -> dict:
    roc = roc_points(scores, labels)
    return {
        "n_images": len(labels),
        "n_positive": int(np.sum(np.asarray(labels, dtype=bool))),
        "threshold": threshold,
        "accuracy": accuracy_at(scores, labels, threshold),
        "auc": roc["auc"],
        "confusion": confusion_at(scores, labels, threshold),
        "roc": roc,
        "sweep": threshold_sweep(scores, labels),
    }
