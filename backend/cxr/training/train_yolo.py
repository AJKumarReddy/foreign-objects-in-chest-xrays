"""Train the ultralytics YOLO detector on object-CXR.

Notebook cells 19-22 could not work as written: the labels were exported to a
working directory the ``data.yaml`` never referenced, and the best checkpoint was
read from ``runs/train/...`` instead of ultralytics' ``runs/detect/...``. Both are
handled here - the dataset is materialised into a proper images/labels tree and
the checkpoint path is taken from the trainer itself.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cxr.config import Settings
from cxr.data.dataset import labels_to_dict, load_labels
from cxr.data.yolo_export import export_dataset
from cxr.device import resolve_device
from cxr.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class YoloTrainResult:
    checkpoint: Path
    run_dir: Path
    data_yaml: Path
    metrics: dict


def prepare_dataset(settings: Settings, output_root: Path | None = None,
                    include_negatives: bool = True) -> Path:
    """Materialise the YOLO dataset tree and return the ``data.yaml`` path."""

    data = settings.data
    if not data.is_available():
        raise FileNotFoundError(f"dataset not found under {data.root}")
    output_root = Path(output_root or settings.artifacts_dir / "yolo_dataset")

    train_labels = labels_to_dict(load_labels(data.train_labels))
    dev_labels = labels_to_dict(load_labels(data.dev_labels))
    if data.train_on_positives_only:
        train_labels = {k: v for k, v in train_labels.items() if v}

    yaml_path, stats = export_dataset(
        data.train_images, train_labels, data.dev_images, dev_labels, output_root,
        include_negatives=include_negatives,
    )
    logger.info("YOLO dataset ready: %s", yaml_path)
    return yaml_path


def train(settings: Settings, epochs: int | None = None, data_yaml: Path | None = None,
          include_negatives: bool = True) -> YoloTrainResult:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "ultralytics is required to train the YOLO model: pip install ultralytics"
        ) from exc

    cfg = settings.training
    yaml_path = Path(data_yaml) if data_yaml else prepare_dataset(
        settings, include_negatives=include_negatives
    )
    device = resolve_device(settings.device)
    project = settings.runs_dir / "yolo"
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg.yolo_base_weights)
    model.train(
        data=str(yaml_path),
        epochs=epochs or cfg.yolo_epochs,
        imgsz=cfg.yolo_imgsz,
        batch=cfg.yolo_batch,
        workers=settings.data.num_workers,
        seed=cfg.seed,
        device=0 if device.type == "cuda" else "cpu",
        project=str(project),
        name="foreign_object_detection",
        exist_ok=True,
    )

    # ultralytics writes to runs/detect/<name> by default; save_dir is authoritative.
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} is missing")

    destination = Path(settings.yolo.weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    logger.info("copied %s -> %s", best, destination)

    validation = model.val()
    metrics = {
        "map50": float(getattr(validation.box, "map50", float("nan"))),
        "map50_95": float(getattr(validation.box, "map", float("nan"))),
        "precision": float(getattr(validation.box, "mp", float("nan"))),
        "recall": float(getattr(validation.box, "mr", float("nan"))),
    }
    manifest = {
        "model": "yolo",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_weights": cfg.yolo_base_weights,
        "epochs": epochs or cfg.yolo_epochs,
        "imgsz": cfg.yolo_imgsz,
        "batch": cfg.yolo_batch,
        "device": str(device),
        "run_dir": str(run_dir),
        "data_yaml": str(yaml_path),
        "detection_metrics": metrics,
    }
    (destination.parent / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return YoloTrainResult(destination, run_dir, yaml_path, metrics)
