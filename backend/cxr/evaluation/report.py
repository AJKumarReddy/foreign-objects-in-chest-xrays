"""End-to-end evaluation of a trained detector on the dev split.

Produces the two CSVs the notebook emitted (``classification.csv`` and
``localization.csv``, in the object-CXR submission format) plus a
``metrics.json`` that the web dashboard reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from cxr.config import Settings
from cxr.data.dataset import labels_to_dict, load_labels
from cxr.evaluation.classification import classification_report
from cxr.evaluation.localization import format_localization, froc, hits_and_misses
from cxr.logging_conf import get_logger
from cxr.models.base import Detector
from cxr.models.registry import ModelRegistry

logger = get_logger(__name__)

#: Low score floor for localization rows - the notebook used 0.1 so that weak
#: candidates still contribute to the FROC curve.
LOCALIZATION_FLOOR = 0.1


@dataclass
class EvaluationResult:
    model: str
    metrics: dict
    output_dir: Path


def evaluate(
    settings: Settings,
    model_name: str = "frcnn",
    split: str = "dev",
    limit: int | None = None,
    threshold: float | None = None,
    registry: ModelRegistry | None = None,
) -> EvaluationResult:
    data = settings.data
    if not data.is_available():
        raise FileNotFoundError(f"dataset not found under {data.root}")

    image_dir = data.dev_images if split == "dev" else data.train_images
    csv_path = data.dev_labels if split == "dev" else data.train_labels
    labels = labels_to_dict(load_labels(csv_path))

    registry = registry or ModelRegistry(settings)
    detector: Detector = registry.load(model_name)
    threshold = threshold if threshold is not None else detector.conf_threshold

    names = sorted(labels)
    if limit:
        names = names[:limit]

    scores: list[float] = []
    truths: list[int] = []
    localizations: list[str] = []
    per_image_hits = []

    for image_name in tqdm(names, desc=f"evaluating {model_name}"):
        path = Path(image_dir) / image_name
        if not path.is_file():
            continue
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        result = detector.predict(image, conf=LOCALIZATION_FLOOR)

        scores.append(result.image_score)
        truths.append(int(bool(labels[image_name])))
        localizations.append(format_localization(result.detections))
        per_image_hits.append(
            hits_and_misses(
                result.detections, labels[image_name], result.image_width, result.image_height
            )
        )

    output_dir = settings.artifacts_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"image_name": names[: len(scores)], "prediction": scores}).to_csv(
        output_dir / "classification.csv", index=False
    )
    pd.DataFrame({"image_name": names[: len(localizations)], "prediction": localizations}).to_csv(
        output_dir / "localization.csv", index=False
    )

    metrics = classification_report(scores, truths, threshold=threshold)
    metrics.update(
        {
            "model": model_name,
            "split": split,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(registry.device),
            "weights": str(detector.weights_path),
            "localization": froc(per_image_hits),
        }
    )
    run_manifest = Path(detector.weights_path).parent / "run.json"
    if run_manifest.is_file():
        metrics["training_run"] = json.loads(run_manifest.read_text(encoding="utf-8"))

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "%s: acc %.4f | auc %.4f | FROC mean sens %.4f -> %s",
        model_name,
        metrics["accuracy"],
        metrics["auc"],
        metrics["localization"]["mean_sensitivity"],
        output_dir,
    )
    return EvaluationResult(model_name, metrics, output_dir)
