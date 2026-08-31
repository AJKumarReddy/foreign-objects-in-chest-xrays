"""Shared fixtures: synthetic images, a fake dataset on disk, a stub detector."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from cxr.config import Settings
from cxr.models.base import Detection, DetectionResult, Detector


def make_image(path: Path, size: tuple[int, int] = (200, 150), box=None) -> None:
    image = Image.new("RGB", size, (40, 40, 40))
    if box:
        ImageDraw.Draw(image).rectangle(box, fill=(240, 240, 240))
    image.save(path)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A miniature object-CXR layout: 3 train images (2 annotated), 2 dev images."""

    root = tmp_path / "object-CXR"
    (root / "train").mkdir(parents=True)
    (root / "dev").mkdir(parents=True)

    train_rows = [
        ("t0.jpg", "0 10 10 60 50"),
        ("t1.jpg", "2 20 20 80 20 80 70 20 70;1 100 30 140 90"),
        ("t2.jpg", ""),
    ]
    dev_rows = [("d0.jpg", "0 15 15 55 45"), ("d1.jpg", "")]

    for name, _ in train_rows:
        make_image(root / "train" / name, box=(10, 10, 60, 50))
    for name, _ in dev_rows:
        make_image(root / "dev" / name, box=(15, 15, 55, 45))

    for filename, rows in (("train.csv", train_rows), ("dev.csv", dev_rows)):
        with (root / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_name", "annotation"])
            writer.writerows(rows)
    return root


@pytest.fixture
def settings(tmp_path: Path, dataset_root: Path) -> Settings:
    return Settings(
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "artifacts" / "runs",
        device="cpu",
        data={"root": dataset_root, "image_size": 64, "num_workers": 0, "batch_size": 2},
        frcnn={"weights": tmp_path / "models" / "frcnn" / "model.pt", "input_size": 64},
        yolo={"weights": tmp_path / "models" / "yolo" / "best.pt", "input_size": 64},
    )


class StubDetector(Detector):
    """A detector that always finds one box, so API tests need no real weights."""

    name = "frcnn"
    display_name = "Stub Faster R-CNN"
    description = "test double"

    def __init__(self, score: float = 0.87) -> None:
        super().__init__(weights=None, input_size=64, conf_threshold=0.5)
        self.score = score
        self._model = object()

    @property
    def weights_present(self) -> bool:
        return True

    def load(self, device: torch.device) -> None:
        self.device = device

    def predict(self, image, conf=None, max_detections=None) -> DetectionResult:
        threshold = self.conf_threshold if conf is None else conf
        width, height = image.size
        detections = []
        if self.score >= threshold:
            detections.append(
                Detection(box=(10.0, 10.0, width / 2, height / 2), score=self.score)
            )
        return DetectionResult(
            model=self.name,
            detections=detections,
            image_width=width,
            image_height=height,
            inference_ms=1.23,
            conf_threshold=threshold,
        )


@pytest.fixture
def image_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), (30, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()
