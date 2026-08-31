"""Ultralytics YOLO backend.

``ultralytics`` is an optional dependency: the app must start and report a clear
status even when it is missing, so the import is deferred until load time.
"""

from __future__ import annotations

import importlib.util
import time

import torch
from PIL import Image

from cxr.logging_conf import get_logger
from cxr.models.base import Detection, DetectionResult, Detector

logger = get_logger(__name__)


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


class YoloDetector(Detector):
    name = "yolo"
    display_name = "YOLO (ultralytics)"
    description = (
        "Single-stage detector trained at 480px. Much faster on CPU and easier to "
        "deploy; typically trades a little recall for speed."
    )

    @staticmethod
    def dependencies_available() -> bool:
        return ultralytics_available()

    def load(self, device: torch.device) -> None:
        if self._model is not None and self.device == device:
            return
        if not ultralytics_available():
            raise ModuleNotFoundError(
                "ultralytics is not installed. Install it with 'pip install ultralytics' "
                "or 'pip install -e backend[yolo]'."
            )
        if not self.weights_present:
            raise FileNotFoundError(
                f"YOLO checkpoint not found at {self.weights_path}. "
                "Train one with 'cxr train yolo' or copy best.pt into models/yolo/."
            )
        from ultralytics import YOLO

        logger.info("loading YOLO weights from %s onto %s", self.weights_path, device)
        model = YOLO(str(self.weights_path))
        model.to(str(device))
        self._model = model
        self.device = device

    def predict(
        self,
        image: Image.Image,
        conf: float | None = None,
        max_detections: int | None = None,
    ) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("model is not loaded; call load(device) first")
        threshold = self.conf_threshold if conf is None else conf
        limit = max_detections or self.max_detections

        image = image.convert("RGB")
        width, height = image.size

        started = time.perf_counter()
        # ultralytics letterboxes internally and returns boxes already mapped
        # back to original image coordinates.
        results = self._model.predict(
            source=image,
            conf=threshold,
            imgsz=self.input_size,
            max_det=limit,
            device=str(self.device),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                    detections.append(
                        Detection(box=tuple(float(v) for v in xyxy), score=float(score))
                    )
        return DetectionResult(
            model=self.name,
            detections=detections,
            image_width=width,
            image_height=height,
            inference_ms=elapsed_ms,
            conf_threshold=threshold,
        )
