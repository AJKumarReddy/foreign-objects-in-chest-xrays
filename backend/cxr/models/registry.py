"""Detector registry: name -> configured, lazily loaded instance."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from cxr.config import Settings
from cxr.device import resolve_device
from cxr.logging_conf import get_logger
from cxr.models.base import Detector
from cxr.models.faster_rcnn import FasterRCNNDetector
from cxr.models.yolo_detector import YoloDetector

logger = get_logger(__name__)

DETECTOR_CLASSES: dict[str, type[Detector]] = {
    FasterRCNNDetector.name: FasterRCNNDetector,
    YoloDetector.name: YoloDetector,
}


class UnknownModelError(KeyError):
    """Raised when a caller asks for a detector that does not exist."""


@dataclass
class ModelStatus:
    name: str
    display_name: str
    description: str
    input_size: int
    default_conf: float
    weights_path: str
    weights_present: bool
    dependencies_available: bool
    ready: bool
    loaded: bool


class ModelRegistry:
    """Owns one instance per detector and caches loaded weights in memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._lock = threading.Lock()
        self._detectors: dict[str, Detector] = {
            FasterRCNNDetector.name: FasterRCNNDetector(
                weights=settings.frcnn.weights,
                input_size=settings.frcnn.input_size,
                conf_threshold=settings.frcnn.conf_threshold,
                max_detections=settings.frcnn.max_detections,
            ),
            YoloDetector.name: YoloDetector(
                weights=settings.yolo.weights,
                input_size=settings.yolo.input_size,
                conf_threshold=settings.yolo.conf_threshold,
                max_detections=settings.yolo.max_detections,
            ),
        }

    # -- introspection -----------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self._detectors)

    def get(self, name: str) -> Detector:
        try:
            return self._detectors[name]
        except KeyError as exc:
            raise UnknownModelError(
                f"unknown model {name!r}; available: {', '.join(self._detectors)}"
            ) from exc

    def status(self, name: str) -> ModelStatus:
        det = self.get(name)
        return ModelStatus(
            name=det.name,
            display_name=det.display_name,
            description=det.description,
            input_size=det.input_size,
            default_conf=det.conf_threshold,
            weights_path=str(det.weights_path),
            weights_present=det.weights_present,
            dependencies_available=det.dependencies_available(),
            ready=det.is_ready(),
            loaded=det.is_loaded,
        )

    def statuses(self) -> list[ModelStatus]:
        return [self.status(name) for name in self._detectors]

    def any_ready(self) -> bool:
        return any(det.is_ready() for det in self._detectors.values())

    # -- loading -----------------------------------------------------------
    def load(self, name: str, device: torch.device | None = None) -> Detector:
        """Load (once) and return a detector ready for inference."""

        det = self.get(name)
        target = device or self.device
        if det.is_loaded and det.device == target:
            return det
        with self._lock:
            if not (det.is_loaded and det.device == target):
                det.load(target)
        return det

    def unload_all(self) -> None:
        with self._lock:
            for det in self._detectors.values():
                det.unload()
