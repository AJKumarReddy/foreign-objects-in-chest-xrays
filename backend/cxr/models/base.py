"""The single abstraction every consumer (API, CLI, evaluation) depends on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image


@dataclass
class Detection:
    """One predicted foreign object, in ORIGINAL image pixel coordinates."""

    box: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    score: float
    label: str = "foreign_object"

    @property
    def center(self) -> tuple[float, float]:
        x_min, y_min, x_max, y_max = self.box
        return (x_min + x_max) / 2, (y_min + y_max) / 2

    @property
    def area(self) -> float:
        x_min, y_min, x_max, y_max = self.box
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


@dataclass
class DetectionResult:
    """Everything one forward pass produces."""

    model: str
    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_ms: float = 0.0
    conf_threshold: float = 0.5
    source: str = "model"  # "model" | "demo"

    @property
    def image_score(self) -> float:
        """Image level probability of a foreign object = best box score."""

        return max((d.score for d in self.detections), default=0.0)

    @property
    def has_foreign_object(self) -> bool:
        return self.image_score >= self.conf_threshold


def rescale_boxes(
    boxes: torch.Tensor, from_size: tuple[int, int], to_size: tuple[int, int]
) -> torch.Tensor:
    """Map boxes from the network's input square back to the original image.

    ``from_size``/``to_size`` are ``(width, height)``.
    """

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    scale = torch.tensor(
        [scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype, device=boxes.device
    )
    return boxes * scale


class Detector(ABC):
    """Common interface implemented by the Faster R-CNN and YOLO backends."""

    name: str = "detector"
    display_name: str = "Detector"
    description: str = ""

    def __init__(self, weights: Path | str | None, input_size: int, conf_threshold: float = 0.5,
                 max_detections: int = 100) -> None:
        self.weights_path = Path(weights) if weights else None
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.device: torch.device | None = None
        self._model = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def weights_present(self) -> bool:
        return bool(self.weights_path and self.weights_path.is_file())

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def is_ready(self) -> bool:
        """True when a prediction can be served right now (or after a load)."""

        return self.weights_present and self.dependencies_available()

    @staticmethod
    def dependencies_available() -> bool:
        return True

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load weights onto ``device``. Must be idempotent."""

    def unload(self) -> None:
        self._model = None

    # -- inference ---------------------------------------------------------
    @abstractmethod
    def predict(self, image: Image.Image, conf: float | None = None,
                max_detections: int | None = None) -> DetectionResult:
        """Run the detector on a PIL image and return original-space boxes."""
