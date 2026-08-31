"""Faster R-CNN (ResNet50-FPN) backend - the notebook's primary model."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cxr.data.transforms import build_transforms
from cxr.logging_conf import get_logger
from cxr.models.base import Detection, DetectionResult, Detector, rescale_boxes

logger = get_logger(__name__)

NUM_CLASSES = 2  # background + foreign_object


def build_faster_rcnn(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
    """Create the detection model with a fresh box predictor head.

    ``pretrained=True`` was removed from torchvision; the modern spelling is an
    explicit weights enum, and ``None`` keeps the network random-initialised
    (which is what we want when we are about to load our own checkpoint).
    """

    weights = (
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if pretrained_backbone
        else None
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class FasterRCNNDetector(Detector):
    name = "frcnn"
    display_name = "Faster R-CNN (ResNet50-FPN)"
    description = (
        "Two-stage detector fine-tuned on object-CXR at 600x600. Higher recall on "
        "small, low-contrast objects; slower than YOLO on CPU."
    )

    def load(self, device: torch.device) -> None:
        if self._model is not None and self.device == device:
            return
        if not self.weights_present:
            raise FileNotFoundError(
                f"Faster R-CNN checkpoint not found at {self.weights_path}. "
                "Train one with 'cxr train frcnn' or copy model.pt into models/frcnn/."
            )
        logger.info("loading Faster R-CNN weights from %s onto %s", self.weights_path, device)
        model = build_faster_rcnn(pretrained_backbone=False)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        self._model = model
        self.device = device
        self._transform = build_transforms(self.input_size)

    @torch.inference_mode()
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
        tensor = self._transform(image).to(self.device)

        started = time.perf_counter()
        outputs = self._model([tensor])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        keep = outputs["scores"] >= threshold
        boxes = outputs["boxes"][keep].detach().cpu()
        scores = outputs["scores"][keep].detach().cpu()
        if boxes.shape[0] > limit:
            boxes, scores = boxes[:limit], scores[:limit]

        boxes = rescale_boxes(boxes, (self.input_size, self.input_size), (width, height))
        detections = [
            Detection(box=tuple(float(v) for v in box), score=float(score))
            for box, score in zip(boxes.tolist(), scores.tolist(), strict=True)
        ]
        return DetectionResult(
            model=self.name,
            detections=detections,
            image_width=width,
            image_height=height,
            inference_ms=elapsed_ms,
            conf_threshold=threshold,
        )

    @torch.inference_mode()
    def forward_batch(self, tensors: list[torch.Tensor]) -> list[dict]:
        """Batched forward pass used by the evaluation loop."""

        if self._model is None:
            raise RuntimeError("model is not loaded; call load(device) first")
        return self._model([t.to(self.device) for t in tensors])


def load_checkpoint_for_eval(weights: Path | str, device: torch.device):
    """Convenience helper for scripts: build + load + eval()."""

    model = build_faster_rcnn(pretrained_backbone=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model
