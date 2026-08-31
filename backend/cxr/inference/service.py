"""Inference orchestration: validation, model access, batching, demo mode."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cxr.config import Settings
from cxr.logging_conf import get_logger
from cxr.models.base import Detection, DetectionResult
from cxr.models.registry import ModelRegistry, UnknownModelError

logger = get_logger(__name__)

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
DEMO_MANIFEST = ASSETS_DIR / "demo.json"

#: Reject absurd resolutions before PIL allocates the pixel buffer.
MAX_PIXELS = 80_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


class ImageValidationError(ValueError):
    """The uploaded bytes are not a usable image."""


class ModelNotReadyError(RuntimeError):
    """A prediction was requested but the model cannot serve it."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def decode_image(payload: bytes, max_bytes: int | None = None) -> Image.Image:
    """Validate and decode uploaded bytes into an RGB :class:`PIL.Image`."""

    if not payload:
        raise ImageValidationError("the uploaded file is empty")
    if max_bytes is not None and len(payload) > max_bytes:
        raise ImageValidationError(
            f"file is {len(payload) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB"
        )
    try:
        probe = Image.open(io.BytesIO(payload))
        probe.verify()  # cheap structural check, consumes the handle
        image = Image.open(io.BytesIO(payload))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("file is not a readable image") from exc

    if image.format and image.format.upper() not in ALLOWED_FORMATS:
        raise ImageValidationError(f"unsupported image format: {image.format}")
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise ImageValidationError(
            f"image is {width}x{height}; the limit is {MAX_PIXELS // 1_000_000} megapixels"
        )
    return image.convert("RGB")


class InferenceService:
    """Everything the API layer needs, with no FastAPI types in sight."""

    def __init__(self, settings: Settings, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self._demo: dict | None = None

    # -- models ------------------------------------------------------------
    @property
    def device(self) -> str:
        return str(self.registry.device)

    def model_names(self) -> list[str]:
        return self.registry.names

    def statuses(self):
        return self.registry.statuses()

    def has_metrics(self, name: str) -> bool:
        return (self.settings.artifacts_dir / name / "metrics.json").is_file()

    def load_model(self, name: str) -> float:
        started = time.perf_counter()
        self.registry.load(name)
        return (time.perf_counter() - started) * 1000

    def default_model(self) -> str:
        for status in self.registry.statuses():
            if status.ready:
                return status.name
        return self.registry.names[0]

    # -- demo mode ---------------------------------------------------------
    @property
    def demo_available(self) -> bool:
        return self.settings.demo_mode_enabled and DEMO_MANIFEST.is_file()

    def _demo_manifest(self) -> dict:
        if self._demo is None:
            self._demo = (
                json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))
                if DEMO_MANIFEST.is_file()
                else {"samples": []}
            )
        return self._demo

    def samples(self) -> list[dict]:
        return self._demo_manifest().get("samples", [])

    def sample_path(self, filename: str) -> Path:
        # basename only - never let a request escape the samples directory
        path = (SAMPLES_DIR / Path(filename).name).resolve()
        if not str(path).startswith(str(SAMPLES_DIR.resolve())) or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def demo_prediction(self, filename: str, model: str, conf: float) -> DetectionResult:
        """Replay recorded detections for a bundled sample image.

        Used only when no checkpoint is installed; every response produced here
        is tagged ``source="demo"`` so the UI can badge it as not-real inference.
        """

        name = Path(filename).name
        for sample in self.samples():
            if sample["filename"] == name:
                recorded = sample.get("detections", {}).get(model, [])
                detections = [
                    Detection(box=tuple(d["box"]), score=float(d["score"]))
                    for d in recorded
                    if float(d["score"]) >= conf
                ]
                return DetectionResult(
                    model=model,
                    detections=detections,
                    image_width=sample["width"],
                    image_height=sample["height"],
                    inference_ms=0.0,
                    conf_threshold=conf,
                    source="demo",
                )
        raise FileNotFoundError(f"no demo record for {filename!r}")

    # -- prediction --------------------------------------------------------
    def predict(
        self,
        image: Image.Image,
        model: str | None = None,
        conf: float | None = None,
        max_detections: int | None = None,
        filename: str | None = None,
    ) -> DetectionResult:
        """Run one image through one detector.

        Falls back to the recorded demo response when the checkpoint is missing
        *and* the image is one of the bundled samples.
        """

        name = model or self.default_model()
        try:
            detector = self.registry.get(name)
        except UnknownModelError as exc:
            raise ModelNotReadyError(str(exc)) from exc

        threshold = detector.conf_threshold if conf is None else conf

        if not detector.is_ready():
            if self.demo_available and filename:
                try:
                    return self.demo_prediction(filename, name, threshold)
                except FileNotFoundError:
                    pass
            raise ModelNotReadyError(*self._not_ready_reason(name))

        self.registry.load(name)
        return detector.predict(image, conf=threshold, max_detections=max_detections)

    def _not_ready_reason(self, name: str) -> tuple[str, str]:
        status = self.registry.status(name)
        if not status.dependencies_available:
            return (
                f"{status.display_name} is unavailable: ultralytics is not installed.",
                "Run: pip install ultralytics",
            )
        return (
            f"{status.display_name} has no checkpoint at {status.weights_path}.",
            f"Train one with 'cxr train {name}' or copy your .pt file to that path. "
            "Bundled sample images still work in demo mode.",
        )

    def compare(
        self,
        image: Image.Image,
        conf: float | None = None,
        filename: str | None = None,
    ) -> list[DetectionResult]:
        """Run every registered detector on the same image."""

        results: list[DetectionResult] = []
        errors: list[str] = []
        for name in self.registry.names:
            try:
                results.append(self.predict(image, model=name, conf=conf, filename=filename))
            except ModelNotReadyError as exc:
                errors.append(str(exc))
        if not results:
            raise ModelNotReadyError("; ".join(errors) or "no detector is available")
        return results

    @staticmethod
    def agreement(results: list[DetectionResult]) -> str:
        positives = {r.model for r in results if r.has_foreign_object}
        if len(results) < 2:
            return "single"
        if len(positives) == len(results):
            return "both"
        if not positives:
            return "neither"
        return f"{sorted(positives)[0]}_only"
