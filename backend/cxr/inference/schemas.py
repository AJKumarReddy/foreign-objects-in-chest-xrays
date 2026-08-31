"""Pydantic DTOs exchanged with the front end."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cxr.models.base import DetectionResult
from cxr.models.registry import ModelStatus


class DetectionDTO(BaseModel):
    """A single predicted box in original-image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label: str = "foreign_object"
    center_x: float
    center_y: float


class PredictionResponse(BaseModel):
    model_name: str = Field(serialization_alias="model", validation_alias="model")
    detections: list[DetectionDTO]
    detection_count: int
    image_score: float
    has_foreign_object: bool
    conf_threshold: float
    image_width: int
    image_height: int
    inference_ms: float
    source: str = "model"
    filename: str | None = None

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class CompareResponse(BaseModel):
    filename: str | None = None
    image_width: int
    image_height: int
    results: list[PredictionResponse]
    agreement: str  # "both" | "frcnn_only" | "yolo_only" | "neither" | "partial"


class BatchItem(BaseModel):
    filename: str
    ok: bool
    error: str | None = None
    prediction: PredictionResponse | None = None


class BatchResponse(BaseModel):
    items: list[BatchItem]
    total: int
    positive: int
    negative: int
    failed: int
    total_ms: float


class ModelInfo(BaseModel):
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
    has_metrics: bool = False


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    device: str
    demo_mode: bool
    any_ready: bool


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    device: str
    torch_version: str
    torchvision_version: str
    ultralytics_available: bool
    models_ready: list[str]
    dataset_available: bool


class SampleImage(BaseModel):
    id: str
    filename: str
    title: str
    description: str
    url: str
    synthetic: bool = True


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None


def detection_result_to_dto(
    result: DetectionResult, filename: str | None = None
) -> PredictionResponse:
    """Adapt the domain object to the wire format."""

    detections = []
    for det in result.detections:
        cx, cy = det.center
        detections.append(
            DetectionDTO(
                x_min=det.box[0],
                y_min=det.box[1],
                x_max=det.box[2],
                y_max=det.box[3],
                score=det.score,
                label=det.label,
                center_x=cx,
                center_y=cy,
            )
        )
    return PredictionResponse(
        model=result.model,
        detections=detections,
        detection_count=len(detections),
        image_score=result.image_score,
        has_foreign_object=result.has_foreign_object,
        conf_threshold=result.conf_threshold,
        image_width=result.image_width,
        image_height=result.image_height,
        inference_ms=round(result.inference_ms, 2),
        source=result.source,
        filename=filename,
    )


def model_status_to_info(status: ModelStatus, has_metrics: bool = False) -> ModelInfo:
    return ModelInfo(**status.__dict__, has_metrics=has_metrics)
