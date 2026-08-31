"""Prediction endpoints: single image, batch, and model comparison."""

from __future__ import annotations

import csv
import io
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from cxr.api.deps import get_service, get_settings_dep
from cxr.config import Settings
from cxr.inference.schemas import (
    BatchItem,
    BatchResponse,
    CompareResponse,
    PredictionResponse,
    detection_result_to_dto,
)
from cxr.inference.service import (
    ImageValidationError,
    InferenceService,
    ModelNotReadyError,
    decode_image,
)

router = APIRouter(tags=["predict"])


def _read_image(payload: bytes, settings: Settings):
    try:
        return decode_image(payload, max_bytes=settings.max_upload_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _not_ready(exc: ModelNotReadyError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"detail": str(exc), "hint": getattr(exc, "hint", None)},
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image"),
    model: str | None = Form(None),
    conf: float | None = Form(None),
    max_detections: int | None = Form(None),
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
) -> PredictionResponse:
    """Detect foreign objects in a single chest X-ray."""

    image = _read_image(await file.read(), settings)
    try:
        result = service.predict(
            image,
            model=model,
            conf=conf,
            max_detections=max_detections,
            filename=file.filename,
        )
    except ModelNotReadyError as exc:
        raise _not_ready(exc) from exc
    return detection_result_to_dto(result, filename=file.filename)


@router.post("/predict/sample/{sample_id}", response_model=PredictionResponse)
def predict_sample(
    sample_id: str,
    model: str | None = Query(None),
    conf: float | None = Query(None),
    service: InferenceService = Depends(get_service),
) -> PredictionResponse:
    """Run a bundled sample image through the pipeline (or its demo record)."""

    sample = next((s for s in service.samples() if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"unknown sample {sample_id!r}")

    from PIL import Image

    path = service.sample_path(sample["filename"])
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    try:
        result = service.predict(image, model=model, conf=conf, filename=sample["filename"])
    except ModelNotReadyError as exc:
        raise _not_ready(exc) from exc
    return detection_result_to_dto(result, filename=sample["filename"])


@router.post("/compare", response_model=CompareResponse)
async def compare(
    file: UploadFile = File(...),
    conf: float | None = Form(None),
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
) -> CompareResponse:
    """Run every detector on the same image for a side-by-side view."""

    image = _read_image(await file.read(), settings)
    try:
        results = service.compare(image, conf=conf, filename=file.filename)
    except ModelNotReadyError as exc:
        raise _not_ready(exc) from exc
    return CompareResponse(
        filename=file.filename,
        image_width=image.size[0],
        image_height=image.size[1],
        results=[detection_result_to_dto(r, filename=file.filename) for r in results],
        agreement=service.agreement(results),
    )


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    files: list[UploadFile] = File(...),
    model: str | None = Form(None),
    conf: float | None = Form(None),
    fmt: str = Query("json", alias="format", pattern="^(json|csv)$"),
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    """Score many images at once; ``?format=csv`` streams a downloadable report."""

    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=422,
            detail=f"too many files ({len(files)}); the limit is {settings.max_batch_files}",
        )

    started = time.perf_counter()
    items: list[BatchItem] = []
    for upload in files:
        try:
            image = decode_image(await upload.read(), max_bytes=settings.max_upload_bytes)
            result = service.predict(image, model=model, conf=conf, filename=upload.filename)
            items.append(
                BatchItem(
                    filename=upload.filename or "unnamed",
                    ok=True,
                    prediction=detection_result_to_dto(result, filename=upload.filename),
                )
            )
        except (ImageValidationError, ModelNotReadyError) as exc:
            items.append(
                BatchItem(filename=upload.filename or "unnamed", ok=False, error=str(exc))
            )

    positive = sum(1 for i in items if i.ok and i.prediction and i.prediction.has_foreign_object)
    failed = sum(1 for i in items if not i.ok)
    response = BatchResponse(
        items=items,
        total=len(items),
        positive=positive,
        negative=len(items) - positive - failed,
        failed=failed,
        total_ms=round((time.perf_counter() - started) * 1000, 1),
    )

    if fmt == "csv":
        return StreamingResponse(
            _to_csv(response),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cxr_batch_results.csv"'},
        )
    return response


def _to_csv(response: BatchResponse) -> io.StringIO:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["image_name", "prediction", "verdict", "detections", "model", "inference_ms", "error"]
    )
    for item in response.items:
        if item.ok and item.prediction:
            pred = item.prediction
            writer.writerow(
                [
                    item.filename,
                    f"{pred.image_score:.6f}",
                    "foreign_object" if pred.has_foreign_object else "clear",
                    pred.detection_count,
                    pred.model_name,
                    pred.inference_ms,
                    "",
                ]
            )
        else:
            writer.writerow([item.filename, "", "error", 0, "", "", item.error or "failed"])
    buffer.seek(0)
    return buffer
