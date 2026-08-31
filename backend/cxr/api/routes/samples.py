"""Bundled synthetic sample images used for demos and quick checks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from cxr.api.deps import get_service
from cxr.inference.schemas import SampleImage
from cxr.inference.service import InferenceService

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("", response_model=list[SampleImage])
def list_samples(service: InferenceService = Depends(get_service)) -> list[SampleImage]:
    return [
        SampleImage(
            id=s["id"],
            filename=s["filename"],
            title=s["title"],
            description=s["description"],
            url=f"/api/samples/{s['id']}/image",
            synthetic=s.get("synthetic", True),
        )
        for s in service.samples()
    ]


@router.get("/{sample_id}/image")
def sample_image(sample_id: str, service: InferenceService = Depends(get_service)):
    sample = next((s for s in service.samples() if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"unknown sample {sample_id!r}")
    try:
        path = service.sample_path(sample["filename"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="sample image is missing") from exc
    return FileResponse(path, media_type="image/png", filename=sample["filename"])
