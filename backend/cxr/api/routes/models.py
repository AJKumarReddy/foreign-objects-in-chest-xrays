"""Model discovery and eager loading."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from cxr.api.deps import get_service
from cxr.inference.schemas import ModelListResponse, model_status_to_info
from cxr.inference.service import InferenceService
from cxr.models.registry import UnknownModelError

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(service: InferenceService = Depends(get_service)) -> ModelListResponse:
    statuses = service.statuses()
    return ModelListResponse(
        models=[model_status_to_info(s, service.has_metrics(s.name)) for s in statuses],
        device=service.device,
        demo_mode=service.demo_available and not any(s.ready for s in statuses),
        any_ready=any(s.ready for s in statuses),
    )


@router.post("/{name}/load")
def load_model(name: str, service: InferenceService = Depends(get_service)) -> dict:
    try:
        service.registry.get(name)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        elapsed = service.load_model(name)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model": name, "loaded": True, "load_ms": round(elapsed, 1)}
