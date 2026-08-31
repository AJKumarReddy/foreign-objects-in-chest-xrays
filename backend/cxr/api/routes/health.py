"""Liveness and environment reporting."""

from __future__ import annotations

import torch
import torchvision
from fastapi import APIRouter, Depends

from cxr import __version__
from cxr.api.deps import get_service, get_settings_dep
from cxr.config import Settings
from cxr.inference.schemas import HealthResponse
from cxr.inference.service import InferenceService
from cxr.models.yolo_detector import ultralytics_available

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    ready = [s.name for s in service.statuses() if s.ready]
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.environment,
        device=service.device,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        ultralytics_available=ultralytics_available(),
        models_ready=ready,
        dataset_available=settings.data.is_available(),
    )
