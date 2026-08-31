"""Evaluation artefacts served to the metrics dashboard."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from cxr.api.deps import get_service, get_settings_dep
from cxr.config import Settings
from cxr.inference.service import InferenceService
from cxr.models.registry import UnknownModelError

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def list_metrics(
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Summary of every evaluation run found under ``artifacts/``."""

    summaries = {}
    for name in service.model_names():
        path = settings.artifacts_dir / name / "metrics.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries[name] = {
                key: payload.get(key)
                for key in ("accuracy", "auc", "threshold", "n_images", "evaluated_at", "split")
            }
    return {"models": summaries}


@router.get("/{name}")
def model_metrics(
    name: str,
    service: InferenceService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Full metrics payload (including ROC points) for one model."""

    try:
        service.registry.get(name)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = settings.artifacts_dir / name / "metrics.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"no evaluation artefacts for {name!r}. "
                f"Run 'cxr evaluate --model {name}' to generate them."
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))
