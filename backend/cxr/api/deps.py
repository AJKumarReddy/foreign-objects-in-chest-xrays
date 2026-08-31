"""FastAPI dependencies.

The service is created once at startup and stashed on ``app.state`` so tests can
swap it out with ``app.dependency_overrides[get_service]``.
"""

from __future__ import annotations

from fastapi import Request

from cxr.config import Settings
from cxr.inference.service import InferenceService


def get_service(request: Request) -> InferenceService:
    return request.app.state.service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings
