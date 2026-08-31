"""FastAPI application factory."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cxr import __version__
from cxr.api.routes import health, metrics, models, predict, samples
from cxr.config import Settings, get_settings
from cxr.inference.service import ImageValidationError, InferenceService, ModelNotReadyError
from cxr.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Detect foreign objects (coins, buttons, jewellery, wires, surgical items) in frontal
chest radiographs.

Two interchangeable detectors are exposed: a torchvision **Faster R-CNN** and an
**ultralytics YOLO** model, both trained on the object-CXR dataset.

Research and educational use only - not a medical device.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    service: InferenceService = app.state.service
    ready = [s.name for s in service.statuses() if s.ready]
    logger.info("device=%s ready models=%s", service.device, ready or "none")
    if not ready:
        logger.warning(
            "no checkpoints found - the API will serve demo predictions for bundled "
            "samples only. Train with 'cxr train frcnn' or drop weights into %s",
            settings.models_dir,
        )
    yield
    service.registry.unload_all()


def create_app(
    settings: Settings | None = None, service: InferenceService | None = None
) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.service = service or InferenceService(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(ImageValidationError)
    async def _image_error(_: Request, exc: ImageValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelNotReadyError)
    async def _model_error(_: Request, exc: ModelNotReadyError):
        return JSONResponse(
            status_code=503, content={"detail": str(exc), "hint": getattr(exc, "hint", None)}
        )

    for router in (health.router, models.router, predict.router, metrics.router, samples.router):
        app.include_router(router, prefix="/api")

    # Serve the built SPA when it exists (single-container deployment).
    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")
        logger.info("serving front end from %s", dist)

    return app


app = create_app()
