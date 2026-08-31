from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cxr.api.app import create_app
from cxr.inference.service import InferenceService
from cxr.models.registry import ModelRegistry
from tests.conftest import StubDetector


@pytest.fixture
def client(settings):
    """App with no checkpoints installed - the state a fresh clone starts in."""

    app = create_app(settings=settings, service=InferenceService(settings))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ready_client(settings):
    """App whose Faster R-CNN slot is filled by a stub detector."""

    registry = ModelRegistry(settings)
    registry._detectors["frcnn"] = StubDetector()
    app = create_app(settings=settings, service=InferenceService(settings, registry))
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_environment(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["models_ready"] == []


def test_models_endpoint_lists_both_backends_as_not_ready(client):
    body = client.get("/api/models").json()
    assert [m["name"] for m in body["models"]] == ["frcnn", "yolo"]
    assert all(m["ready"] is False for m in body["models"])
    assert body["any_ready"] is False
    assert body["demo_mode"] is True


def test_predict_without_weights_returns_503_with_a_hint(client, image_bytes):
    response = client.post(
        "/api/predict",
        files={"file": ("scan.png", image_bytes, "image/png")},
        data={"model": "frcnn"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "checkpoint" in detail["detail"]
    assert "cxr train frcnn" in detail["hint"]


def test_bundled_samples_are_listed_and_downloadable(client):
    samples = client.get("/api/samples").json()
    assert len(samples) >= 4
    image = client.get(samples[0]["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_demo_mode_serves_recorded_detections_flagged_as_demo(client):
    body = client.post("/api/predict/sample/phantom-coin?model=frcnn").json()
    assert body["source"] == "demo"
    assert body["detection_count"] == 1
    assert body["has_foreign_object"] is True


def test_demo_mode_respects_the_confidence_slider(client):
    body = client.post("/api/predict/sample/phantom-coin?model=frcnn&conf=0.99").json()
    assert body["detection_count"] == 0
    assert body["has_foreign_object"] is False


def test_unknown_sample_is_404(client):
    assert client.post("/api/predict/sample/nope").status_code == 404


def test_predict_with_a_loaded_model_returns_original_space_boxes(ready_client, image_bytes):
    response = ready_client.post(
        "/api/predict",
        files={"file": ("scan.png", image_bytes, "image/png")},
        data={"model": "frcnn", "conf": "0.5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "model"
    assert body["image_width"] == 300 and body["image_height"] == 200
    box = body["detections"][0]
    assert box["x_max"] == 150.0 and box["y_max"] == 100.0
    assert box["center_x"] == 80.0
    assert body["image_score"] == pytest.approx(0.87)


def test_conf_threshold_filters_detections(ready_client, image_bytes):
    body = ready_client.post(
        "/api/predict",
        files={"file": ("scan.png", image_bytes, "image/png")},
        data={"conf": "0.95"},
    ).json()
    assert body["detection_count"] == 0
    assert body["has_foreign_object"] is False


def test_rejects_a_non_image_upload(ready_client):
    response = ready_client.post(
        "/api/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 422
    assert "not a readable image" in response.json()["detail"]


def test_rejects_an_oversized_upload(ready_client, settings):
    payload = b"\x89PNG\r\n\x1a\n" + b"0" * (settings.max_upload_bytes + 1)
    response = ready_client.post(
        "/api/predict", files={"file": ("big.png", payload, "image/png")}
    )
    assert response.status_code == 422
    assert "limit" in response.json()["detail"]


def test_batch_summarises_and_reports_per_file_errors(ready_client, image_bytes):
    files = [
        ("files", ("a.png", image_bytes, "image/png")),
        ("files", ("b.png", image_bytes, "image/png")),
        ("files", ("c.txt", b"junk", "text/plain")),
    ]
    body = ready_client.post("/api/predict/batch", files=files).json()
    assert body["total"] == 3 and body["positive"] == 2 and body["failed"] == 1
    assert body["items"][2]["ok"] is False


def test_batch_csv_export(ready_client, image_bytes):
    response = ready_client.post(
        "/api/predict/batch?format=csv",
        files=[("files", ("a.png", image_bytes, "image/png"))],
    )
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("image_name,prediction,verdict")
    assert lines[1].startswith("a.png,0.870000,foreign_object")


def test_batch_rejects_too_many_files(ready_client, image_bytes, settings):
    files = [
        ("files", (f"{i}.png", image_bytes, "image/png"))
        for i in range(settings.max_batch_files + 1)
    ]
    assert ready_client.post("/api/predict/batch", files=files).status_code == 422


def test_compare_runs_every_available_detector(ready_client, image_bytes):
    body = ready_client.post(
        "/api/compare", files={"file": ("scan.png", image_bytes, "image/png")}
    ).json()
    # YOLO has no weights, so only the stubbed Faster R-CNN answers
    assert [r["model"] for r in body["results"]] == ["frcnn"]
    assert body["agreement"] == "single"


def test_metrics_404s_until_an_evaluation_has_run(client):
    response = client.get("/api/metrics/frcnn")
    assert response.status_code == 404
    assert "cxr evaluate" in response.json()["detail"]


def test_metrics_are_served_once_written(client, settings):
    path = settings.artifacts_dir / "frcnn"
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps({"accuracy": 0.9, "auc": 0.95, "threshold": 0.5, "n_images": 10}),
        encoding="utf-8",
    )
    assert client.get("/api/metrics/frcnn").json()["auc"] == 0.95
    assert client.get("/api/metrics").json()["models"]["frcnn"]["accuracy"] == 0.9


def test_grayscale_and_odd_sizes_are_accepted(ready_client):
    buffer = io.BytesIO()
    Image.new("L", (37, 91), 128).save(buffer, format="JPEG")
    body = ready_client.post(
        "/api/predict", files={"file": ("gray.jpg", buffer.getvalue(), "image/jpeg")}
    ).json()
    assert body["image_width"] == 37 and body["image_height"] == 91


def test_loading_a_missing_checkpoint_reports_503(client):
    assert client.post("/api/models/frcnn/load").status_code == 503


def test_loading_an_unknown_model_reports_404(client):
    assert client.post("/api/models/resnet/load").status_code == 404
