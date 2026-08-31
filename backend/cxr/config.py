"""Application configuration.

Settings come from three layers, later layers winning:

1. defaults declared here,
2. a YAML file (``configs/default.yaml``, or whatever ``CXR_CONFIG`` points at),
3. environment variables prefixed with ``CXR_`` (or a ``.env`` file).

Every path in the notebook that was hardcoded to ``/kaggle/input/...`` lives here now.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../Foreign_Objects_in_Chest_X_Rays (this file is backend/cxr/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    """Make a configured path absolute, relative to the repository root."""

    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


class DataSettings(BaseSettings):
    """Where the object-CXR dataset lives and how it is read."""

    root: Path = Field(default=REPO_ROOT / "data" / "object-CXR")
    train_dir: str = "train"
    dev_dir: str = "dev"
    train_csv: str = "train.csv"
    dev_csv: str = "dev.csv"
    image_size: int = 600
    #: The notebook trained on annotated images only (cell 9). Kept, but explicit.
    train_on_positives_only: bool = True
    num_workers: int = 0
    batch_size: int = 8

    @property
    def train_images(self) -> Path:
        return self.root / self.train_dir

    @property
    def dev_images(self) -> Path:
        return self.root / self.dev_dir

    @property
    def train_labels(self) -> Path:
        return self.root / self.train_csv

    @property
    def dev_labels(self) -> Path:
        return self.root / self.dev_csv

    def is_available(self) -> bool:
        return self.train_images.is_dir() and self.train_labels.is_file()

    @field_validator("root", mode="after")
    @classmethod
    def _abs_root(cls, v: Path) -> Path:
        return _resolve(v)


class ModelSettings(BaseSettings):
    """Checkpoint locations and inference defaults for one detector."""

    weights: Path
    input_size: int
    conf_threshold: float = 0.5
    max_detections: int = 100

    @field_validator("weights", mode="after")
    @classmethod
    def _abs_weights(cls, v: Path) -> Path:
        return _resolve(v)


class TrainingSettings(BaseSettings):
    epochs: int = 5
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    print_freq: int = 50
    seed: int = 0
    # YOLO specific
    yolo_base_weights: str = "yolov8n.pt"
    yolo_epochs: int = 10
    yolo_imgsz: int = 480
    yolo_batch: int = 16


class Settings(BaseSettings):
    """Top level settings object; use :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="CXR_",
        env_nested_delimiter="__",
        env_file=str(REPO_ROOT / ".env"),
        extra="ignore",
    )

    app_name: str = "Foreign Object Detection in Chest X-Rays"
    environment: str = "development"
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps

    models_dir: Path = REPO_ROOT / "models"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    runs_dir: Path = REPO_ROOT / "artifacts" / "runs"

    frcnn: ModelSettings = ModelSettings(
        weights=REPO_ROOT / "models" / "frcnn" / "model.pt", input_size=600
    )
    yolo: ModelSettings = ModelSettings(
        weights=REPO_ROOT / "models" / "yolo" / "best.pt", input_size=480
    )

    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_bytes: int = 20 * 1024 * 1024
    max_batch_files: int = 32
    #: Serve bundled canned predictions when no checkpoint is installed.
    demo_mode_enabled: bool = True

    @field_validator("models_dir", "artifacts_dir", "runs_dir", mode="after")
    @classmethod
    def _abs_dirs(cls, v: Path) -> Path:
        return _resolve(v)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_settings(config_path: Path | str | None = None) -> Settings:
    """Build settings from a YAML file (plus env overrides)."""

    candidate = (
        config_path or os.environ.get("CXR_CONFIG") or REPO_ROOT / "configs" / "default.yaml"
    )
    payload = load_yaml_config(candidate)
    if payload:
        base = Settings().model_dump()
        payload = _deep_merge(base, payload)
        return Settings(**payload)
    return Settings()


@lru_cache(maxsize=8)
def get_settings(config_path: str | None = None) -> Settings:
    """Cached settings accessor used by the API and CLI."""

    return build_settings(config_path)
