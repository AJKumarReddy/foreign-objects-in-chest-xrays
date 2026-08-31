"""Command line entry point: cxr <command>."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cxr import __version__
from cxr.config import build_settings
from cxr.device import device_description, resolve_device
from cxr.logging_conf import configure_logging

app = typer.Typer(
    add_completion=False,
    help="Foreign object detection in chest X-rays: data prep, training, evaluation, serving.",
)
train_app = typer.Typer(help="Train a detector.")
app.add_typer(train_app, name="train")
console = Console()

ConfigOption = typer.Option(None, "--config", "-c", help="Path to a YAML config file.")


def _settings(config: str | None):
    configure_logging()
    return build_settings(config)


@app.command()
def info(config: str | None = ConfigOption) -> None:
    """Report the environment, dataset and checkpoint status."""

    import torch
    import torchvision

    from cxr.models.registry import ModelRegistry
    from cxr.models.yolo_detector import ultralytics_available

    settings = _settings(config)
    device = resolve_device(settings.device)

    console.print(f"[bold]{settings.app_name}[/bold]  v{__version__}")
    console.print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    console.print(f"device: [cyan]{device_description(device)}[/cyan]")
    console.print(
        "ultralytics: "
        + ("[green]installed[/green]" if ultralytics_available() else "[yellow]missing[/yellow]")
    )

    data = settings.data
    ok = data.is_available()
    console.print(
        f"dataset:  {'[green]found[/green]' if ok else '[yellow]not found[/yellow]'}  {data.root}"
    )

    table = Table(title="Detectors", header_style="bold")
    for column in ("model", "weights", "present", "ready"):
        table.add_column(column)
    registry = ModelRegistry(settings)
    for status in registry.statuses():
        table.add_row(
            status.display_name,
            str(status.weights_path),
            "[green]yes[/green]" if status.weights_present else "[red]no[/red]",
            "[green]yes[/green]" if status.ready else "[red]no[/red]",
        )
    console.print(table)
    if not registry.any_ready():
        console.print(
            "\n[yellow]No checkpoints installed.[/yellow] Train one with "
            "[bold]cxr train frcnn[/bold], or copy an existing .pt file to the path above. "
            "The web app still runs in demo mode."
        )


@app.command("prepare-yolo")
def prepare_yolo(
    config: str | None = ConfigOption,
    output: str = typer.Option(None, help="Where to build the YOLO dataset tree."),
    include_negatives: bool = typer.Option(True, help="Emit empty label files for clean images."),
) -> None:
    """Materialise the object-CXR dataset in ultralytics layout."""

    from cxr.training.train_yolo import prepare_dataset

    settings = _settings(config)
    path = prepare_dataset(
        settings, Path(output) if output else None, include_negatives=include_negatives
    )
    console.print(f"[green]data.yaml written to[/green] {path}")


@train_app.command("frcnn")
def train_frcnn_cmd(
    config: str | None = ConfigOption,
    epochs: int = typer.Option(None, help="Override the configured epoch count."),
    limit: int = typer.Option(None, help="Use only N images per split (smoke test)."),
) -> None:
    """Fine-tune Faster R-CNN and keep the best-AUC checkpoint."""

    from cxr.training.train_frcnn import train

    settings = _settings(config)
    result = train(settings, epochs=epochs, limit=limit)
    console.print(
        f"[green]done[/green] best AUC {result.best_auc:.4f} "
        f"(epoch {result.best_epoch}) -> {result.checkpoint}"
    )


@train_app.command("yolo")
def train_yolo_cmd(
    config: str | None = ConfigOption,
    epochs: int = typer.Option(None, help="Override the configured epoch count."),
    data_yaml: str = typer.Option(None, help="Reuse an existing data.yaml."),
) -> None:
    """Train the ultralytics YOLO detector."""

    from cxr.training.train_yolo import train

    settings = _settings(config)
    result = train(settings, epochs=epochs, data_yaml=Path(data_yaml) if data_yaml else None)
    console.print(f"[green]done[/green] {result.checkpoint}")
    console.print(json.dumps(result.metrics, indent=2))


@app.command()
def evaluate(
    model: str = typer.Option("frcnn", "--model", "-m", help="frcnn or yolo."),
    config: str | None = ConfigOption,
    split: str = typer.Option("dev", help="dev or train."),
    limit: int = typer.Option(None, help="Evaluate only the first N images."),
    threshold: float = typer.Option(None, help="Operating point for accuracy/confusion."),
) -> None:
    """Score a checkpoint and write classification.csv, localization.csv, metrics.json."""

    from cxr.evaluation.report import evaluate as run_evaluation

    settings = _settings(config)
    result = run_evaluation(
        settings, model_name=model, split=split, limit=limit, threshold=threshold
    )
    metrics = result.metrics
    console.print(
        f"[bold]{model}[/bold]  accuracy {metrics['accuracy']:.4f}  AUC {metrics['auc']:.4f}"
    )
    console.print(f"artefacts: {result.output_dir}")


@app.command()
def predict(
    image: str = typer.Argument(..., help="Path to a chest X-ray image."),
    model: str = typer.Option("frcnn", "--model", "-m"),
    conf: float = typer.Option(None, help="Confidence threshold."),
    config: str | None = ConfigOption,
) -> None:
    """Run one image through a detector and print the result as JSON."""

    from PIL import Image

    from cxr.inference.schemas import detection_result_to_dto
    from cxr.inference.service import InferenceService

    settings = _settings(config)
    service = InferenceService(settings)
    with Image.open(image) as handle:
        picture = handle.convert("RGB")
    result = service.predict(picture, model=model, conf=conf, filename=Path(image).name)
    payload = detection_result_to_dto(result, Path(image).name)
    console.print_json(payload.model_dump_json(by_alias=True))


@app.command()
def serve(
    config: str | None = ConfigOption,
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (development)."),
) -> None:
    """Start the FastAPI server."""

    import os

    import uvicorn

    if config:
        os.environ["CXR_CONFIG"] = str(Path(config).resolve())
    configure_logging()
    uvicorn.run("cxr.api.app:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
