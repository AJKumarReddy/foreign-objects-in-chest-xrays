"""Localization outputs and FROC.

object-CXR scores localization with center points: a prediction hits if its
center falls inside the annotated shape. The submission format is one row per
image, ``score x y`` triples separated by ``;``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from cxr.data.annotations import Shape, parse_annotation
from cxr.models.base import Detection


def format_localization(detections: Sequence[Detection]) -> str:
    """Render detections as the object-CXR localization submission string."""

    parts = []
    for det in detections:
        cx, cy = det.center
        parts.append(f"{det.score} {cx} {cy}")
    return ";".join(parts)


def _shape_mask(annotation: str | None, width: int, height: int) -> np.ndarray | None:
    """Rasterise the annotated shapes so hits can be tested exactly."""

    shapes = parse_annotation(annotation)
    if not shapes:
        return None
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for anno in shapes:
        flat = [coord for point in anno.points for coord in point]
        if anno.shape is Shape.RECTANGLE:
            draw.rectangle(flat, fill=1)
        elif anno.shape is Shape.ELLIPSE:
            draw.ellipse(flat, fill=1)
        else:
            draw.polygon(flat, fill=1)
    return np.array(mask, dtype=bool)


def hits_and_misses(
    detections: Sequence[Detection], annotation: str | None, width: int, height: int
) -> tuple[list[float], list[float], int]:
    """Split detection scores into hits / false positives, plus the object count.

    A detection hits when its center lands inside an annotated shape. Each
    annotated object can only be claimed once (the highest scoring detection
    inside it wins), so duplicates count as false positives.
    """

    shapes = parse_annotation(annotation)
    n_objects = len(shapes)
    mask = _shape_mask(annotation, width, height)
    if mask is None:
        return [], [float(d.score) for d in detections], 0

    claimed: set[int] = set()
    hits: list[float] = []
    false_positives: list[float] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        cx, cy = det.center
        ix, iy = int(round(cx)), int(round(cy))
        inside = 0 <= ix < width and 0 <= iy < height and bool(mask[iy, ix])
        if not inside:
            false_positives.append(float(det.score))
            continue
        # find which object was hit
        index = next(
            (
                i
                for i, anno in enumerate(shapes)
                if i not in claimed
                and anno.bbox[0] <= cx <= anno.bbox[2]
                and anno.bbox[1] <= cy <= anno.bbox[3]
            ),
            None,
        )
        if index is None:
            false_positives.append(float(det.score))
        else:
            claimed.add(index)
            hits.append(float(det.score))
    return hits, false_positives, n_objects


def froc(
    per_image: Sequence[tuple[list[float], list[float], int]],
    fps_per_image: Sequence[float] = (0.125, 0.25, 0.5, 1, 2, 4, 8),
) -> dict:
    """Free-response ROC: sensitivity at fixed average false positives per image.

    ``per_image`` is the output of :func:`hits_and_misses` for every image.
    """

    n_images = len(per_image)
    total_objects = sum(item[2] for item in per_image)
    if not n_images or not total_objects:
        return {"points": [], "mean_sensitivity": float("nan"), "total_objects": total_objects}

    hits = np.array([s for item in per_image for s in item[0]])
    fps = np.array([s for item in per_image for s in item[1]])
    thresholds = np.unique(np.concatenate([hits, fps, np.array([1.0])]))[::-1]

    curve = []
    for target in fps_per_image:
        allowed = target * n_images
        sensitivity = 0.0
        for threshold in thresholds:
            n_fp = int(np.sum(fps >= threshold))
            if n_fp > allowed:
                break
            sensitivity = float(np.sum(hits >= threshold)) / total_objects
        curve.append({"fps_per_image": float(target), "sensitivity": sensitivity})

    return {
        "points": curve,
        "mean_sensitivity": float(np.mean([p["sensitivity"] for p in curve])),
        "total_objects": total_objects,
        "n_images": n_images,
    }
