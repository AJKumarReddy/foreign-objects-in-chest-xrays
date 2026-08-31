"""Generate the bundled demo assets.

These are SYNTHETIC chest-radiograph phantoms - not patient data. They exist so
the whole UI (upload, overlay, compare, batch) can be exercised before any real
checkpoint is installed. Recorded "detections" are the ground truth of the
phantoms we draw, jittered slightly per model, and are always served tagged
``source="demo"``.

Run:  python scripts/generate_demo_samples.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "backend" / "cxr" / "assets" / "samples"
MANIFEST = ROOT / "backend" / "cxr" / "assets" / "demo.json"

W = H = 1024


def _thorax_field() -> np.ndarray:
    """A smooth, radiograph-like intensity field: dark lungs, bright mediastinum."""

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (xx - W / 2) / (W / 2)
    y = (yy - H / 2) / (H / 2)

    base = 0.30 + 0.10 * np.exp(-(y + 0.7) ** 2 * 6)          # apex glow
    spine = 0.26 * np.exp(-(x**2) / 0.010) + 0.10 * np.exp(-(x**2) / 0.05)  # mediastinum
    lungs = -0.20 * (
        np.exp(-((x + 0.42) ** 2) / 0.055 - ((y + 0.05) ** 2) / 0.18)
        + np.exp(-((x - 0.42) ** 2) / 0.055 - ((y + 0.05) ** 2) / 0.18)
    )
    diaphragm = 0.30 * (1 / (1 + np.exp(-(y - 0.45) * 10)))
    shoulders = 0.18 * np.exp(-((abs(x) - 0.88) ** 2) / 0.02 - ((y + 0.8) ** 2) / 0.10)
    vignette = -0.18 * (x**2 + y**2)

    field = base + spine + lungs + diaphragm + shoulders + vignette
    return np.clip(field, 0.02, 1.0)


def _add_ribs(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image, "L")
    for side in (-1, 1):
        for i in range(9):
            top = 180 + i * 68
            span = 300 + i * 22
            box = [
                W / 2 + side * 20 - (0 if side > 0 else span),
                top - span * 0.55,
                W / 2 + side * 20 + (span if side > 0 else 0),
                top + span * 0.55,
            ]
            start, end = (300, 20) if side > 0 else (160, 240)
            draw.arc([min(box[0], box[2]), box[1], max(box[0], box[2]), box[3]],
                     start=start, end=end, fill=165, width=6)
    return image.filter(ImageFilter.GaussianBlur(3.0))


def _base_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    field = _thorax_field()
    noise = rng.normal(0.0, 0.018, size=field.shape).astype(np.float32)
    arr = np.clip((field + noise) * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    image = _add_ribs(image)
    return image.convert("RGB")


def _draw_coin(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> list[float]:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250))
    draw.ellipse([cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5], outline=(215, 215, 215), width=3)
    return [cx - r, cy - r, cx + r, cy + r]


def _draw_button(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> list[float]:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(238, 238, 238))
    for dx, dy in ((-r // 3, -r // 3), (r // 3, -r // 3), (-r // 3, r // 3), (r // 3, r // 3)):
        draw.ellipse([cx + dx - 4, cy + dy - 4, cx + dx + 4, cy + dy + 4], fill=(70, 70, 70))
    return [cx - r, cy - r, cx + r, cy + r]


def _draw_necklace(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> list[float]:
    points = []
    for i in range(28):
        angle = math.pi * (0.08 + 0.84 * i / 27)
        points.append((cx + r * math.cos(angle) * 1.05, cy + r * math.sin(angle) * 0.55))
    for px, py in points:
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(245, 245, 245))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6]


def _draw_wire(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> list[float]:
    points = []
    steps = 40
    for i in range(steps + 1):
        t = i / steps
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t + 26 * math.sin(t * math.pi * 2.4)
        points.append((px, py))
    draw.line(points, fill=(252, 252, 252), width=7, joint="curve")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6]


def _draw_safety_pin(draw: ImageDraw.ImageDraw, cx: int, cy: int, w: int) -> list[float]:
    h = w // 3
    draw.rounded_rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                           radius=h // 2, outline=(250, 250, 250), width=6)
    draw.ellipse([cx - w // 2 - 4, cy - h // 2 - 4, cx - w // 2 + 16, cy + h // 2 + 4],
                 outline=(250, 250, 250), width=5)
    return [cx - w // 2 - 8, cy - h // 2 - 8, cx + w // 2 + 8, cy + h // 2 + 8]


SPECS = [
    {
        "id": "phantom-coin",
        "title": "Coin over right lung field",
        "description": "Synthetic phantom with a single high-density circular object.",
        "seed": 11,
        "objects": [("coin", (360, 430, 46))],
    },
    {
        "id": "phantom-necklace",
        "title": "Necklace and buttons",
        "description": "Synthetic phantom with a chain across the upper thorax plus two buttons.",
        "seed": 23,
        "objects": [("necklace", (512, 250, 210)), ("button", (430, 640, 34)),
                    ("button", (615, 700, 30))],
    },
    {
        "id": "phantom-wire",
        "title": "Monitoring lead across the chest",
        "description": "Synthetic phantom with an elongated wire-like object.",
        "seed": 37,
        "objects": [("wire", (170, 520, 880, 610))],
    },
    {
        "id": "phantom-clear",
        "title": "No foreign object",
        "description": "Synthetic phantom with nothing overlying the chest - the negative case.",
        "seed": 5,
        "objects": [],
    },
    {
        "id": "phantom-pin",
        "title": "Safety pin, low contrast",
        "description": "Synthetic phantom with a small object near the diaphragm.",
        "seed": 41,
        "objects": [("pin", (600, 760, 120))],
    },
]

DRAWERS = {
    "coin": _draw_coin,
    "button": _draw_button,
    "necklace": _draw_necklace,
    "wire": _draw_wire,
    "pin": _draw_safety_pin,
}


def jitter(box: list[float], rng: random.Random, amount: float) -> list[float]:
    """Perturb a ground-truth box so the two demo models do not look identical."""

    w = box[2] - box[0]
    h = box[3] - box[1]
    return [
        box[0] + rng.uniform(-amount, amount) * w,
        box[1] + rng.uniform(-amount, amount) * h,
        box[2] + rng.uniform(-amount, amount) * w,
        box[3] + rng.uniform(-amount, amount) * h,
    ]


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    samples = []

    for spec in SPECS:
        image = _base_image(spec["seed"])
        draw = ImageDraw.Draw(image)
        boxes = [DRAWERS[kind](draw, *args) for kind, args in spec["objects"]]
        image = image.filter(ImageFilter.GaussianBlur(0.6))

        filename = f"{spec['id']}.png"
        image.save(SAMPLES_DIR / filename, format="PNG", optimize=True)

        detections = {}
        for model, conf_hi, conf_lo, amount in (
            ("frcnn", 0.94, 0.71, 0.03),
            ("yolo", 0.89, 0.58, 0.05),
        ):
            recorded = []
            for i, box in enumerate(boxes):
                score = conf_hi if i == 0 else max(conf_lo, conf_hi - 0.12 * i)
                recorded.append(
                    {"box": [round(v, 1) for v in jitter(box, rng, amount)],
                     "score": round(score, 3)}
                )
            detections[model] = recorded

        # the YOLO demo misses the low-contrast pin, mirroring the usual trade-off
        if spec["id"] == "phantom-pin":
            detections["yolo"] = []

        samples.append(
            {
                "id": spec["id"],
                "filename": filename,
                "title": spec["title"],
                "description": spec["description"],
                "width": W,
                "height": H,
                "synthetic": True,
                "detections": detections,
            }
        )
        print(f"wrote {filename} ({len(boxes)} objects)")

    MANIFEST.write_text(
        json.dumps(
            {
                "note": (
                    "Synthetic phantoms and recorded detections used for demo mode only. "
                    "Not patient data and not real model output."
                ),
                "samples": samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
