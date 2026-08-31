# Foreign Object Detection in Chest X-Rays

Detects foreign objects — coins, buttons, jewellery, wires, surgical items — in frontal chest
radiographs from the [object-CXR](https://web.archive.org/web/20201127235812/https://jfhealthcare.github.io/object-CXR/)
dataset, with two interchangeable detectors behind one interface:

| Detector | Backbone | Input | Notes |
|---|---|---|---|
| **Faster R-CNN** | ResNet50-FPN (torchvision) | 600×600 | Two-stage; better recall on small, low-contrast objects |
| **YOLO** | ultralytics | 480×480 | Single-stage; much faster on CPU, simpler to deploy |

This is a production restructuring of the original research notebook
(`notebooks/foreign-objects-in-chest-x-rays.ipynb`): an installable Python package with a CLI,
a FastAPI service, and a React front end.

> **Research and educational use only. Not a medical device.** Predictions must not be used for
> diagnosis or any clinical decision.

---

## What it looks like

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Foreign Object Detection   Detect  Compare  Batch  Metrics    demo · cpu │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Upload / samples │  X-ray with detection overlay (zoom, pan, hover)      │
│ Detector picker  │                    ┌────┐ 94%                         │
│ Confidence 0.50  │                    └────┘                             │
│ [Detect]         │  Verdict: FOREIGN OBJECT · 94.0% · 1 object · 412 ms  │
└──────────────────┴───────────────────────────────────────────────────────┘
```

* **Detect** — drop an X-ray, pick a detector, drag the confidence slider; boxes are drawn in
  original-image coordinates and stay aligned at any zoom.
* **Compare** — the same image through both detectors, side by side with synchronised zoom.
* **Batch** — screen many images at once, sortable table, CSV export.
* **Metrics** — ROC curve, threshold sweep, confusion matrix, FROC and a model card, read from
  whatever `cxr evaluate` last wrote.

---

## Quick start

```bash
# 1. backend
pip install -e backend            # add [yolo] for ultralytics, [dev] for tests
python -m cxr.cli info            # environment, dataset and checkpoint report
python -m cxr.cli serve           # http://127.0.0.1:8000  (docs at /api/docs)

# 2. front end (second terminal)
cd frontend && npm install && npm run dev    # http://localhost:5173
```

`pip install -e backend` also installs a `cxr` console script; use it in place of
`python -m cxr.cli` if your Python scripts directory is on `PATH`.

**No checkpoints yet?** That is the expected first-run state. The app starts, reports
`demo mode`, and serves recorded detections for five bundled *synthetic phantom* images so the
whole UI is usable. Those responses are tagged `source: "demo"` and badged in the interface —
they are never presented as real inference.

### Docker

```bash
docker compose up --build     # SPA on :8080, API on :8000
```

`./models`, `./artifacts` and `./data` are bind-mounted, so checkpoints you drop into `models/`
are picked up on restart.

---

## Getting a trained model

The dataset (~9 000 images) and training are not bundled. On a GPU machine — Kaggle, Colab, or
your own box:

```bash
# point the config at your copy of the dataset
export CXR_DATA__ROOT=/path/to/object-CXR      # <root>/train, <root>/dev, train.csv, dev.csv

cxr train frcnn --config configs/frcnn.yaml    # ~5 epochs, keeps the best-AUC checkpoint
cxr train yolo  --config configs/yolo.yaml     # exports the YOLO tree, then trains

cxr evaluate --model frcnn                     # writes the CSVs + metrics.json the UI reads
```

A one-minute smoke test without a GPU:

```bash
cxr train frcnn --epochs 1 --limit 20
```

Copy the resulting `model.pt` / `best.pt` into `models/frcnn/` and `models/yolo/`
(see `models/README.md`), restart the API, and the UI leaves demo mode.

---

## Layout

```
backend/cxr/
  config.py            layered settings: defaults -> YAML -> CXR_* env vars
  device.py            auto CPU / CUDA / MPS selection
  data/                annotation parsing, dataset, transforms, YOLO export
  models/              Detector interface + Faster R-CNN and YOLO backends + registry
  training/            train loops, meters, per-model training entry points
  evaluation/          accuracy/AUC/ROC, centre-point localization, FROC, report writer
  inference/           upload validation, model cache, batch + demo mode
  api/                 FastAPI app and routes
  cli.py               cxr info | prepare-yolo | train | evaluate | predict | serve
frontend/src/
  api/                 typed client and DTOs
  lib/scale.ts         image <-> canvas coordinate maths (unit-tested)
  components/          canvas overlay, charts, controls, result panels
  pages/               Detect, Compare, Batch, Metrics
configs/               default.yaml, frcnn.yaml, yolo.yaml
docker/                backend + frontend images, nginx config
scripts/               synthetic demo asset generator
notebooks/             the original research notebook, unchanged
```

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | device, versions, which models are ready |
| GET | `/api/models` | per-detector status (weights present, dependencies, loaded) |
| POST | `/api/models/{name}/load` | warm a model up |
| POST | `/api/predict` | one image → detections, verdict, timing |
| POST | `/api/predict/batch` | many images; `?format=csv` streams a report |
| POST | `/api/compare` | one image through every detector |
| GET | `/api/metrics[/{name}]` | stored evaluation artefacts |
| GET | `/api/samples` | bundled synthetic phantoms |

Interactive docs: <http://127.0.0.1:8000/api/docs>.

### Configuration

Three layers, later wins: defaults in `config.py` → YAML (`--config`, or `CXR_CONFIG`) →
environment variables prefixed `CXR_` (nested keys use `__`, e.g. `CXR_DATA__ROOT`).
Relative paths resolve against the repository root. See `.env.example`.

---

## Development

```bash
pip install -e "backend[dev]"
cd backend && pytest -q          # 49 tests, no dataset or weights required
cd frontend && npm test          # coordinate-mapping unit tests
cd frontend && npm run build     # type-check + production bundle
```

Tests use synthetic fixtures and a stub detector throughout, so CI needs neither the dataset nor
a GPU.

---

## What changed from the notebook

The port is not a copy-paste. Beyond the restructuring, these defects were found and fixed:

1. **YOLO trained without labels.** Labels were written to `/kaggle/working/data/<split>/labels`
   while `data.yaml` pointed `train`/`val` at the read-only Kaggle image folders. Ultralytics
   resolves labels by swapping `/images/` for `/labels/` in the image path, which could not
   match, so the run saw empty targets. `data/yolo_export.py` now builds a real
   `images/{train,val}` + `labels/{train,val}` tree and a `data.yaml` that uses `path:` with
   relative subdirectories.
2. **Wrong checkpoint path.** Predictions loaded `runs/train/.../best.pt`; ultralytics ≥ 8 writes
   to `runs/detect/...`. The trainer's own `save_dir` is used instead.
3. **Crash on negative images.** `torch.as_tensor([])` is 1-D, so `boxes[:, 3]` raised; the
   notebook only avoided it by dropping every un-annotated training image. Boxes are now
   reshaped to `(0, 4)`, and training on positives only is an explicit config flag.
4. **Batch-size-dependent evaluation.** The eval loops indexed `outputs[-1]`, `label[-1]`,
   `width[-1]` — correct only at batch size 1. Both loops are batch-aware.
5. **Shape codes ignored.** Annotations were parsed with `anno[2:]`, assuming a single-character
   shape code and skipping validation. Rectangles, ellipses and polygons are now parsed
   explicitly, and malformed or degenerate entries are dropped instead of producing silent
   garbage coordinates.
6. **Deprecated / unsafe APIs.** `pretrained=True` → weights enum, `torch.load` →
   `weights_only=True`, hardcoded `cuda:0` → auto device, `sys.exit(1)` on divergence → a raised
   exception, and the vendored distributed-training helpers (unused on one GPU) removed.

Localization output keeps the object-CXR submission format (`score x y` triples per image), and
`cxr evaluate` still emits `classification.csv` and `localization.csv` alongside the JSON the
dashboard reads.

## Limitations

* Trained on annotated images only, so the detector never saw a clean radiograph in training.
* The image-level score is the maximum box confidence — one spurious box drives the verdict.
* Frontal radiographs only; behaviour on other views or modalities is untested.
* The bundled sample images are synthetic phantoms, not patient data.
