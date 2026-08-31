# Checkpoints

This directory is git-ignored: trained weights live here but are never committed.

```
models/
├── frcnn/
│   ├── model.pt     # torchvision Faster R-CNN state_dict
│   └── run.json     # written by `cxr train frcnn`
└── yolo/
    ├── best.pt      # ultralytics checkpoint
    └── run.json     # written by `cxr train yolo`
```

## Getting weights

**Train them** (needs the object-CXR dataset and, realistically, a GPU):

```bash
cxr train frcnn --config configs/frcnn.yaml
cxr train yolo  --config configs/yolo.yaml
```

Both commands copy the best checkpoint to the paths above automatically.

**Or bring your own** — drop the files in place and restart the API. Verify with:

```bash
cxr info
```

Paths can be redirected without moving files:

```bash
CXR_FRCNN__WEIGHTS=/somewhere/else/model.pt cxr serve
```

Until a checkpoint exists the API answers `/api/predict` with HTTP 503 and the web
app runs in demo mode against the bundled synthetic phantoms.
