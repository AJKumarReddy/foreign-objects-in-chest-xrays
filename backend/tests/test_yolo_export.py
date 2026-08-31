from __future__ import annotations

import pytest
import yaml

from cxr.data.dataset import labels_to_dict, load_labels
from cxr.data.yolo_export import export_dataset, to_yolo_lines


def test_boxes_are_normalised_centre_width_height():
    (line,) = to_yolo_lines("0 50 100 150 200", width=300, height=400)
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    # label files are written with 6 decimal places
    assert float(cx) == pytest.approx(100 / 300, abs=1e-6)
    assert float(cy) == pytest.approx(150 / 400, abs=1e-6)
    assert float(w) == pytest.approx(100 / 300, abs=1e-6)
    assert float(h) == pytest.approx(100 / 400, abs=1e-6)


def test_out_of_frame_coordinates_are_clipped():
    (line,) = to_yolo_lines("0 -20 -20 400 500", width=300, height=400)
    _, cx, cy, w, h = line.split()
    assert float(w) == 1.0 and float(h) == 1.0
    assert float(cx) == 0.5 and float(cy) == 0.5


def test_export_builds_a_tree_ultralytics_can_resolve(dataset_root, tmp_path):
    """The notebook's data.yaml pointed at image dirs with no sibling labels dir."""

    out = tmp_path / "yolo"
    train = labels_to_dict(load_labels(dataset_root / "train.csv"))
    dev = labels_to_dict(load_labels(dataset_root / "dev.csv"))

    yaml_path, stats = export_dataset(
        dataset_root / "train", train, dataset_root / "dev", dev, out
    )

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert config["train"] == "images/train" and config["val"] == "images/val"
    assert config["nc"] == 1 and config["names"] == ["foreign_object"]

    # every exported image has a label file at the mirrored path
    for image in (out / "images" / "train").iterdir():
        assert (out / "labels" / "train" / f"{image.stem}.txt").is_file()

    train_stats = stats[0]
    assert train_stats.images == 3          # includes the negative
    assert train_stats.labels == 2          # two of them carry objects
    assert train_stats.objects == 3         # 1 + 2 boxes
    # a negative image gets an empty label file, which is how YOLO encodes background
    assert (out / "labels" / "train" / "t2.txt").read_text(encoding="utf-8") == ""


def test_negatives_can_be_excluded(dataset_root, tmp_path):
    out = tmp_path / "yolo_pos"
    train = labels_to_dict(load_labels(dataset_root / "train.csv"))
    dev = labels_to_dict(load_labels(dataset_root / "dev.csv"))
    _, stats = export_dataset(
        dataset_root / "train", train, dataset_root / "dev", dev, out, include_negatives=False
    )
    assert stats[0].images == 2
