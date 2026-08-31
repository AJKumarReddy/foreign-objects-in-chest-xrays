from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from cxr.data.dataset import ForeignObjectDataset, collate_fn, labels_to_dict, load_labels
from cxr.data.transforms import build_transforms


def test_load_labels_can_filter_to_positives(dataset_root):
    everything = load_labels(dataset_root / "train.csv")
    positives = load_labels(dataset_root / "train.csv", positives_only=True)
    assert len(everything) == 3
    assert len(positives) == 2
    assert "t2.jpg" not in set(positives.image_name)


def test_train_mode_returns_detection_targets(dataset_root):
    labels = labels_to_dict(load_labels(dataset_root / "train.csv"))
    dataset = ForeignObjectDataset(
        dataset_root / "train", labels, mode="train",
        transform=build_transforms(64), image_size=64,
    )
    image, target = dataset[1]  # t1.jpg has a polygon and an ellipse
    assert image.shape == (3, 64, 64)
    assert target["boxes"].shape == (2, 4)
    assert torch.all(target["labels"] == 1)
    assert target["area"].shape == (2,)


def test_negative_image_yields_an_empty_but_2d_box_tensor(dataset_root):
    """The notebook crashed here: a bare empty tensor is 1-D and boxes[:, 3] fails."""

    labels = labels_to_dict(load_labels(dataset_root / "train.csv"))
    dataset = ForeignObjectDataset(
        dataset_root / "train", labels, mode="train",
        transform=build_transforms(64), image_size=64,
    )
    _, target = dataset[2]  # t2.jpg is un-annotated
    assert target["boxes"].shape == (0, 4)
    assert target["area"].shape == (0,)


def test_dev_mode_returns_image_level_labels_and_original_size(dataset_root):
    labels = labels_to_dict(load_labels(dataset_root / "dev.csv"))
    dataset = ForeignObjectDataset(
        dataset_root / "dev", labels, mode="dev",
        transform=build_transforms(64), image_size=64,
    )
    _, label_pos, width, height = dataset[0]
    _, label_neg, _, _ = dataset[1]
    assert (label_pos, label_neg) == (1, 0)
    assert (width, height) == (200, 150)


def test_collate_keeps_variable_length_targets(dataset_root):
    labels = labels_to_dict(load_labels(dataset_root / "train.csv"))
    dataset = ForeignObjectDataset(
        dataset_root / "train", labels, mode="train",
        transform=build_transforms(64), image_size=64,
    )
    images, targets = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_fn)))
    assert len(images) == len(targets) == 3
    assert {t["boxes"].shape[0] for t in targets} == {1, 2, 0}
