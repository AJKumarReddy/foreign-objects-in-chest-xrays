from __future__ import annotations

import math

import pytest
from sklearn.metrics import roc_auc_score

from cxr.evaluation.classification import (
    accuracy_at,
    classification_report,
    confusion_at,
    roc_points,
)
from cxr.evaluation.localization import format_localization, froc, hits_and_misses
from cxr.models.base import Detection, DetectionResult, rescale_boxes

SCORES = [0.05, 0.2, 0.45, 0.55, 0.8, 0.95, 0.3, 0.7]
LABELS = [0, 0, 0, 1, 1, 1, 1, 0]


def test_accuracy_matches_a_hand_count():
    # >= 0.5 predicts positive: correct for 0.05,0.2,0.45 (TN), 0.55,0.8,0.95 (TP)
    assert accuracy_at(SCORES, LABELS, 0.5) == pytest.approx(6 / 8)


def test_auc_matches_sklearn():
    assert roc_points(SCORES, LABELS)["auc"] == pytest.approx(roc_auc_score(LABELS, SCORES))


def test_confusion_counts_and_derived_rates():
    stats = confusion_at(SCORES, LABELS, 0.5)
    assert (stats["tp"], stats["fp"], stats["tn"], stats["fn"]) == (3, 1, 3, 1)
    assert stats["sensitivity"] == pytest.approx(0.75)
    assert stats["specificity"] == pytest.approx(0.75)


def test_report_is_json_serialisable_and_complete():
    report = classification_report(SCORES, LABELS, threshold=0.5)
    assert report["n_images"] == 8 and report["n_positive"] == 4
    assert 0.0 <= report["roc"]["best_threshold"] <= 1.0
    assert len(report["sweep"]) == 21
    import json

    json.dumps(report)


def test_rescale_boxes_maps_back_to_original_pixels():
    import torch

    boxes = torch.tensor([[0.0, 0.0, 300.0, 600.0]])
    out = rescale_boxes(boxes, (600, 600), (1200, 300))
    assert out.tolist() == [[0.0, 0.0, 600.0, 300.0]]


def test_rescale_handles_no_detections():
    import torch

    assert rescale_boxes(torch.empty(0), (600, 600), (100, 100)).shape == (0, 4)


def test_image_score_is_the_best_box():
    result = DetectionResult(
        model="frcnn",
        detections=[Detection((0, 0, 1, 1), 0.3), Detection((2, 2, 3, 3), 0.71)],
        conf_threshold=0.5,
    )
    assert result.image_score == pytest.approx(0.71)
    assert result.has_foreign_object is True


def test_localization_string_is_score_x_y_triples():
    detections = [Detection((0, 0, 10, 20), 0.9), Detection((10, 10, 30, 30), 0.4)]
    assert format_localization(detections) == "0.9 5.0 10.0;0.4 20.0 20.0"


def test_hits_are_decided_by_centre_inside_the_annotated_shape():
    detections = [
        Detection((8, 8, 12, 12), 0.9),     # centre (10, 10) inside the rectangle
        Detection((90, 90, 98, 98), 0.6),   # centre (94, 94) outside
    ]
    hits, false_positives, n_objects = hits_and_misses(detections, "0 0 0 20 20", 100, 100)
    assert hits == [0.9] and false_positives == [0.6] and n_objects == 1


def test_duplicate_detections_on_one_object_count_once():
    detections = [Detection((8, 8, 12, 12), 0.9), Detection((9, 9, 11, 11), 0.8)]
    hits, false_positives, _ = hits_and_misses(detections, "0 0 0 20 20", 100, 100)
    assert hits == [0.9] and false_positives == [0.8]


def test_froc_reports_sensitivity_at_each_operating_point():
    per_image = [([0.9], [0.2], 1), ([], [0.4], 1), ([0.8], [], 1)]
    curve = froc(per_image)
    assert curve["total_objects"] == 3
    assert [p["fps_per_image"] for p in curve["points"]][0] == 0.125
    assert all(0.0 <= p["sensitivity"] <= 1.0 for p in curve["points"])
    assert not math.isnan(curve["mean_sensitivity"])
