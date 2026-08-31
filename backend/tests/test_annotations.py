from __future__ import annotations

import pytest

from cxr.data.annotations import (
    AnnotationParseError,
    Shape,
    annotation_to_boxes,
    has_foreign_object,
    parse_annotation,
)


def test_parses_all_three_shape_codes():
    annos = parse_annotation("0 10 20 30 40;1 5 5 15 25;2 0 0 10 0 10 10 0 10")
    assert [a.shape for a in annos] == [Shape.RECTANGLE, Shape.ELLIPSE, Shape.POLYGON]
    assert annos[2].bbox == (0.0, 0.0, 10.0, 10.0)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_annotation_is_a_negative(value):
    assert parse_annotation(value) == []
    assert has_foreign_object(value) is False


def test_polygon_bbox_is_the_extent_of_its_points():
    (anno,) = parse_annotation("2 30 10 5 60 45 25")
    assert anno.bbox == (5.0, 10.0, 45.0, 60.0)


def test_boxes_are_rescaled_into_the_network_input_square():
    boxes = annotation_to_boxes("0 50 100 150 200", width=300, height=400, target_size=600)
    assert boxes == [[100.0, 150.0, 300.0, 300.0]]


def test_degenerate_and_malformed_entries_are_dropped():
    assert annotation_to_boxes("0 10 10 10 10") == []          # zero area
    assert annotation_to_boxes("0 1 2 3") == []                # too few coordinates
    assert annotation_to_boxes("banana 1 2 3 4") == []         # bad shape code
    assert annotation_to_boxes("2 1 2 3 4 5") == []            # odd coordinate count


def test_strict_mode_raises_instead_of_skipping():
    with pytest.raises(AnnotationParseError):
        parse_annotation("0 1 2 3", strict=True)


def test_rescaling_requires_the_original_size():
    with pytest.raises(ValueError):
        annotation_to_boxes("0 1 2 3 4", target_size=600)
