from copy import deepcopy

import cv2
import numpy as np
import pytest

from industrial_segpose.config import DEFAULT_CONFIG
from industrial_segpose.measurement.coordinates import (
    CoordinateTransform,
    output_axis_angle_from_cv,
    output_directed_angle_from_cv,
)
from industrial_segpose.measurement.orientation import (
    min_area_rect_orientation,
    pca_orientation,
)
from industrial_segpose.pipeline import SegPosePipeline


def _config(*, resize: int = 0, roi=None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["input"]["resize_max_dimension"] = resize
    config["input"]["roi"] = roi
    config["preprocessing"]["gaussian_blur_kernel"] = 0
    config["morphology"]["open_iterations"] = 0
    config["morphology"]["close_iterations"] = 0
    config["filter"]["min_area_px"] = 20
    return config


def _rectangle_image() -> np.ndarray:
    image = np.zeros((200, 400, 3), np.uint8)
    cv2.rectangle(image, (120, 60), (280, 120), (255, 255, 255), -1)
    return image


@pytest.mark.parametrize(
    ("resize", "roi"),
    [(0, None), (200, None), (0, (80, 40, 240, 120)), (200, (40, 20, 120, 60))],
)
def test_pipeline_restores_original_output_coordinates(resize, roi):
    result = SegPosePipeline(_config(resize=resize, roi=roi)).process(_rectangle_image())
    assert result.image_width == 400
    assert result.image_height == 200
    assert result.object_count == 1
    item = result.objects[0]
    assert item.center_x == pytest.approx(200, abs=2)
    assert item.center_y == pytest.approx(109, abs=2)
    assert item.bbox_xywh == pytest.approx((120, 79, 161, 61), abs=2)
    assert np.asarray(item.contour_points)[:, 0].min() == pytest.approx(120, abs=2)
    assert np.asarray(item.contour_points)[:, 1].min() == pytest.approx(79, abs=2)
    assert np.asarray(item.rotated_box_points)[:, 0].max() == pytest.approx(280, abs=2)
    assert item._mask.shape == (200, 400)


def test_coordinate_transform_round_trip_and_bottom_left_conversion():
    transform = CoordinateTransform(400, 200, 200, 100, (40, 20, 120, 60))
    processed = np.asarray([[0.0, 0.0], [120.0, 60.0], [60.0, 30.0]])
    output = transform.processed_to_output(processed)
    assert np.allclose(output, [[80, 159], [320, 39], [200, 99]])
    assert np.allclose(transform.output_to_processed(output), processed)


def test_mask_restore_places_roi_in_original_raster():
    transform = CoordinateTransform(400, 200, 200, 100, (40, 20, 120, 60))
    mask = np.full((60, 120), 255, np.uint8)
    restored = transform.restore_mask(mask)
    assert restored.shape == (200, 400)
    assert restored[50, 100] == 255
    assert restored[10, 10] == 0


@pytest.mark.parametrize(
    ("raw", "axis", "directed"),
    [(0, 0, 0), (90, 90, 270), (180, 0, 180), (270, 90, 90), (359, 1, 1)],
)
def test_output_angle_conversion_boundaries(raw, axis, directed):
    assert output_axis_angle_from_cv(raw) == pytest.approx(axis)
    assert output_directed_angle_from_cv(raw) == pytest.approx(directed)


@pytest.mark.parametrize("angle_cv", [0.0, 1.0, 89.0, 90.0, 91.0, 179.0])
def test_pca_and_min_area_rect_use_output_axis_convention(angle_cv):
    contour = cv2.boxPoints(((120, 90), (100, 24), angle_cv)).astype(np.int32).reshape(-1, 1, 2)
    expected = (-angle_cv) % 180.0
    rect_angle, _ = min_area_rect_orientation(contour)
    pca_angle, _ = pca_orientation(contour)
    circular = lambda value: min(abs(value - expected), 180.0 - abs(value - expected))
    assert circular(rect_angle) < 2.0
    assert circular(pca_angle) < 2.0


def test_directed_angle_is_null_when_direction_is_unknown():
    result = SegPosePipeline(_config()).process(_rectangle_image())
    item = result.objects[0]
    assert item.axis_angle_deg == item.angle_deg
    assert item.directed_angle_deg is None
    assert item.direction_confidence is None
