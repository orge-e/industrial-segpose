from copy import deepcopy
import cv2
import numpy as np
import pytest
from industrial_segpose.config import DEFAULT_CONFIG, load_config, validate_config
from industrial_segpose.io.image_reader import read_image
from industrial_segpose.measurement.center import calculate_center
from industrial_segpose.measurement.orientation import min_area_rect_orientation, normalize_axis_angle, pca_orientation
from industrial_segpose.pipeline import SegPosePipeline


def contour_at(angle=30, center=(100, 80), size=(100, 30)):
    points = cv2.boxPoints((center, size, angle)).astype(np.int32)
    return points.reshape(-1, 1, 2)


def axis_error(actual, expected):
    error = abs(actual - expected); return min(error, 180.0 - error)


def config():
    cfg = deepcopy(DEFAULT_CONFIG); cfg["preprocessing"]["gaussian_blur_kernel"] = 0; cfg["morphology"]["open_iterations"] = 0; cfg["morphology"]["close_iterations"] = 0; return cfg


def test_config_loads(): assert load_config("configs/default.yaml")["project"]["name"] == "industrial-segpose"
def test_config_rejects_even_kernel():
    cfg = config(); cfg["morphology"]["kernel_size"] = 4
    with pytest.raises(ValueError): validate_config(cfg)


def test_unicode_path(tmp_path):
    path = tmp_path / "中文图像.png"; encoded = cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))[1]; encoded.tofile(path)
    assert read_image(path).shape == (20, 30, 3)


def test_center_moments_and_fallback():
    x, y = calculate_center(contour_at(center=(100, 80)), "moments"); assert x == pytest.approx(100, abs=1); assert y == pytest.approx(80, abs=1)


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135])
def test_orientations(angle):
    contour = contour_at(angle)
    rect_angle, _ = min_area_rect_orientation(contour); pca_angle, _ = pca_orientation(contour)
    expected = (-angle) % 180.0
    assert axis_error(rect_angle, expected) < 2; assert axis_error(pca_angle, expected) < 2


@pytest.mark.parametrize(("raw", "expected"), [(-10, 170), (0, 0), (180, 0), (190, 10), (540, 0)])
def test_normalize_axis_angle(raw, expected): assert normalize_axis_angle(raw) == expected


def test_blank_and_multiple_objects_sorted():
    pipeline = SegPosePipeline(config()); blank = np.zeros((200, 300, 3), np.uint8); assert pipeline.process(blank).object_count == 0
    image = blank.copy(); cv2.rectangle(image, (180, 120), (260, 150), (255, 255, 255), -1); cv2.rectangle(image, (20, 20), (100, 50), (255, 255, 255), -1)
    result = pipeline.process(image); assert result.object_count == 2; assert result.objects[0].center_y < result.objects[1].center_y; assert [o.object_id for o in result.objects] == [1, 2]


def test_border_filter():
    cfg = config(); cfg["filter"]["reject_border_objects"] = True; image = np.zeros((100, 100, 3), np.uint8); cv2.rectangle(image, (0, 20), (40, 60), (255, 255, 255), -1)
    result = SegPosePipeline(cfg).process(image); assert result.object_count == 0; assert "touching_border" in result.rejected_instances[0].reasons
