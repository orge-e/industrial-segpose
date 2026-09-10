import cv2
import numpy as np
import pytest

from industrial_segpose.template_matching import METHOD_LABELS, analyze_template_mask, build_assisted_mask, locate_assisted_template


def synthetic_irregular(background, foreground):
    image = np.full((180, 240, 3), background, np.uint8)
    truth = np.zeros(image.shape[:2], np.uint8)
    points = np.array([[45, 28], [185, 40], [165, 85], [205, 142], [110, 132], [55, 155], [68, 88]], np.int32)
    cv2.fillPoly(truth, [points], 255)
    image[truth > 0] = foreground
    cv2.circle(image, (110, 75), 15, (120, 120, 120), -1)
    return image, truth


def mask_iou(first, second):
    first, second = first > 0, second > 0
    return np.count_nonzero(first & second) / max(np.count_nonzero(first | second), 1)


@pytest.mark.parametrize(
    ("background", "foreground"),
    [((35, 35, 35), (220, 220, 220)), ((220, 220, 220), (30, 30, 30)), ((80, 120, 180), (190, 70, 60))],
)
def test_auto_assisted_mask_extracts_irregular_target(background, foreground):
    image, truth = synthetic_irregular(background, foreground)
    result = build_assisted_mask(image, "auto")
    assert result.method in METHOD_LABELS
    assert 0.1 < result.coverage < 0.8
    assert result.quality_score > 0.5
    assert mask_iou(result.mask, truth) > 0.95


def test_explicit_light_and_dark_otsu_modes():
    light_image, light_truth = synthetic_irregular((25, 25, 25), (230, 230, 230))
    dark_image, dark_truth = synthetic_irregular((230, 230, 230), (25, 25, 25))
    assert mask_iou(build_assisted_mask(light_image, "otsu_light").mask, light_truth) > 0.95
    assert mask_iou(build_assisted_mask(dark_image, "otsu_dark").mask, dark_truth) > 0.95


def test_assisted_mask_rejects_invalid_input():
    with pytest.raises(ValueError):
        build_assisted_mask(np.zeros((5, 5, 3), np.uint8))
    with pytest.raises(ValueError):
        build_assisted_mask(np.zeros((20, 20, 3), np.uint8), "unknown")


def test_textile_chroma_extracts_low_contrast_pink_shape():
    image = np.full((180, 280, 3), (190, 190, 190), np.uint8)
    truth = np.zeros(image.shape[:2], np.uint8)
    points = np.array([[25, 55], [115, 42], [145, 72], [245, 50], [255, 100], [160, 105], [130, 145], [70, 135]], np.int32)
    cv2.fillPoly(truth, [points], 255)
    image[truth > 0] = (185, 165, 205)
    cv2.line(image, (45, 75), (220, 85), (70, 55, 170), 4)
    result = build_assisted_mask(image, "textile_chroma")
    assert result.method == "textile_chroma"
    assert mask_iou(result.mask, truth) > 0.95


def test_textile_chroma_is_direction_independent_for_pale_yellow_shape():
    image = np.full((220, 300, 3), (132, 132, 132), np.uint8)
    truth = np.zeros(image.shape[:2], np.uint8)
    points = np.array(
        [[55, 38], [205, 42], [235, 90], [217, 176], [158, 190],
         [118, 171], [66, 185], [42, 118]],
        np.int32,
    )
    cv2.fillPoly(truth, [points], 255)
    image[truth > 0] = (142, 151, 162)

    result = build_assisted_mask(image, "textile_chroma")

    assert mask_iou(result.mask, truth) > 0.95


def test_dark_textile_texture_extracts_black_part_from_dark_surface():
    gradient = np.linspace(32, 72, 480, dtype=np.uint8)
    image = np.repeat(gradient[None, :, None], 360, axis=0)
    image = np.repeat(image, 3, axis=2)
    truth = np.zeros(image.shape[:2], np.uint8)
    points = np.array([[75, 95], [180, 70], [245, 110], [390, 90], [405, 205],
                       [330, 260], [255, 245], [205, 305], [95, 285], [115, 190]], np.int32)
    cv2.fillPoly(truth, [points], 255)
    yy, xx = np.indices(truth.shape)
    weave = np.where(((xx // 3 + yy // 3) % 2) == 0, 32, 84).astype(np.uint8)
    for channel in range(3):
        image[:, :, channel][truth > 0] = weave[truth > 0]
    rng = np.random.default_rng(11)
    noise_y = rng.integers(0, image.shape[0], 900)
    noise_x = rng.integers(0, image.shape[1], 900)
    image[noise_y, noise_x] = rng.choice([0, 255], 900)[:, None]
    result = build_assisted_mask(image, "dark_textile")
    assert result.method == "dark_textile"
    assert mask_iou(result.mask, truth) > 0.88


def test_full_image_auto_location_ignores_dark_border_and_returns_local_mask():
    image = np.full((300, 460, 3), (195, 195, 195), np.uint8)
    image[:28] = 25
    image[-24:] = 18
    truth = np.zeros(image.shape[:2], np.uint8)
    points = np.array([[70, 105], [180, 82], [225, 122], [380, 96], [405, 165],
                       [270, 180], [235, 238], [145, 218], [82, 245], [105, 170]], np.int32)
    cv2.fillPoly(truth, [points], 255)
    image[truth > 0] = (184, 162, 215)
    selection = locate_assisted_template(image, "textile_chroma")
    x, y, width, height = selection.roi_xywh
    assert y > 28
    assert y + height < image.shape[0] - 24
    assert selection.mask.shape == (height, width)
    restored = np.zeros_like(truth)
    restored[y : y + height, x : x + width] = selection.mask
    assert mask_iou(restored, truth) > 0.94


def test_template_mask_quality_reports_fragmentation_and_border_contact():
    mask = np.zeros((100, 140), np.uint8)
    cv2.rectangle(mask, (0, 15), (45, 80), 255, -1)
    cv2.rectangle(mask, (90, 25), (130, 75), 255, -1)

    quality = analyze_template_mask(mask)

    assert not quality.valid
    assert quality.component_count == 2
    assert quality.touches_border
    assert "分离区域" in quality.message
    assert "边界" in quality.message
