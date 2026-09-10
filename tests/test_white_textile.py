import json

import cv2
import numpy as np
import pytest

from industrial_segpose.template_matching import (
    MatchParameters,
    TemplateMatcher,
    TemplateModel,
    build_assisted_mask,
    save_white_textile_diagnostics,
    segment_white_textile,
)
from industrial_segpose.cli import main as cli_main


def _mask_iou(first, second):
    first = first > 0
    second = second > 0
    return np.count_nonzero(first & second) / max(np.count_nonzero(first | second), 1)


def _white_cut_scene(height=300, width=420):
    yy, xx = np.indices((height, width))
    illumination = 216.0 + 20.0 * xx / width - 13.0 * yy / height
    image = np.repeat(illumination[:, :, None], 3, axis=2).astype(np.uint8)
    truth = np.zeros((height, width), np.uint8)
    points = np.array(
        [[74, 101], [145, 68], [218, 91], [245, 128], [350, 86],
         [371, 155], [305, 190], [257, 181], [218, 248], [143, 217],
         [86, 239], [104, 171]],
        np.int32,
    )
    cv2.fillPoly(truth, [points], 255)
    # The part and source sheet deliberately share the same colour.  Only a
    # narrow shadowed cut seam identifies the workpiece.
    contour = points.reshape(-1, 1, 2)
    cv2.polylines(image, [contour], True, (118, 118, 118), 4, cv2.LINE_AA)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 2.2, image.shape[:2])
    image = np.clip(image.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    return image, truth


def test_white_cut_seam_segmentation_recovers_same_colour_part():
    image, truth = _white_cut_scene()

    result = segment_white_textile(image)

    assert result.candidate_count >= 1
    assert result.quality_score > 0.55
    assert _mask_iou(result.mask, truth) > 0.86
    assert result.normalized.shape == truth.shape
    assert result.seam_response.shape == truth.shape


def test_white_cut_seam_recovers_short_interrupted_cut_sections():
    image, truth = _white_cut_scene()
    # Simulate glare and loose fibres hiding several short portions of the
    # cutting seam.  The part remains the same colour as the source sheet.
    hidden = np.zeros(truth.shape, np.uint8)
    for point in ((182, 78), (361, 124), (275, 198)):
        cv2.circle(hidden, point, 18, 255, -1)
    local_background = cv2.GaussianBlur(image, (0, 0), 25)
    image[hidden > 0] = local_background[hidden > 0]

    result = segment_white_textile(image)

    assert result.candidate_count >= 1
    assert result.quality_score > 0.55
    assert _mask_iou(result.mask, truth) > 0.82


def test_white_cut_seam_is_available_as_assisted_mask_mode():
    image, truth = _white_cut_scene()

    result = build_assisted_mask(image, "white_cut_seam")

    assert result.method == "white_cut_seam"
    assert _mask_iou(result.mask, truth) > 0.84


def test_white_cut_seam_diagnostics_are_complete(tmp_path):
    image, _ = _white_cut_scene()
    result = segment_white_textile(image)

    destination = save_white_textile_diagnostics(result, tmp_path, image)

    expected = {
        "01_original.png",
        "02_illumination_normalized.png",
        "03_seam_response.png",
        "04_seam_binary.png",
        "05_closed_boundary.png",
        "06_segmented_mask.png",
        "result.json",
    }
    assert {item.name for item in destination.iterdir()} == expected
    summary = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] >= 1


def test_white_diagnose_cli_processes_image_directory(tmp_path):
    image, _ = _white_cut_scene()
    input_path = tmp_path / "sample.png"
    success, encoded = cv2.imencode(".png", image)
    assert success
    input_path.write_bytes(encoded.tobytes())
    output = tmp_path / "diagnostics"

    exit_code = cli_main([
        "white-diagnose", "--input", str(input_path), "--output", str(output),
    ])

    assert exit_code == 0
    assert (output / "001_sample" / "result.json").is_file()


def test_white_cut_seam_template_matching_reports_360_degree_pose():
    template_image, template_mask = _white_cut_scene(220, 320)
    points = cv2.findNonZero(template_mask)
    x, y, width, height = cv2.boundingRect(points)
    local_image = template_image[y : y + height, x : x + width].copy()
    local_mask = template_mask[y : y + height, x : x + width].copy()
    model = TemplateModel("white-cut-part", local_image, mask=local_mask)

    scene = np.full((520, 700, 3), 224, np.uint8)
    angle = 124.0
    center = np.asarray(model.reference_center_xy, np.float32)
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    matrix[0, 2] += 380.0 - center[0]
    matrix[1, 2] += 270.0 - center[1]
    transformed_mask = cv2.warpAffine(
        local_mask, matrix, (scene.shape[1], scene.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    contours, _ = cv2.findContours(transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(scene, contours, -1, (112, 112, 112), 5, cv2.LINE_AA)
    gradient = np.linspace(-12, 15, scene.shape[1], dtype=np.float32)
    scene = np.clip(scene.astype(np.float32) + gradient[None, :, None], 0, 255).astype(np.uint8)

    parameters = MatchParameters(
        score_threshold=0.46,
        angle_min=-180,
        angle_max=178,
        angle_step=2,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=0.05,
        feature_mode="white_cut_seam",
        use_edges=False,
        max_results=2,
        coarse_trigger_transforms=120,
    )
    matches = TemplateMatcher(model, parameters).match(scene)

    assert matches
    best = max(matches, key=lambda item: item.score)
    assert best.center_x == pytest.approx(380.0, abs=5)
    assert best.center_y == pytest.approx(270.0, abs=5)
    assert best.angle_deg == pytest.approx(angle, abs=5)
    assert 0.0 <= best.angle_deg < 360.0


def test_white_cut_seam_template_matching_counts_multiple_parts():
    template_image, template_mask = _white_cut_scene(220, 320)
    x, y, width, height = cv2.boundingRect(cv2.findNonZero(template_mask))
    local_image = template_image[y : y + height, x : x + width].copy()
    local_mask = template_mask[y : y + height, x : x + width].copy()
    model = TemplateModel("white-multi", local_image, mask=local_mask)
    scene = np.full((620, 860, 3), 226, np.uint8)
    reference = np.asarray(model.reference_center_xy, np.float32)
    expected = [(260.0, 220.0, 38.0), (625.0, 405.0, 286.0)]
    for center_x, center_y, angle in expected:
        signed_angle = angle if angle <= 180.0 else angle - 360.0
        matrix = cv2.getRotationMatrix2D(tuple(reference), signed_angle, 1.0)
        matrix[0, 2] += center_x - reference[0]
        matrix[1, 2] += center_y - reference[1]
        transformed = cv2.warpAffine(
            local_mask, matrix, (scene.shape[1], scene.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        contours, _ = cv2.findContours(transformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(scene, contours, -1, (108, 108, 108), 5, cv2.LINE_AA)

    parameters = MatchParameters(
        score_threshold=0.50,
        angle_min=-180,
        angle_max=178,
        angle_step=2,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=0.05,
        feature_mode="white_cut_seam",
        use_edges=False,
        max_results=4,
    )
    matches = TemplateMatcher(model, parameters).match(scene)

    assert len(matches) == 2
    for match, (center_x, center_y, angle) in zip(matches, expected):
        assert match.center_x == pytest.approx(center_x, abs=6)
        assert match.center_y == pytest.approx(center_y, abs=6)
        circular_error = abs((match.angle_deg - angle + 180.0) % 360.0 - 180.0)
        assert circular_error <= 6.0
