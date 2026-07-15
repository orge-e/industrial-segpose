import cv2
import numpy as np
import pytest

from industrial_segpose.template_matching import MatchParameters, TemplateMatcher, TemplateModel
from industrial_segpose.template_matching.matcher import _rotate_expanded, _rotation_geometry, draw_template_matches


def asymmetric_template():
    image = np.zeros((50, 70, 3), np.uint8)
    cv2.rectangle(image, (5, 8), (60, 18), (255, 255, 255), -1)
    cv2.rectangle(image, (5, 8), (16, 43), (255, 255, 255), -1)
    return image


def paste_rotated(scene, template, center, angle):
    rotated = _rotate_expanded(template, angle, 0)
    height, width = rotated.shape[:2]
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)
    scene[y : y + height, x : x + width] = np.maximum(
        scene[y : y + height, x : x + width], rotated
    )


def test_template_save_and_load(tmp_path):
    source = np.zeros((100, 120, 3), np.uint8)
    source[20:70, 30:100] = asymmetric_template()
    model = TemplateModel.from_roi(source, (30, 20, 70, 50), "L-target", "参考图.png")
    metadata = model.save(tmp_path / "模板.json")
    loaded = TemplateModel.load(metadata)
    assert loaded.name == "L-target"
    assert loaded.roi_xywh == (30, 20, 70, 50)
    assert np.array_equal(loaded.image, model.image)
    assert np.array_equal(loaded.mask, model.mask)


def test_irregular_mask_center_and_persistence(tmp_path):
    image = asymmetric_template()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[8:44, 5:17] = 255
    mask[8:19, 5:61] = 255
    model = TemplateModel("irregular-L", image, mask=mask)
    center_x, center_y = model.reference_center_xy
    assert center_x < image.shape[1] / 2
    assert center_y < image.shape[0] / 2
    loaded = TemplateModel.load(model.save(tmp_path / "irregular.json"))
    assert loaded.reference_center_xy == pytest.approx(model.reference_center_xy)
    assert loaded.contour.shape[0] >= 6


def test_irregular_mask_rotated_match_uses_mask_centroid():
    template = asymmetric_template()
    mask = np.zeros(template.shape[:2], np.uint8)
    cv2.rectangle(mask, (5, 8), (60, 18), 255, -1)
    cv2.rectangle(mask, (5, 8), (16, 43), 255, -1)
    model = TemplateModel("irregular-L", template, mask=mask)
    angle = 24
    rotated = _rotate_expanded(template, angle, 0)
    scene = np.zeros((220, 300, 3), np.uint8)
    left, top = 120, 70
    height, width = rotated.shape[:2]
    scene[top : top + height, left : left + width] = rotated
    matrix, _, _ = _rotation_geometry(template.shape[1], template.shape[0], angle)
    expected = cv2.transform(np.asarray([[model.reference_center_xy]], np.float32), matrix).reshape(2)
    params = MatchParameters(score_threshold=0.6, angle_min=20, angle_max=28, angle_step=2, use_edges=True)
    matches = TemplateMatcher(model, params).match(scene)
    assert len(matches) == 1
    assert matches[0].template_name == "irregular-L"
    assert matches[0].center_x == pytest.approx(left + expected[0], abs=2)
    assert matches[0].center_y == pytest.approx(top + expected[1], abs=2)
    assert matches[0].angle_deg == pytest.approx(angle, abs=2)
    assert len(matches[0].contour_points) >= 3


def test_rotated_multi_object_matching():
    template = asymmetric_template()
    scene = np.zeros((260, 360, 3), np.uint8)
    paste_rotated(scene, template, (90, 80), 30)
    paste_rotated(scene, template, (260, 170), -44)
    parameters = MatchParameters(
        score_threshold=0.55,
        angle_min=-60,
        angle_max=60,
        angle_step=2,
        use_edges=True,
        nms_iou_threshold=0.3,
    )
    matches = TemplateMatcher(TemplateModel("L-target", template), parameters).match(scene)
    assert len(matches) == 2
    assert matches[0].center_x == pytest.approx(90, abs=2)
    assert matches[0].center_y == pytest.approx(80, abs=2)
    assert matches[0].angle_deg == pytest.approx(30, abs=2)
    assert matches[1].center_x == pytest.approx(260, abs=2)
    assert matches[1].center_y == pytest.approx(170, abs=2)
    assert matches[1].angle_deg == pytest.approx(-44, abs=2)
    assert draw_template_matches(scene, matches).shape == scene.shape


def test_match_parameter_validation():
    with pytest.raises(ValueError):
        MatchParameters(angle_step=0).validate()
    with pytest.raises(ValueError):
        MatchParameters(score_threshold=1.2).validate()


def test_template_variants_are_cached():
    matcher = TemplateMatcher(
        TemplateModel("cache", asymmetric_template()),
        MatchParameters(angle_min=-10, angle_max=10, angle_step=5),
    )
    first = matcher._variants()
    second = matcher._variants()
    assert first is second
    assert len(first) == 5


def test_textile_chroma_feature_mode_matches_pink_workpiece():
    template = np.full((60, 90, 3), 190, np.uint8)
    mask = np.zeros(template.shape[:2], np.uint8)
    points = np.array([[5, 20], [45, 8], [84, 18], [70, 50], [30, 44], [12, 55]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    template[mask > 0] = (180, 160, 205)
    cv2.line(template, (18, 27), (70, 25), (55, 45, 165), 3)
    scene = np.full((180, 260, 3), 205, np.uint8)
    rotated = _rotate_expanded(template, 16, 205)
    rotated_mask = _rotate_expanded(mask, 16, 0, cv2.INTER_NEAREST)
    top, left = int(90 - rotated.shape[0] / 2), int(130 - rotated.shape[1] / 2)
    region = scene[top : top + rotated.shape[0], left : left + rotated.shape[1]]
    region[rotated_mask > 0] = rotated[rotated_mask > 0]
    params = MatchParameters(score_threshold=0.65, angle_min=10, angle_max=22, angle_step=2, use_edges=False, feature_mode="textile_chroma")
    matches = TemplateMatcher(TemplateModel("pink-textile", template, mask=mask), params).match(scene)
    assert len(matches) == 1
    assert matches[0].angle_deg == pytest.approx(16, abs=2)


def test_pose_tolerant_mode_handles_mild_textile_edge_warp():
    template = np.full((86, 142, 3), 205, np.uint8)
    mask = np.zeros(template.shape[:2], np.uint8)
    points = np.array([[7, 33], [28, 17], [66, 22], [82, 35], [125, 19], [136, 43],
                       [116, 61], [87, 57], [75, 78], [49, 70], [31, 79], [36, 55], [12, 54]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    template[mask > 0] = (184, 162, 215)
    cv2.line(template, (25, 39), (116, 43), (58, 45, 168), 4, cv2.LINE_AA)

    warped_image = template.copy()
    warped_mask = mask.copy()
    source = np.float32([[0, 0], [141, 0], [0, 85], [141, 85]])
    target = np.float32([[2, 4], [137, 0], [0, 82], [141, 85]])
    perspective = cv2.getPerspectiveTransform(source, target)
    warped_image = cv2.warpPerspective(warped_image, perspective, (142, 86), borderValue=(205, 205, 205))
    warped_mask = cv2.warpPerspective(warped_mask, perspective, (142, 86), flags=cv2.INTER_NEAREST)
    rotated = _rotate_expanded(warped_image, 20, 205)
    rotated_mask = _rotate_expanded(warped_mask, 20, 0, cv2.INTER_NEAREST)
    scene = np.full((230, 330, 3), 205, np.uint8)
    top, left = int(115 - rotated.shape[0] / 2), int(165 - rotated.shape[1] / 2)
    region = scene[top : top + rotated.shape[0], left : left + rotated.shape[1]]
    region[rotated_mask > 0] = rotated[rotated_mask > 0]
    moments = cv2.moments(rotated_mask, binaryImage=True)
    expected_x = left + moments["m10"] / moments["m00"]
    expected_y = top + moments["m01"] / moments["m00"]

    params = MatchParameters(score_threshold=0.50, angle_min=12, angle_max=28, angle_step=2,
                             feature_mode="pose_tolerant", use_edges=False)
    matches = TemplateMatcher(TemplateModel("warped-textile", template, mask=mask), params).match(scene)
    assert len(matches) == 1
    assert matches[0].center_x == pytest.approx(expected_x, abs=4)
    assert matches[0].center_y == pytest.approx(expected_y, abs=4)
    assert matches[0].angle_deg == pytest.approx(20, abs=4)


def test_pose_tolerant_mode_auto_scales_light_part_on_dark_conveyor():
    template = np.full((86, 142, 3), 30, np.uint8)
    mask = np.zeros(template.shape[:2], np.uint8)
    points = np.array([[7, 33], [28, 17], [66, 22], [82, 35], [125, 19], [136, 43],
                       [116, 61], [87, 57], [75, 78], [49, 70], [31, 79], [36, 55], [12, 54]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    template[mask > 0] = (184, 162, 215)

    scene = np.full((420, 620, 3), 24, np.uint8)
    cv2.rectangle(scene, (30, 80), (230, 340), (36, 36, 36), -1)  # neutral textile distractor
    scale, angle = 1.45, 72.0
    matrix = cv2.getRotationMatrix2D((71, 43), angle, scale)
    matrix[0, 2] += 430 - 71
    matrix[1, 2] += 220 - 43
    transformed_image = cv2.warpAffine(template, matrix, (620, 420), borderValue=(24, 24, 24))
    transformed_mask = cv2.warpAffine(mask, matrix, (620, 420), flags=cv2.INTER_NEAREST)
    scene[transformed_mask > 0] = transformed_image[transformed_mask > 0]
    scene = cv2.convertScaleAbs(scene, alpha=1.35, beta=8)
    rng = np.random.default_rng(5)
    noise_y = rng.integers(0, scene.shape[0], 500)
    noise_x = rng.integers(0, scene.shape[1], 500)
    scene[noise_y, noise_x] = rng.choice([0, 255], 500)[:, None]
    transformed_moments = cv2.moments(transformed_mask, binaryImage=True)
    expected_x = transformed_moments["m10"] / transformed_moments["m00"]
    expected_y = transformed_moments["m01"] / transformed_moments["m00"]

    params = MatchParameters(
        score_threshold=0.72,
        angle_min=-180,
        angle_max=178,
        angle_step=2,
        scale_min=0.5,
        scale_max=1.8,
        scale_step=0.05,
        feature_mode="pose_tolerant",
        use_edges=False,
    )
    matches = TemplateMatcher(TemplateModel("conveyor-part", template, mask=mask), params).match(scene)
    assert len(matches) == 1
    assert matches[0].center_x == pytest.approx(expected_x, abs=4)
    assert matches[0].center_y == pytest.approx(expected_y, abs=4)
    assert matches[0].angle_deg == pytest.approx(angle, abs=5)
    assert matches[0].scale == pytest.approx(scale, abs=0.12)


def test_dark_textile_mode_rejects_large_smooth_border_background(monkeypatch):
    rng = np.random.default_rng(17)
    template = rng.integers(24, 76, (90, 120, 3), dtype=np.uint8)
    mask = np.full(template.shape[:2], 255, np.uint8)
    scene_line = np.linspace(34, 92, 320, dtype=np.float32)
    scene = np.repeat(scene_line[None, :, None], 220, axis=0)
    scene = np.repeat(scene, 3, axis=2).astype(np.uint8)
    false_component = np.zeros(scene.shape[:2], np.uint8)
    cv2.rectangle(false_component, (0, 30), (175, 205), 255, -1)
    params = MatchParameters(
        score_threshold=0.50,
        angle_min=-180,
        angle_max=178,
        angle_step=2,
        scale_min=0.5,
        scale_max=1.8,
        scale_step=0.05,
        feature_mode="dark_textile",
        use_edges=False,
    )
    matcher = TemplateMatcher(TemplateModel("dark-textile", template, mask=mask), params)
    monkeypatch.setattr(matcher, "_pose_candidate_mask", lambda _image: false_component)

    assert matcher.match(scene) == []
