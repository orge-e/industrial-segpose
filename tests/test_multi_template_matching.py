from dataclasses import replace
import csv
import json

import cv2
import numpy as np
import pytest

from industrial_segpose.template_matching import (
    MatchParameters,
    MultiTemplateMatcher,
    TemplateLibrary,
    TemplateMatch,
    TemplateModel,
    draw_multi_template_matches,
    write_multi_template_result,
)
from industrial_segpose.template_matching.matcher import _rotate_expanded


def part_model(name, kind):
    image = np.zeros((55, 75, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    if kind == "L":
        cv2.rectangle(mask, (6, 8), (65, 18), 255, -1)
        cv2.rectangle(mask, (6, 8), (16, 47), 255, -1)
    else:
        points = np.array([[37, 5], [69, 46], [8, 46]], np.int32)
        cv2.fillPoly(mask, [points], 255)
        cv2.circle(mask, (37, 32), 8, 0, -1)
    image[mask > 0] = 230
    cv2.circle(image, (25 if kind == "L" else 37, 14), 4, (70, 70, 70), -1)
    return TemplateModel(name, image, mask=mask)


def paste(scene, image, center, angle=0):
    rotated = _rotate_expanded(image, angle, 0)
    height, width = rotated.shape[:2]
    left, top = int(center[0] - width / 2), int(center[1] - height / 2)
    scene[top : top + height, left : left + width] = np.maximum(scene[top : top + height, left : left + width], rotated)


def test_multi_template_end_to_end_counts_types(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    params = MatchParameters(score_threshold=0.72, angle_min=-40, angle_max=40, angle_step=2, use_edges=False)
    a = part_model("工件A", "L")
    b = part_model("工件B", "triangle")
    library.add_model(a, params)
    library.add_model(b, params)
    scene = np.zeros((280, 440, 3), np.uint8)
    paste(scene, a.image, (100, 85), 24)
    paste(scene, a.image, (320, 80), -18)
    paste(scene, b.image, (225, 200), 30)
    result = MultiTemplateMatcher.from_library(library).match(scene)
    assert result.object_count == 3
    assert result.counts_by_template == {"工件A": 2, "工件B": 1}
    assert result.ambiguous_count == 0
    assert {item.template_name for item in result.objects} == {"工件A", "工件B"}


def test_large_detection_image_is_resized_and_coordinates_are_restored(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    params = MatchParameters(
        score_threshold=0.72,
        angle_min=-40,
        angle_max=40,
        angle_step=2,
        use_edges=False,
        max_results=1,
    )
    model = part_model("A", "L")
    library.add_model(model, params)
    scene = np.zeros((280, 440, 3), np.uint8)
    paste(scene, model.image, (310, 175), 20)
    matcher = MultiTemplateMatcher.from_library(library)
    matcher.max_processing_pixels = 60_000
    matcher.max_processing_edge = 400

    processed, processing_scale = matcher._prepare_detection_image(scene)
    processed_match = TemplateMatch(
        1,
        "A",
        310 * processing_scale,
        175 * processing_scale,
        20.0,
        0.9,
        processing_scale,
        tuple((x * processing_scale, y * processing_scale) for x, y in ((280, 150), (340, 150), (340, 200), (280, 200))),
        tuple((x * processing_scale, y * processing_scale) for x, y in ((280, 150), (340, 150), (340, 200), (280, 200))),
    )
    restored = matcher._restore_source_coordinates(processed_match, processing_scale)

    assert processing_scale < 1.0
    assert processed.shape[1] < 440
    assert restored.center_x == pytest.approx(310)
    assert restored.center_y == pytest.approx(175)
    assert restored.scale == pytest.approx(1.0)


def fake_match(name, score):
    return TemplateMatch(
        1, name, 100.0, 80.0, 20.0, score, 1.0,
        ((70.0, 60.0), (130.0, 60.0), (130.0, 100.0), (70.0, 100.0)),
        ((70.0, 60.0), (130.0, 60.0), (130.0, 100.0), (70.0, 100.0)),
    )


class FakeMatcher:
    def __init__(self, result):
        self.result = result

    def match(self, _image):
        return [self.result]


def conflict_matcher(tmp_path, score_a, score_b, margin=0.08):
    library = TemplateLibrary(tmp_path / "templates")
    params = MatchParameters(score_threshold=0.7, angle_min=0, angle_max=0)
    first = library.add_model(part_model("A", "L"), params)
    second = library.add_model(part_model("B", "triangle"), params)
    matcher = MultiTemplateMatcher.from_library(library)
    matcher.ambiguity_margin = margin
    matcher._matchers[first.template_id] = FakeMatcher(fake_match("A", score_a))
    matcher._matchers[second.template_id] = FakeMatcher(fake_match("B", score_b))
    return matcher


def test_cross_template_small_margin_is_ambiguous(tmp_path):
    result = conflict_matcher(tmp_path, 0.90, 0.89).match(np.zeros((160, 200, 3), np.uint8))
    assert result.object_count == 1
    assert result.ambiguous_count == 1
    assert result.objects[0].classification_status == "ambiguous"
    assert result.objects[0].template_name == "待确认"
    assert all(value == 0 for value in result.counts_by_template.values())


def test_cross_template_clear_winner_is_confirmed(tmp_path):
    result = conflict_matcher(tmp_path, 0.96, 0.80).match(np.zeros((160, 200, 3), np.uint8))
    assert result.object_count == 1
    assert result.ambiguous_count == 0
    assert result.objects[0].template_name == "A"
    assert result.counts_by_template["A"] == 1


def test_all_templates_disabled_is_rejected(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    entry = library.add_model(part_model("A", "L"), MatchParameters(angle_min=0, angle_max=0))
    library.set_enabled(entry.template_id, False)
    matcher = MultiTemplateMatcher.from_library(library)
    assert matcher.valid_enabled_count == 0
    try:
        matcher.match(np.zeros((100, 100, 3), np.uint8))
        assert False, "expected no-template error"
    except ValueError as exc:
        assert "No valid enabled templates" in str(exc)


def test_multi_result_json_csv_and_visual_export(tmp_path):
    result = conflict_matcher(tmp_path / "library", 0.90, 0.89).match(np.zeros((160, 200, 3), np.uint8))
    annotated = draw_multi_template_matches(np.zeros((160, 200, 3), np.uint8), result)
    run_dir = write_multi_template_result(tmp_path / "outputs", "检测图.png", annotated, result, "fixed_run")
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["object_count"] == 1
    assert payload["ambiguous_count"] == 1
    assert len(payload["templates"]) == 2
    assert len(payload["objects"][0]["candidate_templates"]) == 2
    assert payload["coordinate_system"]["origin"] == "bottom_left"
    assert payload["coordinate_system"]["angle_positive_direction"] == "counter_clockwise"
    assert payload["objects"][0]["center_x"] == pytest.approx(100.0)
    assert payload["objects"][0]["center_y"] == pytest.approx(79.0)
    assert payload["objects"][0]["angle_deg"] == pytest.approx(340.0)
    assert payload["objects"][0]["axis_angle_deg"] == pytest.approx(160.0)
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["classification_status"] == "ambiguous"
    assert rows[0]["template_name"] == "待确认"
    assert float(rows[0]["center_y"]) == pytest.approx(79.0)
    assert float(rows[0]["angle_deg"]) == pytest.approx(340.0)
    assert float(rows[0]["axis_angle_deg"]) == pytest.approx(160.0)
    assert float(rows[0]["directed_angle_deg"]) == pytest.approx(340.0)
    assert rows[0]["coordinate_origin"] == "bottom_left"
    assert json.loads(rows[0]["candidate_templates"])[0]["template_name"] in {"A", "B"}
    assert (run_dir / "annotated.png").is_file()


def test_output_coordinate_serialization_does_not_change_native_match_result(tmp_path):
    result = conflict_matcher(tmp_path / "library", 0.96, 0.80).match(
        np.zeros((160, 200, 3), np.uint8)
    )

    native = result.to_dict()["objects"][0]
    exported = result.to_output_dict()["objects"][0]

    assert native["center_y"] == pytest.approx(80.0)
    assert native["angle_deg"] == pytest.approx(20.0)
    assert exported["center_y"] == pytest.approx(79.0)
    assert exported["angle_deg"] == pytest.approx(340.0)
    assert exported["box_points"][0] == pytest.approx([70.0, 99.0])
