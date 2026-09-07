import json

import cv2
import numpy as np
import pytest

from industrial_segpose.evaluation.benchmark import (
    EvaluationThresholds,
    GroundTruthObject,
    SyntheticSceneGenerator,
    SyntheticStressConfig,
    circular_angle_error,
    evaluate_scene,
    run_synthetic_benchmark,
)
from industrial_segpose.template_matching import (
    MatchParameters,
    MultiTemplateResult,
    RecognizedObject,
    TemplateLibrary,
    TemplateModel,
)


def _library(root):
    image = np.zeros((48, 72, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (5, 8), (62, 18), 255, -1)
    cv2.rectangle(mask, (5, 8), (16, 41), 255, -1)
    image[mask > 0] = (40, 180, 235)
    library = TemplateLibrary(root)
    entry = library.add_model(
        TemplateModel("测试工件", image, mask=mask),
        MatchParameters(
            score_threshold=0.60,
            angle_min=0,
            angle_max=0,
            scale_min=1,
            scale_max=1,
            use_edges=False,
            feature_mode="gray",
            max_results=10,
        ),
    )
    return library, entry


def _truth(template_id="template-a", name="工件A"):
    points = ((80.0, 60.0), (120.0, 60.0), (120.0, 100.0), (80.0, 100.0))
    return GroundTruthObject(1, template_id, name, 100.0, 80.0, 179.0, 1.0, points, points)


def _result(template_id="template-a", name="工件A", center=(103.0, 84.0), angle=-179.0):
    points = ((80.0, 60.0), (120.0, 60.0), (120.0, 100.0), (80.0, 100.0))
    obj = RecognizedObject(
        1, "confirmed", template_id, name, center[0], center[1], angle, 0.91, 0.8, 1.0,
        (0, 220, 0), points, points, (),
    )
    return MultiTemplateResult((obj,), {name: 1}, 0, {}, ())


def test_angle_error_wraps_at_360_degrees():
    assert circular_angle_error(179, -179) == pytest.approx(2)


def test_evaluator_reports_center_angle_and_classification():
    evaluation = evaluate_scene(
        "scene_1", [_truth()], _result(), 12.5,
        EvaluationThresholds(center_tolerance_px=10, angle_tolerance_deg=5),
    )
    assert evaluation.true_positive == 1
    assert evaluation.false_positive == 0
    assert evaluation.false_negative == 0
    assert evaluation.center_errors_px == pytest.approx([5.0])
    assert evaluation.angle_errors_deg == pytest.approx([2.0])
    assert evaluation.count_exact


def test_synthetic_generator_is_repeatable(tmp_path):
    library, _ = _library(tmp_path / "templates")
    config = SyntheticStressConfig(width=360, height=240, min_objects=1, max_objects=1, profile="balanced")
    first_image, first_truth = SyntheticSceneGenerator(library, config, seed=7).generate(0)
    second_image, second_truth = SyntheticSceneGenerator(library, config, seed=7).generate(0)
    assert np.array_equal(first_image, second_image)
    assert first_truth == second_truth


def test_benchmark_writes_reusable_machine_readable_report(tmp_path):
    library, _ = _library(tmp_path / "templates")
    output, summary = run_synthetic_benchmark(
        library.root,
        tmp_path / "report",
        scene_count=1,
        seed=3,
        config=SyntheticStressConfig(width=360, height=240, min_objects=1, max_objects=1, profile="clean"),
        thresholds=EvaluationThresholds(center_tolerance_px=30, angle_tolerance_deg=10),
        save_images=False,
    )
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary.scene_count == 1
    assert payload["ground_truth_count"] == 1
    assert (output / "benchmark_config.json").is_file()
    assert (output / "ground_truth.jsonl").is_file()
    assert (output / "predictions.jsonl").is_file()
    assert (output / "evaluations.jsonl").is_file()
    assert (output / "per_scene.csv").is_file()
    assert not (output / "scenes").exists()
