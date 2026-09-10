from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_segpose.evaluation.runner import (
    ALGORITHM_NAMES,
    BenchmarkRunner,
    axis_angle_error,
    directed_angle_error,
    mask_dice,
    mask_iou,
    match_instances,
)
from industrial_segpose.datasets import SyntheticConfig, SyntheticDatasetGenerator
from industrial_segpose.experiment_db import ExperimentDatabase
from industrial_segpose.types import ObjectResult


def _prediction(mask, center, angle=0.0):
    points = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask))).astype(float).tolist()
    return ObjectResult(
        1, center[0], center[1], angle, "pca", True, 1.0,
        float(cv2.countNonZero(mask)), 1.0, (0, 0, 1, 1), points,
        1.0, 1.0, 1.0, False, axis_angle_deg=angle, _mask=mask,
    )


def test_mask_and_angle_metrics_handle_boundaries():
    first = np.zeros((20, 20), np.uint8)
    second = np.zeros_like(first)
    first[2:12, 2:12] = 255
    second[7:17, 2:12] = 255
    assert mask_iou(first, second) == pytest.approx(1 / 3)
    assert mask_dice(first, second) == pytest.approx(0.5)
    assert axis_angle_error(179, 1) == pytest.approx(2)
    assert axis_angle_error(89, 91) == pytest.approx(2)
    assert directed_angle_error(359, 1) == pytest.approx(2)


def test_instance_matching_uses_iou_not_object_index(tmp_path):
    db = ExperimentDatabase(tmp_path / "matching.db")
    db.initialize()
    dataset = db.create_dataset("matching", "synthetic")
    image_file = tmp_path / "image.bin"
    image_file.write_bytes(b"x")
    image = db.register_image(dataset.id, image_file, 100, 100)
    masks = []
    ground_truth = []
    for index, x in enumerate((10, 60), 1):
        mask = np.zeros((100, 100), np.uint8)
        mask[20:50, x:x + 25] = 255
        mask_path = tmp_path / f"mask{index}.png"
        cv2.imencode(".png", mask)[1].tofile(mask_path)
        masks.append(mask)
        ground_truth.append(
            db.add_ground_truth(
                image.id, index, center_x=x + 12, center_y=65, mask_path=mask_path,
                axis_angle_deg=0, directed_angle_deg=None,
            )
        )
    predictions = [_prediction(masks[1], (72, 65)), _prediction(masks[0], (22, 65))]
    matches = match_instances(ground_truth, masks, predictions)
    assert {(item.gt_index, item.prediction_index) for item in matches} == {(0, 1), (1, 0)}
    assert all(item.iou == pytest.approx(1.0) for item in matches)


def test_runner_executes_existing_four_algorithms_and_writes_all_reports(tmp_path):
    db = ExperimentDatabase(tmp_path / "benchmark.db")
    generated = SyntheticDatasetGenerator(
        SyntheticConfig(
            image_width=320,
            image_height=240,
            image_count=2,
            min_objects=1,
            max_objects=2,
            profile="clean",
            random_seed=8,
        )
    ).generate(tmp_path / "dataset", db, dataset_name="benchmark_clean")
    output = tmp_path / "reports"
    summaries = BenchmarkRunner(db, output, minimum_iou=0.20).run(generated.dataset_id)
    assert set(summaries) == set(ALGORITHM_NAMES)
    assert all(summary.image_count == 2 for summary in summaries.values())
    assert (output / "summary.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "details.json").is_file()
    assert (output / "summary.md").is_file()
    runs = db.runs_for_dataset(generated.dataset_id)
    assert len(runs) == 4
    assert all(run.status == "completed" for run in runs)
    assert all(run.metrics for run in runs)
    assert all((output / name / "prediction_masks").is_dir() for name in ALGORITHM_NAMES)
