import json

import pytest
from sqlalchemy import delete

from industrial_segpose.experiment_db import Dataset, ExperimentDatabase


def test_complete_database_lifecycle_uses_isolated_sqlite(tmp_path):
    database_path = tmp_path / "benchmark.db"
    image_path = tmp_path / "测试图像.bin"
    image_path.write_bytes(b"synthetic-image")
    db = ExperimentDatabase(database_path)
    db.initialize()

    dataset = db.create_dataset(
        "合成测试", "synthetic", version="v1", generator_config={"seed": 42, "count": 2}
    )
    image = db.register_image(
        dataset.id, image_path, 320, 240, metadata={"profile": "clean"}
    )
    gt = db.add_ground_truth(
        image.id,
        1,
        center_x=100.5,
        center_y=80.5,
        mask_path=tmp_path / "mask.png",
        axis_angle_deg=35.0,
        directed_angle_deg=None,
        bbox=[50, 40, 100, 80],
        metadata={"occluded": False},
    )
    algorithm = db.get_or_create_algorithm("threshold")
    assert db.get_or_create_algorithm("threshold").id == algorithm.id
    run = db.create_run(
        algorithm.id, dataset.id, {"threshold": {"method": "otsu"}}, git_state="dirty"
    )
    prediction = db.add_prediction(
        run.id,
        image.id,
        1,
        center_x=101.0,
        center_y=80.0,
        axis_angle_deg=34.0,
        directed_angle_deg=None,
        confidence=0.9,
        processing_time_ms=12.5,
        bbox=[51, 40, 100, 80],
    )
    metric = db.add_metric(run.id, "iou", 0.91, image_id=image.id, object_index=1)
    db.finish_run(run.id)

    assert database_path.is_file()
    assert gt.id and prediction.id and metric.id
    loaded_images = db.images_for_dataset(dataset.id)
    assert loaded_images[0].metadata_json == {"profile": "clean"}
    assert loaded_images[0].ground_truth_objects[0].center_y == 80.5
    loaded_runs = db.runs_for_dataset(dataset.id)
    assert loaded_runs[0].status == "completed"
    assert loaded_runs[0].config == {"threshold": {"method": "otsu"}}
    assert loaded_runs[0].algorithm.name == "threshold"
    assert loaded_runs[0].metrics[0].value == pytest.approx(0.91)


def test_foreign_keys_and_rebuild(tmp_path):
    db = ExperimentDatabase(tmp_path / "foreign_keys.db")
    db.initialize()
    with pytest.raises(Exception):
        db.register_image(999, tmp_path / "missing.bin", 10, 10, file_hash="0" * 64)
    dataset = db.create_dataset("real", "real")
    assert db.datasets()[0].id == dataset.id
    db.rebuild()
    assert db.datasets() == []


def test_source_type_validation_and_json_round_trip(tmp_path):
    db = ExperimentDatabase(tmp_path / "json.db")
    db.initialize()
    config = {"nested": {"values": [1, 2, 3]}, "unicode": "纺织件"}
    dataset = db.create_dataset("版本化数据", "historical", generator_config=config)
    loaded = db.datasets()[0]
    assert json.loads(json.dumps(loaded.generator_config, ensure_ascii=False)) == config
    with pytest.raises(ValueError, match="source_type"):
        db.create_dataset("bad", "unknown")
