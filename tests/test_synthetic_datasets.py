import json
from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.datasets import (
    MaskDirectoryAdapter,
    SyntheticConfig,
    SyntheticDatasetGenerator,
)
from industrial_segpose.experiment_db import ExperimentDatabase


def _generate(root: Path, seed: int = 123, profile: str = "mixed"):
    db = ExperimentDatabase(root / "benchmark.db")
    config = SyntheticConfig(
        image_width=320,
        image_height=240,
        image_count=3,
        min_objects=2,
        max_objects=3,
        profile=profile,
        random_seed=seed,
    )
    result = SyntheticDatasetGenerator(config).generate(root / "dataset", db)
    return db, result


def test_synthetic_generation_is_reproducible_and_registered(tmp_path):
    first_db, first = _generate(tmp_path / "first")
    second_db, second = _generate(tmp_path / "second")
    first_images = sorted((first.output_dir / "images").glob("*.png"))
    second_images = sorted((second.output_dir / "images").glob("*.png"))
    assert [path.read_bytes() for path in first_images] == [path.read_bytes() for path in second_images]
    assert first.image_count == second.image_count == 3
    assert first.object_count == second.object_count
    assert json.loads(first.config_path.read_text(encoding="utf-8"))["random_seed"] == 123
    records = first_db.images_for_dataset(first.dataset_id)
    assert len(records) == 3
    assert sum(len(record.ground_truth_objects) for record in records) == first.object_count


def test_ground_truth_uses_bottom_left_coordinates_and_complete_files(tmp_path):
    db, result = _generate(tmp_path / "ground_truth", profile="rotation")
    for record in db.images_for_dataset(result.dataset_id):
        assert Path(record.file_path).is_file()
        label_path = Path(record.metadata_json["label_map_path"])
        assert label_path.is_file()
        label_map = cv2.imdecode(np.fromfile(label_path, np.uint8), cv2.IMREAD_UNCHANGED)
        for item in record.ground_truth_objects:
            assert 0 <= item.center_x < record.width
            assert 0 <= item.center_y < record.height
            assert 0 <= item.axis_angle_deg < 180
            assert 0 <= item.directed_angle_deg < 360
            assert Path(item.mask_path).is_file()
            mask = cv2.imdecode(np.fromfile(item.mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
            assert mask.shape == (record.height, record.width)
            assert np.count_nonzero(mask) > 0
            assert item.object_index in np.unique(label_map)


def test_touching_profile_generates_multiple_visible_instances(tmp_path):
    db, result = _generate(tmp_path / "touching", profile="touching")
    assert result.object_count >= 6
    for record in db.images_for_dataset(result.dataset_id):
        assert len(record.ground_truth_objects) >= 2


def test_production_profiles_do_not_generate_overlapping_instances(tmp_path):
    profiles = (
        "clean", "rotation", "scale_variation", "color_variation",
        "lighting", "noise", "dense_separated", "edge_partial",
        "production_mixed",
    )
    for index, profile in enumerate(profiles):
        root = tmp_path / profile
        db = ExperimentDatabase(root / "benchmark.db")
        result = SyntheticDatasetGenerator(SyntheticConfig(
            image_width=512,
            image_height=384,
            image_count=2,
            min_objects=3,
            max_objects=4,
            min_scale=0.68 if profile == "dense_separated" else 0.75,
            max_scale=0.82 if profile == "dense_separated" else 1.10,
            profile=profile,
            random_seed=900 + index,
        )).generate(root / "dataset", db)
        records = db.images_for_dataset(result.dataset_id)
        assert all(
            not item.metadata_json["partially_occluded"]
            for record in records
            for item in record.ground_truth_objects
        )


def test_mask_directory_adapter_imports_objects_with_shape_axis_angles(tmp_path):
    source_db, generated = _generate(tmp_path / "source", profile="clean")
    imported_db = ExperimentDatabase(tmp_path / "imported.db")
    dataset_id = MaskDirectoryAdapter().import_dataset(
        generated.output_dir, imported_db, "public_example"
    )
    records = imported_db.images_for_dataset(dataset_id)
    assert len(records) == 3
    assert all(record.ground_truth_objects for record in records)
    assert all(
        item.axis_angle_deg is not None and item.directed_angle_deg is None
        for record in records
        for item in record.ground_truth_objects
    )
