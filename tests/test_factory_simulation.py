from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.datasets.factory_simulation import (
    PhotoAugmentationConfig,
    PickupSimulationConfig,
    VisualTwinConfig,
    export_workpiece_assets,
    generate_photo_augmentations,
    generate_pickup_scenes,
    generate_visual_twin_scenes,
    load_instance_label_map,
)
from industrial_segpose.io.image_reader import write_image


def _sample(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((120, 180, 3), 55, np.uint8)
    labels = np.zeros((120, 180), np.uint8)
    cv2.rectangle(image, (20, 25), (70, 80), (190, 210, 230), -1)
    cv2.rectangle(image, (105, 35), (155, 90), (175, 205, 220), -1)
    cv2.rectangle(labels, (20, 25), (70, 80), 255, -1)
    cv2.rectangle(labels, (105, 35), (155, 90), 255, -1)
    image_path, label_path = tmp_path / "现场图.jpg", tmp_path / "实例.png"
    write_image(image_path, image)
    write_image(label_path, labels)
    return image_path, label_path


def test_binary_label_map_is_converted_to_instance_ids(tmp_path: Path):
    _, label_path = _sample(tmp_path)
    labels = load_instance_label_map(label_path, (120, 180))
    assert set(np.unique(labels)) == {0, 1, 2}


def test_reviewed_instances_export_as_transparent_assets(tmp_path: Path):
    import json

    image_path, label_path = _sample(tmp_path)
    sidecar = label_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({
            "objects": [
                {"object_id": 1, "label": "鞋面A", "axis_angle_deg": 12.0},
                {"object_id": 2, "label": "鞋面B", "axis_angle_deg": 24.0},
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    result = export_workpiece_assets(
        image_path, label_path, tmp_path / "数字资产", padding_px=5
    )
    assert result.asset_count == 2
    assert result.preview_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["class_name"] for item in manifest["assets"]] == ["鞋面A", "鞋面B"]
    asset = cv2.imdecode(
        np.fromfile(result.output_dir / manifest["assets"][0]["image_path"], np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert asset.shape[2] == 4
    assert np.any(asset[:, :, 3] == 0)
    assert np.any(asset[:, :, 3] == 255)


def test_photo_augmentation_writes_repeatable_dataset(tmp_path: Path):
    image_path, label_path = _sample(tmp_path)
    result = generate_photo_augmentations(
        image_path,
        tmp_path / "augmented",
        label_map_path=label_path,
        config=PhotoAugmentationConfig(variant_count=3, random_seed=7),
    )
    assert result.image_count == 3
    assert result.source_instance_count == 2
    assert len(list((result.output_dir / "images").glob("*.jpg"))) == 3
    assert len(list((result.output_dir / "label_maps").glob("*.png"))) == 3
    assert result.manifest_path.is_file()
    assert result.preview_path.is_file()


def test_pickup_simulation_updates_visible_count(tmp_path: Path):
    image_path, label_path = _sample(tmp_path)
    result = generate_pickup_scenes(
        image_path,
        label_path,
        tmp_path / "pickup",
        config=PickupSimulationConfig(
            scene_count=2,
            random_seed=9,
            removal_ratios=(0.0, 0.5),
            augmentation=PhotoAugmentationConfig(
                variant_count=1,
                random_seed=9,
                brightness_range=(1.0, 1.0),
                contrast_range=(1.0, 1.0),
                gamma_range=(1.0, 1.0),
                color_temperature_shift=0,
                noise_sigma_range=(0.0, 0.0),
                blur_probability=0.0,
                shadow_probability=0.0,
                max_rotation_deg=0.0,
                max_translation_px=0.0,
                jpeg_quality_range=(100, 100),
            ),
        ),
    )
    first = cv2.imdecode(
        np.fromfile(result.output_dir / "label_maps" / "pickup_00000.labels.png", np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    second = cv2.imdecode(
        np.fromfile(result.output_dir / "label_maps" / "pickup_00001.labels.png", np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert set(np.unique(first)) == {0, 1, 2}
    assert set(np.unique(second)) == {0, 1}


def test_visual_twin_composes_non_overlapping_typed_instances(tmp_path: Path):
    image_path, label_path = _sample(tmp_path)
    metadata_path = tmp_path / "工件属性.json"
    metadata_path.write_text(
        '{"objects":{"1":{"class_name":"鞋面A","template_id":"a",'
        '"reference_angle_deg":15},"2":{"class_name":"鞋面B",'
        '"template_id":"b","reference_angle_deg":195}}}',
        encoding="utf-8",
    )
    result = generate_visual_twin_scenes(
        image_path,
        label_path,
        tmp_path / "视觉孪生",
        asset_metadata_path=metadata_path,
        config=VisualTwinConfig(
            scene_count=3,
            random_seed=17,
            min_objects=2,
            max_objects=3,
            min_scale=0.55,
            max_scale=0.7,
            min_gap_px=4,
            profile="clean",
        ),
    )
    assert result.image_count == 3
    assert result.source_instance_count == 2
    assert result.preview_path.is_file()
    summary = (result.output_dir / "generation_summary.json").read_text(encoding="utf-8")
    assert '"source_asset_count": 2' in summary
    for metadata_file in sorted((result.output_dir / "metadata").glob("*.json")):
        import json

        record = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert 1 <= record["generated_object_count"] <= 3
        assert {item["class_name"] for item in record["objects"]} <= {"鞋面A", "鞋面B"}
        assert all(0.0 <= item["directed_angle_deg"] < 360.0 for item in record["objects"])
        label_map = cv2.imdecode(
            np.fromfile(result.output_dir / record["label_map"], np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        ids = [int(value) for value in np.unique(label_map) if value > 0]
        for instance_id in ids:
            own = np.where(label_map == instance_id, 255, 0).astype(np.uint8)
            expanded = cv2.dilate(own, np.ones((5, 5), np.uint8))
            assert not np.any((expanded > 0) & (label_map > 0) & (label_map != instance_id))


def test_visual_twin_is_repeatable_for_same_seed(tmp_path: Path):
    image_path, label_path = _sample(tmp_path)
    config = VisualTwinConfig(
        scene_count=1,
        random_seed=23,
        min_objects=2,
        max_objects=2,
        min_scale=0.5,
        max_scale=0.5,
        profile="balanced",
    )
    first = generate_visual_twin_scenes(
        image_path, label_path, tmp_path / "first", config=config
    )
    second = generate_visual_twin_scenes(
        image_path, label_path, tmp_path / "second", config=config
    )
    first_image = (first.output_dir / "images" / "visual_twin_00000.jpg").read_bytes()
    second_image = (second.output_dir / "images" / "visual_twin_00000.jpg").read_bytes()
    assert first_image == second_image
