"""Factory-photo simulation and repeatable camera-condition augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..io.image_reader import read_image, write_image
from ..measurement.orientation import min_area_rect_orientation


VISUAL_TWIN_PROFILES = {"clean", "balanced", "harsh"}


@dataclass(frozen=True)
class PhotoAugmentationConfig:
    variant_count: int = 12
    random_seed: int = 20260824
    brightness_range: tuple[float, float] = (0.72, 1.28)
    contrast_range: tuple[float, float] = (0.82, 1.18)
    gamma_range: tuple[float, float] = (0.78, 1.28)
    color_temperature_shift: int = 18
    noise_sigma_range: tuple[float, float] = (0.0, 12.0)
    blur_probability: float = 0.45
    shadow_probability: float = 0.65
    max_rotation_deg: float = 1.5
    max_translation_px: float = 5.0
    jpeg_quality_range: tuple[int, int] = (82, 100)

    def validate(self) -> None:
        if self.variant_count < 1:
            raise ValueError("variant_count must be positive")
        if self.brightness_range[0] <= 0 or self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("brightness_range is invalid")
        if self.contrast_range[0] <= 0 or self.contrast_range[0] > self.contrast_range[1]:
            raise ValueError("contrast_range is invalid")
        if self.gamma_range[0] <= 0 or self.gamma_range[0] > self.gamma_range[1]:
            raise ValueError("gamma_range is invalid")
        if not 0 <= self.blur_probability <= 1 or not 0 <= self.shadow_probability <= 1:
            raise ValueError("probabilities must be in [0, 1]")


@dataclass(frozen=True)
class PickupSimulationConfig:
    scene_count: int = 12
    random_seed: int = 20260824
    removal_ratios: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8)
    inpaint_radius: int = 7
    augmentation: PhotoAugmentationConfig = field(
        default_factory=lambda: PhotoAugmentationConfig(variant_count=1)
    )

    def validate(self) -> None:
        if self.scene_count < 1:
            raise ValueError("scene_count must be positive")
        if self.inpaint_radius < 1:
            raise ValueError("inpaint_radius must be positive")
        if not self.removal_ratios or any(not 0.0 <= value <= 1.0 for value in self.removal_ratios):
            raise ValueError("removal_ratios must contain values in [0, 1]")
        self.augmentation.validate()


@dataclass(frozen=True)
class VisualTwinConfig:
    """Configuration for non-overlapping 2D factory-scene composition."""

    scene_count: int = 100
    random_seed: int = 20260831
    min_objects: int = 1
    max_objects: int = 12
    min_scale: float = 0.85
    max_scale: float = 1.15
    min_gap_px: int = 6
    placement_attempts: int = 240
    profile: str = "balanced"
    background_inpaint_radius: int = 9
    placement_roi: tuple[int, int, int, int] | None = None

    def validate(self) -> None:
        if self.scene_count < 1:
            raise ValueError("scene_count must be positive")
        if not 1 <= self.min_objects <= self.max_objects:
            raise ValueError("object count range is invalid")
        if not 0.2 <= self.min_scale <= self.max_scale <= 3.0:
            raise ValueError("scale range is invalid")
        if self.min_gap_px < 0 or self.placement_attempts < 1:
            raise ValueError("placement constraints are invalid")
        if self.profile not in VISUAL_TWIN_PROFILES:
            raise ValueError(f"Unsupported visual-twin profile: {self.profile}")
        if self.background_inpaint_radius < 1:
            raise ValueError("background_inpaint_radius must be positive")
        if self.placement_roi is not None:
            x, y, width, height = self.placement_roi
            if x < 0 or y < 0 or width < 1 or height < 1:
                raise ValueError("placement_roi must be x, y, width, height with positive size")


@dataclass(frozen=True)
class FactoryGenerationResult:
    output_dir: Path
    image_count: int
    source_instance_count: int
    manifest_path: Path
    preview_path: Path


@dataclass(frozen=True)
class WorkpieceAssetExportResult:
    output_dir: Path
    asset_count: int
    manifest_path: Path
    preview_path: Path


def load_instance_label_map(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load integer instance labels; turn a binary mask into connected IDs."""
    source = Path(path)
    data = np.fromfile(source, np.uint8)
    label_map = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"Could not decode label map: {source}")
    if label_map.ndim == 3:
        label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2GRAY)
    if label_map.shape != expected_shape:
        raise ValueError(
            f"Label map shape {label_map.shape} does not match image shape {expected_shape}"
        )
    unique = np.unique(label_map)
    if len(unique) <= 2 and unique[-1] > 0:
        _, label_map = cv2.connectedComponents((label_map > 0).astype(np.uint8), connectivity=8)
    return label_map.astype(np.uint16)


def export_workpiece_assets(
    source_image: str | Path,
    label_map_path: str | Path,
    output_dir: str | Path,
    *,
    asset_metadata_path: str | Path | None = None,
    padding_px: int = 6,
) -> WorkpieceAssetExportResult:
    """Export reviewed instances as reusable transparent digital assets."""
    if padding_px < 0:
        raise ValueError("padding_px must be non-negative")
    source_path, labels_path = Path(source_image), Path(label_map_path)
    image = read_image(source_path)
    labels = load_instance_label_map(labels_path, image.shape[:2])
    metadata_source = asset_metadata_path
    if metadata_source is None:
        sidecar = labels_path.with_suffix(".json")
        metadata_source = sidecar if sidecar.is_file() else None
    metadata = _load_asset_metadata(metadata_source)
    root = Path(output_dir)
    image_dir, mask_dir, metadata_dir = root / "images", root / "masks", root / "metadata"
    for directory in (image_dir, mask_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    height, width = labels.shape
    for source_id in (int(value) for value in np.unique(labels) if value > 0):
        mask = np.where(labels == source_id, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, box_width, box_height = cv2.boundingRect(points)
        x0, y0 = max(0, x - padding_px), max(0, y - padding_px)
        x1 = min(width, x + box_width + padding_px)
        y1 = min(height, y + box_height + padding_px)
        crop, crop_mask = image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()
        transparent = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        transparent[:, :, 3] = crop_mask
        stem = f"asset_{source_id:03d}"
        image_relative = f"images/{stem}.png"
        mask_relative = f"masks/{stem}.mask.png"
        write_image(root / image_relative, transparent)
        write_image(root / mask_relative, crop_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
        axis_angle, confidence = min_area_rect_orientation(contour)
        info = metadata.get(source_id, {})
        reference = info.get("reference_angle_deg")
        record = {
            "asset_id": stem,
            "source_object_id": source_id,
            "class_name": str(info.get("class_name", "workpiece")),
            "template_id": info.get("template_id"),
            "image_path": image_relative,
            "mask_path": mask_relative,
            "source_bbox_xywh": [int(x), int(y), int(box_width), int(box_height)],
            "source_center_xy": center,
            "crop_origin_xy": [int(x0), int(y0)],
            "crop_size_wh": [int(x1 - x0), int(y1 - y0)],
            "area_px": int(cv2.countNonZero(mask)),
            "axis_angle_deg": float(info.get("axis_angle_deg", axis_angle)),
            "orientation_confidence": float(info.get("orientation_confidence", confidence)),
            "reference_angle_deg": None if reference is None else float(reference) % 360.0,
        }
        metadata_relative = f"metadata/{stem}.json"
        record["metadata_path"] = metadata_relative
        (root / metadata_relative).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
    if not records:
        raise ValueError("Label map contains no usable workpiece instances")

    manifest_path = root / "asset_library.json"
    manifest = {
        "schema_version": 1,
        "source_image": str(source_path),
        "source_label_map": str(labels_path),
        "source_image_sha256": _file_sha256(source_path),
        "source_label_map_sha256": _file_sha256(labels_path),
        "asset_count": len(records),
        "angle_definition": {
            "axis_angle_deg": "0-180 degree undirected geometric axis",
            "reference_angle_deg": "optional 0-360 degree semantic direction",
        },
        "assets": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    preview_path = root / "asset_preview.jpg"
    _write_asset_preview(root, records, preview_path)
    return WorkpieceAssetExportResult(root, len(records), manifest_path, preview_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_asset_preview(root: Path, records: list[dict], output_path: Path) -> None:
    columns, tile_width, tile_height = 6, 220, 180
    panels: list[np.ndarray] = []
    for record in records:
        asset = cv2.imdecode(
            np.fromfile(root / record["image_path"], np.uint8), cv2.IMREAD_UNCHANGED
        )
        if asset.ndim != 3 or asset.shape[2] != 4:
            continue
        available_height = tile_height - 42
        scale = min((tile_width - 20) / asset.shape[1], available_height / asset.shape[0])
        size = (max(1, int(asset.shape[1] * scale)), max(1, int(asset.shape[0] * scale)))
        resized = cv2.resize(asset, size, interpolation=cv2.INTER_AREA)
        panel = np.full((tile_height, tile_width, 3), (24, 29, 37), np.uint8)
        y = 8 + (available_height - size[1]) // 2
        x = (tile_width - size[0]) // 2
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        roi = panel[y:y + size[1], x:x + size[0]].astype(np.float32)
        panel[y:y + size[1], x:x + size[0]] = np.clip(
            resized[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha), 0, 255
        ).astype(np.uint8)
        cv2.putText(
            panel, record["asset_id"], (10, tile_height - 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (236, 241, 248), 1, cv2.LINE_AA,
        )
        cv2.putText(
            panel, record["class_name"][:22], (10, tile_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (145, 164, 188), 1, cv2.LINE_AA,
        )
        panels.append(panel)
    while len(panels) % columns:
        panels.append(np.full((tile_height, tile_width, 3), (24, 29, 37), np.uint8))
    rows = [cv2.hconcat(panels[index:index + columns]) for index in range(0, len(panels), columns)]
    write_image(output_path, cv2.vconcat(rows))


def generate_photo_augmentations(
    source_image: str | Path,
    output_dir: str | Path,
    *,
    label_map_path: str | Path | None = None,
    config: PhotoAugmentationConfig | None = None,
) -> FactoryGenerationResult:
    """Generate camera/lighting variants and transform labels in lockstep."""
    cfg = config or PhotoAugmentationConfig()
    cfg.validate()
    source_path = Path(source_image)
    image = read_image(source_path)
    labels = (
        load_instance_label_map(label_map_path, image.shape[:2])
        if label_map_path is not None
        else None
    )
    root = Path(output_dir)
    _make_dataset_dirs(root, labels is not None)
    rng = np.random.default_rng(cfg.random_seed)
    records: list[dict] = []
    for index in range(cfg.variant_count):
        variant, variant_labels, parameters = augment_image_and_labels(image, labels, rng, cfg)
        stem = f"augmentation_{index:05d}"
        write_image(root / "images" / f"{stem}.jpg", variant)
        objects = _write_labels_and_metadata(root, stem, variant_labels) if variant_labels is not None else []
        metadata = {
            "source_image": str(source_path),
            "mode": "photo_augmentation",
            "augmentation": parameters,
            "objects": objects,
        }
        (root / "metadata" / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append({"image": f"images/{stem}.jpg", **metadata})
    return _finish_generation(root, cfg, records, _instance_count(labels))


def generate_pickup_scenes(
    source_image: str | Path,
    label_map_path: str | Path,
    output_dir: str | Path,
    *,
    config: PickupSimulationConfig | None = None,
) -> FactoryGenerationResult:
    """Simulate workpieces being removed from a labelled factory photograph."""
    cfg = config or PickupSimulationConfig()
    cfg.validate()
    source_path = Path(source_image)
    image = read_image(source_path)
    labels = load_instance_label_map(label_map_path, image.shape[:2])
    instance_ids = [int(value) for value in np.unique(labels) if value > 0]
    if not instance_ids:
        raise ValueError("Label map contains no workpiece instances")
    root = Path(output_dir)
    _make_dataset_dirs(root, True)
    rng = np.random.default_rng(cfg.random_seed)
    records: list[dict] = []
    for index in range(cfg.scene_count):
        ratio = cfg.removal_ratios[index % len(cfg.removal_ratios)]
        remove_count = min(len(instance_ids), int(round(len(instance_ids) * ratio)))
        removed_ids = sorted(
            int(value) for value in rng.choice(instance_ids, size=remove_count, replace=False)
        ) if remove_count else []
        removal_mask = np.isin(labels, removed_ids).astype(np.uint8) * 255
        simulated = (
            cv2.inpaint(image, removal_mask, cfg.inpaint_radius, cv2.INPAINT_TELEA)
            if removed_ids else image.copy()
        )
        kept_labels = labels.copy()
        kept_labels[np.isin(kept_labels, removed_ids)] = 0
        simulated, transformed_labels, augmentation = augment_image_and_labels(
            simulated, kept_labels, rng, cfg.augmentation
        )
        stem = f"pickup_{index:05d}"
        write_image(root / "images" / f"{stem}.jpg", simulated)
        objects = _write_labels_and_metadata(root, stem, transformed_labels)
        metadata = {
            "source_image": str(source_path),
            "mode": "pickup_simulation",
            "removal_ratio": ratio,
            "removed_source_ids": removed_ids,
            "augmentation": augmentation,
            "objects": objects,
        }
        (root / "metadata" / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append({"image": f"images/{stem}.jpg", **metadata})
    return _finish_generation(root, cfg, records, len(instance_ids))


def generate_visual_twin_scenes(
    source_image: str | Path,
    label_map_path: str | Path,
    output_dir: str | Path,
    *,
    background_image: str | Path | None = None,
    asset_metadata_path: str | Path | None = None,
    config: VisualTwinConfig | None = None,
) -> FactoryGenerationResult:
    """Compose non-overlapping scenes from reviewed real workpiece masks.

    Optional asset metadata is JSON keyed by source instance ID. Entries may
    provide ``class_name``, ``template_id`` and ``reference_angle_deg``. The
    reference angle is required for a semantic 0-360 degree ground truth;
    otherwise the generator only promises the 0-180 degree geometric axis.
    """
    cfg = config or VisualTwinConfig()
    cfg.validate()
    source_path = Path(source_image)
    source = read_image(source_path)
    labels = load_instance_label_map(label_map_path, source.shape[:2])
    metadata_source = asset_metadata_path
    if metadata_source is None:
        sidecar = Path(label_map_path).with_suffix(".json")
        metadata_source = sidecar if sidecar.is_file() else None
    metadata = _load_asset_metadata(metadata_source)
    assets = _extract_assets(source, labels, metadata)
    if not assets:
        raise ValueError("Label map contains no usable workpiece instances")

    if background_image is None:
        removal_mask = np.where(labels > 0, 255, 0).astype(np.uint8)
        background = cv2.inpaint(
            source, removal_mask, cfg.background_inpaint_radius, cv2.INPAINT_TELEA
        )
        background_source = "inpainted_source"
    else:
        background_path = Path(background_image)
        background = read_image(background_path)
        if background.shape[:2] != source.shape[:2]:
            background = cv2.resize(
                background, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA
            )
        background_source = str(background_path)

    root = Path(output_dir)
    _make_dataset_dirs(root, True)
    rng = np.random.default_rng(cfg.random_seed)
    augmentation_config = _visual_twin_augmentation(cfg.profile)
    records: list[dict] = []
    total_instances = 0
    for scene_index in range(cfg.scene_count):
        requested = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        scene, scene_labels, attributes = _compose_scene(
            background, assets, requested, rng, cfg
        )
        scene, transformed_labels, augmentation = augment_image_and_labels(
            scene, scene_labels, rng, augmentation_config
        )
        global_rotation = float(augmentation["rotation_deg"])
        for item in attributes.values():
            if item.get("directed_angle_deg") is not None:
                item["directed_angle_deg"] = (
                    float(item["directed_angle_deg"]) + global_rotation
                ) % 360.0

        stem = f"visual_twin_{scene_index:05d}"
        write_image(root / "images" / f"{stem}.jpg", scene)
        objects = _write_labels_and_metadata(root, stem, transformed_labels)
        for obj in objects:
            obj.update(attributes.get(int(obj["object_id"]), {}))
        total_instances += len(objects)
        record = {
            "image": f"images/{stem}.jpg",
            "label_map": f"label_maps/{stem}.labels.png",
            "source_image": str(source_path),
            "background_source": background_source,
            "mode": "visual_twin_composition",
            "requested_object_count": requested,
            "generated_object_count": len(objects),
            "augmentation": augmentation,
            "objects": objects,
        }
        (root / "metadata" / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
    result = _finish_generation(root, cfg, records, len(assets))
    summary = {
        "scene_count": result.image_count,
        "source_asset_count": len(assets),
        "generated_instance_count": total_instances,
        "profile": cfg.profile,
        "angle_definition": {
            "axis_angle_deg": "0-180 degree undirected long-axis angle in image coordinates",
            "directed_angle_deg": "0-360 degree semantic angle; null unless a reference angle is supplied",
        },
    }
    (root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _load_asset_metadata(path: str | Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = payload.get("objects", payload)
    if isinstance(objects, list):
        return {
            int(item["source_object_id"] if "source_object_id" in item else item["object_id"]): {
                **dict(item),
                "class_name": item.get("class_name", item.get("label", "workpiece")),
            }
            for item in objects
        }
    return {int(key): dict(value) for key, value in objects.items()}


def _extract_assets(image: np.ndarray, labels: np.ndarray, metadata: dict[int, dict]) -> list[dict]:
    assets: list[dict] = []
    height, width = labels.shape
    for source_id in (int(value) for value in np.unique(labels) if value > 0):
        mask = np.where(labels == source_id, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, w, h = cv2.boundingRect(points)
        padding = 4
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
        info = metadata.get(source_id, {})
        reference = info.get("reference_angle_deg")
        assets.append({
            "source_object_id": source_id,
            "image": image[y0:y1, x0:x1].copy(),
            "mask": mask[y0:y1, x0:x1].copy(),
            "class_name": str(info.get("class_name", "workpiece")),
            "template_id": info.get("template_id"),
            "reference_angle_deg": None if reference is None else float(reference) % 360.0,
        })
    return assets


def _compose_scene(background, assets, requested, rng, config):
    scene = background.copy()
    height, width = scene.shape[:2]
    labels = np.zeros((height, width), np.uint16)
    occupied = np.zeros((height, width), np.uint8)
    attributes: dict[int, dict] = {}
    if config.placement_roi is None:
        roi_x, roi_y, roi_width, roi_height = 0, 0, width, height
    else:
        roi_x, roi_y, roi_width, roi_height = config.placement_roi
        if roi_x + roi_width > width or roi_y + roi_height > height:
            raise ValueError("placement_roi exceeds the source image dimensions")
    kernel_size = config.min_gap_px * 2 + 1
    gap_kernel = np.ones((kernel_size, kernel_size), np.uint8) if config.min_gap_px else None
    output_id = 1
    for _ in range(requested):
        asset = assets[int(rng.integers(0, len(assets)))]
        rotation = float(rng.uniform(0.0, 360.0))
        scale = float(rng.uniform(config.min_scale, config.max_scale))
        patch, patch_mask = _transform_asset(asset["image"], asset["mask"], rotation, scale)
        patch_h, patch_w = patch_mask.shape
        if patch_w >= roi_width or patch_h >= roi_height:
            continue
        placed = False
        for _attempt in range(config.placement_attempts):
            x = int(rng.integers(roi_x, roi_x + roi_width - patch_w + 1))
            y = int(rng.integers(roi_y, roi_y + roi_height - patch_h + 1))
            candidate = patch_mask > 0
            occupied_roi = occupied[y:y + patch_h, x:x + patch_w]
            if np.any(occupied_roi[candidate]):
                continue
            scene_roi = scene[y:y + patch_h, x:x + patch_w]
            scene_roi[candidate] = patch[candidate]
            labels_roi = labels[y:y + patch_h, x:x + patch_w]
            labels_roi[candidate] = output_id
            placed_mask = np.zeros_like(occupied)
            placed_mask[y:y + patch_h, x:x + patch_w][candidate] = 255
            expanded = placed_mask if gap_kernel is None else cv2.dilate(placed_mask, gap_kernel)
            occupied[expanded > 0] = 255
            reference = asset["reference_angle_deg"]
            attributes[output_id] = {
                "source_object_id": int(asset["source_object_id"]),
                "class_name": asset["class_name"],
                "template_id": asset["template_id"],
                "rotation_delta_deg": rotation,
                "directed_angle_deg": None if reference is None else (reference + rotation) % 360.0,
                "scale": scale,
            }
            output_id += 1
            placed = True
            break
        if not placed:
            continue
    return scene, labels, attributes


def _transform_asset(image, mask, angle, scale):
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    output_width = max(1, int(np.ceil(height * sine + width * cosine)))
    output_height = max(1, int(np.ceil(height * cosine + width * sine)))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    patch = cv2.warpAffine(
        image, matrix, (output_width, output_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    transformed_mask = cv2.warpAffine(
        mask, matrix, (output_width, output_height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return patch, transformed_mask


def _visual_twin_augmentation(profile: str) -> PhotoAugmentationConfig:
    common = dict(variant_count=1, random_seed=0)
    if profile == "clean":
        return PhotoAugmentationConfig(
            **common, brightness_range=(1.0, 1.0), contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0), color_temperature_shift=0,
            noise_sigma_range=(0.0, 0.0), blur_probability=0.0,
            shadow_probability=0.0, max_rotation_deg=0.0,
            max_translation_px=0.0, jpeg_quality_range=(100, 100),
        )
    if profile == "balanced":
        return PhotoAugmentationConfig(
            **common, brightness_range=(0.85, 1.15), contrast_range=(0.9, 1.1),
            gamma_range=(0.9, 1.1), color_temperature_shift=12,
            noise_sigma_range=(0.0, 6.0), blur_probability=0.25,
            shadow_probability=0.45, max_rotation_deg=0.6,
            max_translation_px=2.0, jpeg_quality_range=(90, 100),
        )
    return PhotoAugmentationConfig(**common)


def augment_image_and_labels(image, labels, rng, config):
    """Apply deterministic geometric and photometric camera effects."""
    height, width = image.shape[:2]
    angle = float(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg))
    tx = float(rng.uniform(-config.max_translation_px, config.max_translation_px))
    ty = float(rng.uniform(-config.max_translation_px, config.max_translation_px))
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[:, 2] += (tx, ty)
    result = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    output_labels = None if labels is None else cv2.warpAffine(
        labels, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.uint16)

    brightness = float(rng.uniform(*config.brightness_range))
    contrast = float(rng.uniform(*config.contrast_range))
    gamma = float(rng.uniform(*config.gamma_range))
    temperature = int(rng.integers(-config.color_temperature_shift, config.color_temperature_shift + 1))
    work = result.astype(np.float32)
    mean = work.mean(axis=(0, 1), keepdims=True)
    work = (work - mean) * contrast + mean
    work *= brightness
    work[:, :, 2] += temperature
    work[:, :, 0] -= temperature
    work = np.clip(work, 0, 255)
    lookup = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    result = cv2.LUT(work.astype(np.uint8), lookup)

    shadow_applied = bool(rng.random() < config.shadow_probability)
    if shadow_applied:
        shadow = np.zeros((height, width), np.uint8)
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (max(20, int(width * rng.uniform(0.18, 0.48))), max(20, int(height * rng.uniform(0.15, 0.38))))
        cv2.ellipse(shadow, center, axes, float(rng.uniform(0, 180)), 0, 360, int(rng.integers(45, 115)), -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(width, height) * 0.035)
        result = np.clip(result.astype(np.float32) * (1.0 - shadow[:, :, None] / 360.0), 0, 255).astype(np.uint8)

    noise_sigma = float(rng.uniform(*config.noise_sigma_range))
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    blurred = bool(rng.random() < config.blur_probability)
    if blurred:
        result = cv2.GaussianBlur(result, (5, 5), float(rng.uniform(0.6, 1.6)))
    quality = int(rng.integers(config.jpeg_quality_range[0], config.jpeg_quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result, output_labels, {
        "rotation_deg": angle,
        "translation_px": [tx, ty],
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "color_temperature_shift": temperature,
        "shadow": shadow_applied,
        "noise_sigma": noise_sigma,
        "blur": blurred,
        "jpeg_quality": quality,
    }


def _make_dataset_dirs(root: Path, with_labels: bool) -> None:
    directories = [root / "images", root / "metadata"]
    if with_labels:
        directories.extend([root / "label_maps", root / "instance_masks"])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def _write_labels_and_metadata(root: Path, stem: str, labels: np.ndarray) -> list[dict]:
    remapped = np.zeros(labels.shape, np.uint16)
    objects: list[dict] = []
    for output_id, source_id in enumerate((int(v) for v in np.unique(labels) if v > 0), 1):
        mask = np.where(labels == source_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 4:
            continue
        remapped[mask > 0] = output_id
        mask_path = root / "instance_masks" / f"{stem}.object_{output_id:03d}.png"
        write_image(mask_path, mask)
        moments = cv2.moments(contour)
        center_x = moments["m10"] / moments["m00"]
        center_y_cv = moments["m01"] / moments["m00"]
        x, y, width, height = cv2.boundingRect(contour)
        axis_angle, confidence = min_area_rect_orientation(contour)
        objects.append({
            "object_id": output_id,
            "source_object_id": source_id,
            "center": [float(center_x), float(mask.shape[0] - 1 - center_y_cv)],
            "axis_angle_deg": axis_angle,
            "orientation_confidence": confidence,
            "bbox": [float(x), float(mask.shape[0] - (y + height)), float(width), float(height)],
            "area_px": int(cv2.countNonZero(mask)),
            "mask_path": str(mask_path),
        })
    label_path = root / "label_maps" / f"{stem}.labels.png"
    ok, encoded = cv2.imencode(".png", remapped)
    if not ok:
        raise OSError(f"Could not encode label map: {label_path}")
    encoded.tofile(label_path)
    return objects


def _finish_generation(root, config, records, source_instance_count) -> FactoryGenerationResult:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"config": asdict(config), "source_instance_count": source_instance_count, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preview_path = root / "preview.jpg"
    _write_preview(root, records, preview_path)
    return FactoryGenerationResult(root, len(records), source_instance_count, manifest_path, preview_path)


def _instance_count(labels: np.ndarray | None) -> int:
    return 0 if labels is None else int(np.count_nonzero(np.unique(labels) > 0))


def _write_preview(root: Path, records: list[dict], output_path: Path) -> None:
    columns, tile_width, header_height = 4, 360, 42
    panels: list[np.ndarray] = []
    for record in records[:12]:
        image = read_image(root / record["image"])
        scale = tile_width / image.shape[1]
        tile_height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        label_relative = record.get("label_map")
        if label_relative:
            label_path = root / label_relative
            if label_path.is_file():
                labels = cv2.imdecode(np.fromfile(label_path, np.uint8), cv2.IMREAD_UNCHANGED)
                if labels is not None:
                    for instance_id in (int(v) for v in np.unique(labels) if v > 0):
                        mask = np.where(labels == instance_id, 255, 0).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        scaled = [
                            np.round(contour.astype(np.float32) * scale).astype(np.int32)
                            for contour in contours
                        ]
                        cv2.drawContours(resized, scaled, -1, (255, 190, 35), 2, cv2.LINE_AA)
        panel = np.full((header_height + tile_height, tile_width, 3), (18, 21, 27), np.uint8)
        panel[header_height:] = resized
        name = Path(record["image"]).stem
        mode = record.get("mode", "simulation")
        cv2.putText(panel, name, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 242, 248), 1, cv2.LINE_AA)
        cv2.putText(panel, mode, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 166, 185), 1, cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        return
    panel_height = max(panel.shape[0] for panel in panels)
    while len(panels) % columns:
        panels.append(np.full((panel_height, tile_width, 3), (18, 21, 27), np.uint8))
    rows = []
    for start in range(0, len(panels), columns):
        row_panels = []
        for panel in panels[start:start + columns]:
            if panel.shape[0] < panel_height:
                panel = cv2.copyMakeBorder(
                    panel, 0, panel_height - panel.shape[0], 0, 0,
                    cv2.BORDER_CONSTANT, value=(18, 21, 27),
                )
            row_panels.append(panel)
        rows.append(cv2.hconcat(row_panels))
    write_image(output_path, cv2.vconcat(rows))
