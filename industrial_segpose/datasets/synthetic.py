"""Deterministic synthetic multi-object scenes with complete ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..experiment_db import ExperimentDatabase
from ..io.image_reader import write_image


PROFILES = {
    "clean",
    "rotation",
    "scale_variation",
    "color_variation",
    "lighting",
    "noise",
    "blur",
    "dense_separated",
    "edge_partial",
    "production_mixed",
    # Retained only for explicit algorithm stress tests. These two profiles
    # are not part of the production benchmark because factory workpieces do
    # not overlap or stack.
    "touching",
    "mixed",
}


@dataclass(frozen=True)
class SyntheticConfig:
    image_width: int = 512
    image_height: int = 384
    image_count: int = 10
    min_objects: int = 1
    max_objects: int = 4
    min_scale: float = 0.75
    max_scale: float = 1.25
    profile: str = "clean"
    random_seed: int = 42
    class_name: str = "synthetic_textile"

    def validate(self) -> None:
        if self.image_width < 96 or self.image_height < 96:
            raise ValueError("Synthetic image dimensions must be at least 96 pixels")
        if self.image_count < 1:
            raise ValueError("image_count must be positive")
        if not 1 <= self.min_objects <= self.max_objects:
            raise ValueError("Object count range is invalid")
        if not 0.2 <= self.min_scale <= self.max_scale <= 3.0:
            raise ValueError("Scale range is invalid")
        if self.profile not in PROFILES:
            raise ValueError(f"Unsupported synthetic profile: {self.profile}")


@dataclass(frozen=True)
class SyntheticGenerationResult:
    dataset_id: int
    output_dir: Path
    image_count: int
    object_count: int
    config_path: Path


class SyntheticDatasetGenerator:
    def __init__(self, config: SyntheticConfig):
        config.validate()
        self.config = config

    def generate(
        self,
        output_dir: str | Path,
        database: ExperimentDatabase,
        *,
        dataset_name: str | None = None,
        dataset_version: str = "1",
    ) -> SyntheticGenerationResult:
        root = Path(output_dir)
        image_dir = root / "images"
        mask_dir = root / "instance_masks"
        label_dir = root / "label_maps"
        metadata_dir = root / "metadata"
        for directory in (image_dir, mask_dir, label_dir, metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)
        config_path = root / "generator_config.json"
        config_path.write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        database.initialize()
        dataset = database.create_dataset(
            dataset_name or f"synthetic_{self.config.profile}",
            "synthetic",
            version=dataset_version,
            description=f"Deterministic synthetic profile: {self.config.profile}",
            generator_config=asdict(self.config),
        )
        rng = np.random.default_rng(self.config.random_seed)
        total_objects = 0
        for image_index in range(self.config.image_count):
            scene, label_map, objects = self._generate_scene(rng, image_index)
            stem = f"scene_{image_index:05d}"
            image_path = image_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.labels.png"
            metadata_path = metadata_dir / f"{stem}.json"
            write_image(image_path, scene)
            self._write_png(label_path, label_map.astype(np.uint16))
            metadata_path.write_text(
                json.dumps({"profile": self.config.profile, "objects": objects}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            image_record = database.register_image(
                dataset.id,
                image_path,
                self.config.image_width,
                self.config.image_height,
                source="synthetic_generator",
                metadata={
                    "profile": self.config.profile,
                    "scene_index": image_index,
                    "label_map_path": str(label_path),
                    "scene_metadata_path": str(metadata_path),
                },
            )
            for item in objects:
                object_index = int(item["object_id"])
                visible_mask = np.where(label_map == object_index, 255, 0).astype(np.uint8)
                mask_path = mask_dir / f"{stem}.object_{object_index:03d}.png"
                write_image(mask_path, visible_mask)
                database.add_ground_truth(
                    image_record.id,
                    object_index,
                    center_x=float(item["center"][0]),
                    center_y=float(item["center"][1]),
                    mask_path=mask_path,
                    axis_angle_deg=float(item["axis_angle_deg"]),
                    directed_angle_deg=float(item["directed_angle_deg"]),
                    bbox=item["bbox"],
                    class_name=self.config.class_name,
                    metadata={
                        "scale": item["scale"],
                        "color_bgr": item["color_bgr"],
                        "partially_occluded": item["partially_occluded"],
                    },
                )
                total_objects += 1
        return SyntheticGenerationResult(
            dataset.id, root, self.config.image_count, total_objects, config_path
        )

    def _generate_scene(
        self, rng: np.random.Generator, image_index: int
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        cfg = self.config
        background_level = int(rng.integers(22, 42))
        scene = np.full((cfg.image_height, cfg.image_width, 3), background_level, np.uint8)
        label_map = np.zeros((cfg.image_height, cfg.image_width), np.uint16)
        count = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        placements = self._placements(rng, count)
        objects: list[dict[str, Any]] = []
        for object_id, (center_x_cv, center_y_cv) in enumerate(placements, 1):
            if cfg.profile in {"rotation", "production_mixed", "mixed"}:
                directed_angle = float(rng.uniform(0.0, 360.0))
            else:
                directed_angle = float(rng.uniform(-18.0, 18.0) % 360.0)
            scale = float(rng.uniform(cfg.min_scale, cfg.max_scale))
            color_limit = 45 if cfg.profile in {"color_variation", "production_mixed"} else 14
            color_shift = int(rng.integers(-color_limit, color_limit + 1))
            color = tuple(int(np.clip(value + color_shift, 0, 255)) for value in (185, 205, 225))
            mask = self._object_mask(center_x_cv, center_y_cv, directed_angle, scale)
            scene[mask > 0] = color
            label_map[mask > 0] = object_id
            objects.append(
                self._object_metadata(object_id, mask, center_x_cv, center_y_cv, directed_angle, scale, color)
            )
        self._apply_profile(scene, rng)
        # Recompute visible boxes and occlusion flags after later objects have overwritten labels.
        for item in objects:
            visible = np.where(label_map == item["object_id"], 255, 0).astype(np.uint8)
            points = cv2.findNonZero(visible)
            item["partially_occluded"] = cv2.countNonZero(visible) < item.pop("full_area")
            if points is not None:
                x, y, width, height = cv2.boundingRect(points)
                item["bbox"] = [float(x), float(cfg.image_height - (y + height)), float(width), float(height)]
        return scene, label_map, objects

    def _placements(self, rng: np.random.Generator, count: int) -> list[tuple[float, float]]:
        cfg = self.config
        margin_x, margin_y = 70, 55
        if cfg.profile in {"touching", "mixed"}:
            start_x = float(rng.uniform(margin_x + 30, max(margin_x + 31, cfg.image_width - margin_x - 70 * count)))
            center_y = float(rng.uniform(margin_y, cfg.image_height - margin_y))
            return [(start_x + index * 66.0, center_y + float(rng.uniform(-8, 8))) for index in range(count)]
        if cfg.profile == "dense_separated":
            # One compact row with a guaranteed visual gap. The production
            # suite limits this profile's scale so objects remain independent.
            xs = np.linspace(margin_x, cfg.image_width - margin_x, count)
            center_y = float(rng.uniform(95, cfg.image_height - 95))
            return [
                (float(x), center_y + float(rng.uniform(-5, 5)))
                for x in xs
            ]
        if cfg.profile == "production_mixed":
            columns = 2 if count > 1 else 1
            rows = int(np.ceil(count / columns))
            xs = np.linspace(95, cfg.image_width - 95, columns)
            ys = np.linspace(80, cfg.image_height - 80, rows)
            grid = [(float(x), float(y)) for y in ys for x in xs][:count]
            rng.shuffle(grid)
            return [
                (x + float(rng.uniform(-7, 7)), y + float(rng.uniform(-7, 7)))
                for x, y in grid
            ]
        if cfg.profile == "edge_partial":
            anchors = [
                (30.0, cfg.image_height * 0.30),
                (cfg.image_width - 30.0, cfg.image_height * 0.30),
                (cfg.image_width * 0.30, 26.0),
                (cfg.image_width * 0.70, cfg.image_height - 26.0),
            ]
            rng.shuffle(anchors)
            return anchors[:count]
        if count == 1:
            return [(
                float(rng.uniform(margin_x, cfg.image_width - margin_x)),
                float(rng.uniform(margin_y, cfg.image_height - margin_y)),
            )]
        # Production profiles use a jittered grid so the generator guarantees
        # independent workpieces instead of relying on a best-effort distance
        # rejection that can still overlap at large scales.
        columns = 2
        rows = int(np.ceil(count / columns))
        xs = np.linspace(95, cfg.image_width - 95, columns)
        ys = np.linspace(80, cfg.image_height - 80, rows)
        grid = [(float(x), float(y)) for y in ys for x in xs][:count]
        rng.shuffle(grid)
        return [
            (x + float(rng.uniform(-16, 16)), y + float(rng.uniform(-16, 16)))
            for x, y in grid
        ]

    def _object_mask(self, center_x: float, center_y: float, angle: float, scale: float) -> np.ndarray:
        # Asymmetric arrow-like textile profile gives synthetic objects a semantic direction.
        local = np.asarray(
            [[-48, -25], [18, -25], [18, -39], [52, 0], [18, 39], [18, 25], [-48, 25], [-35, 0]],
            np.float32,
        )
        local *= scale
        radians = np.deg2rad(angle)
        # Output +Y points upward, therefore raster Y uses the negative sine term.
        matrix = np.asarray(
            [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]], np.float32
        )
        points = local @ matrix.T + np.asarray([center_x, center_y], np.float32)
        mask = np.zeros((self.config.image_height, self.config.image_width), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        return mask

    def _object_metadata(self, object_id, mask, center_x_cv, center_y_cv, angle, scale, color):
        points = cv2.findNonZero(mask)
        x, y, width, height = cv2.boundingRect(points)
        return {
            "object_id": object_id,
            "center": [float(center_x_cv), float(self.config.image_height - 1 - center_y_cv)],
            "axis_angle_deg": float(angle % 180.0),
            "directed_angle_deg": float(angle % 360.0),
            "bbox": [float(x), float(self.config.image_height - (y + height)), float(width), float(height)],
            "scale": scale,
            "color_bgr": list(color),
            "partially_occluded": False,
            "full_area": int(cv2.countNonZero(mask)),
        }

    def _apply_profile(self, scene: np.ndarray, rng: np.random.Generator) -> None:
        profile = self.config.profile
        if profile in {"lighting", "production_mixed", "mixed"}:
            gradient = np.linspace(0.65, 1.25, scene.shape[1], dtype=np.float32)[None, :, None]
            scene[:] = np.clip(scene.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
            center = (int(rng.integers(0, scene.shape[1])), int(rng.integers(0, scene.shape[0])))
            shadow = np.zeros(scene.shape[:2], np.uint8)
            cv2.ellipse(shadow, center, (scene.shape[1] // 3, scene.shape[0] // 4), 20, 0, 360, 90, -1)
            scene[:] = np.clip(scene.astype(np.float32) * (1.0 - shadow[:, :, None] / 420.0), 0, 255).astype(np.uint8)
        if profile in {"noise", "production_mixed", "mixed"}:
            noise = rng.normal(0.0, 10.0, scene.shape).astype(np.float32)
            scene[:] = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if profile in {"blur", "production_mixed", "mixed"}:
            scene[:] = cv2.GaussianBlur(scene, (7, 7), 1.5)

    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"Could not encode PNG: {path}")
        encoded.tofile(path)
