"""Adapters that normalize external datasets into the internal DB schema."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..experiment_db import ExperimentDatabase
from ..io.image_reader import read_image, write_image
from ..measurement.orientation import min_area_rect_orientation


class DatasetAdapter(Protocol):
    def import_dataset(
        self, source_dir: str | Path, database: ExperimentDatabase, dataset_name: str
    ) -> int: ...


@dataclass
class MaskDirectoryAdapter:
    """Import `images/*.png` plus `label_maps/<stem>.labels.png` datasets."""

    source_type: str = "public"
    version: str = "1"

    def import_dataset(
        self, source_dir: str | Path, database: ExperimentDatabase, dataset_name: str
    ) -> int:
        root = Path(source_dir)
        image_dir, label_dir = root / "images", root / "label_maps"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError("Adapter expects images/ and label_maps/ directories")
        database.initialize()
        dataset = database.create_dataset(
            dataset_name, self.source_type, version=self.version, description="Mask directory import"
        )
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.labels.png"
            if not label_path.is_file():
                continue
            image = read_image(image_path)
            label_map = cv2.imdecode(np.fromfile(label_path, np.uint8), cv2.IMREAD_UNCHANGED)
            if label_map is None or label_map.shape[:2] != image.shape[:2]:
                raise ValueError(f"Invalid label map for {image_path.name}")
            record = database.register_image(
                dataset.id,
                image_path,
                image.shape[1],
                image.shape[0],
                source="mask_directory_adapter",
                metadata={"label_map_path": str(label_path)},
            )
            for object_index in sorted(int(value) for value in np.unique(label_map) if value > 0):
                mask = np.where(label_map == object_index, 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-9:
                    continue
                center_x = moments["m10"] / moments["m00"]
                center_y_cv = moments["m01"] / moments["m00"]
                x, y, width, height = cv2.boundingRect(contour)
                axis_angle, orientation_confidence = min_area_rect_orientation(contour)
                mask_path = root / "instance_masks" / f"{image_path.stem}.object_{object_index:03d}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                write_image(mask_path, mask)
                database.add_ground_truth(
                    record.id,
                    object_index,
                    center_x=float(center_x),
                    center_y=float(image.shape[0] - 1 - center_y_cv),
                    mask_path=mask_path,
                    axis_angle_deg=axis_angle,
                    directed_angle_deg=None,
                    bbox=[float(x), float(image.shape[0] - (y + height)), float(width), float(height)],
                    metadata={
                        "angles_available": True,
                        "orientation_confidence": orientation_confidence,
                    },
                )
        return dataset.id
