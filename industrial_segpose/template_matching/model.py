"""Persistent template model stored as JSON metadata plus a PNG image."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from ..io.image_reader import read_image, write_image


@dataclass
class TemplateModel:
    name: str
    image: np.ndarray
    source_image: str | None = None
    roi_xywh: tuple[int, int, int, int] | None = None
    created_at: str | None = None
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.image is None or self.image.size == 0:
            raise ValueError("Template image is empty")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Template image must be a BGR color image")
        if self.image.shape[0] < 5 or self.image.shape[1] < 5:
            raise ValueError("Template must be at least 5 x 5 pixels")
        if not self.name.strip():
            raise ValueError("Template name cannot be empty")
        if self.mask is None:
            self.mask = np.full(self.image.shape[:2], 255, dtype=np.uint8)
        else:
            mask = np.asarray(self.mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape != self.image.shape[:2]:
                raise ValueError("Template mask size must match the template image")
            self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(self.mask)) < 25:
            raise ValueError("Template mask must contain at least 25 pixels")

    @property
    def reference_center_xy(self) -> tuple[float, float]:
        moments = cv2.moments(self.mask, binaryImage=True)
        if abs(moments["m00"]) > 1e-9:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
        return self.image.shape[1] / 2.0, self.image.shape[0] / 2.0

    @property
    def contour(self) -> np.ndarray:
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 2), dtype=np.float32)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    @classmethod
    def from_roi(
        cls,
        source: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
        name: str,
        source_image: str | None = None,
        mask: np.ndarray | None = None,
    ) -> "TemplateModel":
        x, y, width, height = map(int, roi_xywh)
        image_height, image_width = source.shape[:2]
        if width < 5 or height < 5:
            raise ValueError("Selected ROI must be at least 5 x 5 pixels")
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise ValueError("Selected ROI is outside the source image")
        crop = source[y : y + height, x : x + width].copy()
        crop_mask = None
        if mask is not None:
            candidate = np.asarray(mask)
            if candidate.shape[:2] == source.shape[:2]:
                candidate = candidate[y : y + height, x : x + width]
            if candidate.shape[:2] != (height, width):
                raise ValueError("Selection mask must match either the source image or selected ROI")
            crop_mask = candidate.copy()
        return cls(
            name=name.strip(),
            image=crop,
            source_image=source_image,
            roi_xywh=(x, y, width, height),
            mask=crop_mask,
        )

    def save(self, metadata_path: str | Path) -> Path:
        path = Path(metadata_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        image_path = path.with_suffix(".png")
        mask_path = path.with_name(f"{path.stem}.mask.png")
        write_image(image_path, self.image)
        write_image(mask_path, self.mask)
        created_at = self.created_at or datetime.now(timezone.utc).isoformat()
        center_x, center_y = self.reference_center_xy
        payload = {
            "format_version": 2,
            "name": self.name,
            "image_file": image_path.name,
            "mask_file": mask_path.name,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "reference_center_xy": [center_x, center_y],
            "mask_area_px": int(np.count_nonzero(self.mask)),
            "contour_points": self.contour.astype(float).tolist(),
            "source_image": self.source_image,
            "roi_xywh": list(self.roi_xywh) if self.roi_xywh else None,
            "created_at": created_at,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.created_at = created_at
        return path

    @classmethod
    def load(cls, metadata_path: str | Path) -> "TemplateModel":
        path = Path(metadata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Template metadata does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version not in {1, 2}:
            raise ValueError("Unsupported template format version")
        image_file = payload.get("image_file")
        if not isinstance(image_file, str) or not image_file:
            raise ValueError("Template metadata has no image_file")
        image = read_image(path.parent / image_file)
        expected = (int(payload.get("height", -1)), int(payload.get("width", -1)))
        if image.shape[:2] != expected:
            raise ValueError(
                f"Template image size {image.shape[1]}x{image.shape[0]} does not match metadata"
            )
        roi = payload.get("roi_xywh")
        mask = None
        if version == 2:
            mask_file = payload.get("mask_file")
            if not isinstance(mask_file, str) or not mask_file:
                raise ValueError("Template metadata has no mask_file")
            mask = read_image(path.parent / mask_file)[:, :, 0]
        return cls(
            name=str(payload.get("name", path.stem)),
            image=image,
            mask=mask,
            source_image=payload.get("source_image"),
            roi_xywh=tuple(map(int, roi)) if roi else None,
            created_at=payload.get("created_at"),
        )
