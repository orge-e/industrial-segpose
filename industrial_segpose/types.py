"""Shared result types."""

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np


@dataclass
class SegmentationResult:
    label_map: np.ndarray
    masks: list[np.ndarray]
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ObjectResult:
    object_id: int
    center_x: float
    center_y: float
    angle_deg: float
    angle_method: str
    angle_reliable: bool
    orientation_confidence: float | None
    area_px: float
    perimeter_px: float
    bbox_xywh: tuple[int, int, int, int]
    rotated_box_points: list[list[float]]
    width_px: float
    height_px: float
    aspect_ratio: float
    touches_border: bool
    contour_points: list[list[float]] = field(default_factory=list)
    axis_angle_deg: float | None = None
    directed_angle_deg: float | None = None
    direction_confidence: float | None = None
    mask_filename: str | None = None
    _mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_mask", None)
        return data


@dataclass
class RejectedInstance:
    source_label: int
    reasons: list[str]
    area_px: float


@dataclass
class ImageResult:
    image_name: str
    image_width: int
    image_height: int
    object_count: int
    objects: list[ObjectResult]
    processing_time_ms: float
    backend: str
    success: bool = True
    error_message: str | None = None
    rejected_instances: list[RejectedInstance] = field(default_factory=list)
    coordinate_system: str = "original_bottom_left_x_right_y_up"
    mask_raster_coordinate_system: str = "original_top_left_row_major"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name, "image_width": self.image_width,
            "image_height": self.image_height, "object_count": self.object_count,
            "backend": self.backend, "processing_time_ms": self.processing_time_ms,
            "success": self.success, "error_message": self.error_message,
            "objects": [obj.to_dict() for obj in self.objects],
            "rejected_instances": [asdict(item) for item in self.rejected_instances],
        }
