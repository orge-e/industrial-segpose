"""Canonical records exchanged by vision, calibration and task control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from shared_protocol import auto_pick_allowed, flag_names


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class PickPoint:
    pixel: Point2D
    score: float
    safe_radius_px: float
    local_mm: Point2D | None = None
    global_mm: Point2D | None = None


@dataclass
class TargetRecord:
    target_id: int
    class_id: str
    template_id: str
    template_name: str
    scan_id: int
    view_id: int
    pixel_center: Point2D
    theta_deg: float
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    rotated_rect: list[list[float]] = field(default_factory=list)
    area_px: float = 0.0
    perimeter_px: float = 0.0
    primary_pick_point: PickPoint | None = None
    candidate_pick_points: list[PickPoint] = field(default_factory=list)
    local_position_mm: Point2D | None = None
    global_position_mm: Point2D | None = None
    flags: int = 0
    reachable: bool = True
    status: str = "detected"
    created_at_ms: int = 0

    @property
    def auto_pick_allowed(self) -> bool:
        return self.reachable and auto_pick_allowed(self.flags)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flag_names"] = flag_names(self.flags)
        payload["auto_pick_allowed"] = self.auto_pick_allowed
        return payload


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    recipe_name: str
    template_ids: tuple[str, ...]
    camera_profile: str
    segmentation_profile: str
    matching_profile: str
    pick_head_profile: str
    calibration_profile: str
    target_sort_profile: str
    rephoto_policy: str = "on_quality_failure"
    version: int = 1
    crc: str = ""


@dataclass
class DetectionResultRecord:
    scan_id: int
    view_id: int
    frame_id: int
    timestamp_ms: int
    axis_snapshot_mm: Point2D
    image_quality: dict[str, Any]
    targets: list[TargetRecord]
    processing_time_ms: float
    algorithm_version: str
    recipe_version: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = [target.to_dict() for target in self.targets]
        payload["target_count"] = len(self.targets)
        return payload
