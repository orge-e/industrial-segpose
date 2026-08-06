"""Small dependency-free tracker and conveyor crossing counter."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable

from .template_matching.multi_matcher import RecognizedObject


@dataclass(frozen=True)
class TrackingConfig:
    max_match_distance_px: float = 140.0
    max_missed_frames: int = 8
    min_confirmed_frames: int = 2
    line_axis: str = "x"
    line_position: float = 0.5
    direction: str = "positive"
    crossing_hysteresis_px: float = 6.0

    def validate(self) -> None:
        if self.max_match_distance_px <= 0 or self.max_missed_frames < 0 or self.min_confirmed_frames < 1:
            raise ValueError("Invalid tracking lifetime or distance")
        if self.line_axis not in {"x", "y"}:
            raise ValueError("line_axis must be x or y")
        if not 0.0 < self.line_position < 1.0:
            raise ValueError("line_position must be in (0, 1)")
        if self.direction not in {"positive", "negative", "both"}:
            raise ValueError("direction must be positive, negative or both")
        if self.crossing_hysteresis_px < 0:
            raise ValueError("crossing_hysteresis_px cannot be negative")


@dataclass
class _Track:
    track_id: int
    template_name: str
    center_x: float
    center_y: float
    angle_deg: float
    last_frame: int
    hits: int = 1
    missed: int = 0
    counted: bool = False
    stable_side: int = 0


@dataclass(frozen=True)
class TrackObservation:
    object_id: int
    track_id: int
    template_name: str
    center_x: float
    center_y: float
    angle_deg: float
    counted_now: bool
    counted: bool


@dataclass(frozen=True)
class TrackingSnapshot:
    observations: tuple[TrackObservation, ...]
    cumulative_total: int
    counts_by_template: dict[str, int]
    active_track_count: int


class ConveyorTracker:
    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self.config.validate()
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame_index = 0
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0
        self._counts.clear()

    def update(
        self,
        objects: Iterable[RecognizedObject],
        frame_size: tuple[int, int],
    ) -> TrackingSnapshot:
        self._frame_index += 1
        detections = [item for item in objects if item.classification_status == "confirmed"]
        available_tracks = set(self._tracks)
        assignments: dict[int, int] = {}
        pairs: list[tuple[float, int, int]] = []
        for detection_index, item in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.template_name != item.template_name:
                    continue
                distance = hypot(track.center_x - item.center_x, track.center_y - item.center_y)
                if distance <= self.config.max_match_distance_px:
                    pairs.append((distance, detection_index, track_id))
        used_detections: set[int] = set()
        for _, detection_index, track_id in sorted(pairs):
            if detection_index in used_detections or track_id not in available_tracks:
                continue
            assignments[detection_index] = track_id
            used_detections.add(detection_index)
            available_tracks.remove(track_id)

        for track_id in available_tracks:
            self._tracks[track_id].missed += 1
        for track_id in [key for key, item in self._tracks.items() if item.missed > self.config.max_missed_frames]:
            del self._tracks[track_id]

        observations: list[TrackObservation] = []
        for detection_index, item in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                track = _Track(
                    track_id,
                    item.template_name,
                    item.center_x,
                    item.center_y,
                    item.angle_deg,
                    self._frame_index,
                    stable_side=self._side(item.center_x, item.center_y, frame_size),
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                track.center_x = item.center_x
                track.center_y = item.center_y
                track.angle_deg = item.angle_deg
                track.last_frame = self._frame_index
                track.hits += 1
                track.missed = 0
            current_side = self._side(item.center_x, item.center_y, frame_size)
            counted_now = False
            if current_side != 0 and track.stable_side != 0 and current_side != track.stable_side:
                direction_ok = (
                    self.config.direction == "both"
                    or (self.config.direction == "positive" and track.stable_side < current_side)
                    or (self.config.direction == "negative" and track.stable_side > current_side)
                )
                if direction_ok and not track.counted and track.hits >= self.config.min_confirmed_frames:
                    track.counted = True
                    counted_now = True
                    self._counts[track.template_name] = self._counts.get(track.template_name, 0) + 1
                track.stable_side = current_side
            elif current_side != 0:
                track.stable_side = current_side
            observations.append(TrackObservation(
                item.object_id,
                track.track_id,
                track.template_name,
                item.center_x,
                item.center_y,
                item.angle_deg,
                counted_now,
                track.counted,
            ))
        return TrackingSnapshot(
            tuple(observations),
            sum(self._counts.values()),
            dict(self._counts),
            len(self._tracks),
        )

    def _side(self, center_x: float, center_y: float, frame_size: tuple[int, int]) -> int:
        width, height = frame_size
        coordinate = center_x if self.config.line_axis == "x" else center_y
        extent = width if self.config.line_axis == "x" else height
        delta = coordinate - extent * self.config.line_position
        if delta > self.config.crossing_hysteresis_px:
            return 1
        if delta < -self.config.crossing_hysteresis_px:
            return -1
        return 0
