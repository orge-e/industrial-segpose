"""生产运行：串联图像质量门控、匹配、跟踪和计数。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from .quality import ImageQualityConfig, ImageQualityReport, evaluate_image_quality
from ..template_matching import MultiTemplateMatcher, MultiTemplateResult, draw_multi_template_matches
from .tracking import ConveyorTracker, TrackingConfig, TrackingSnapshot


class FrameDetector(Protocol):
    """Detector contract shared by template and colour-first pipelines."""

    def match(self, image: np.ndarray) -> MultiTemplateResult: ...


@dataclass(frozen=True)
class ConveyorFrameResult:
    detection: MultiTemplateResult | None
    quality: ImageQualityReport
    tracking: TrackingSnapshot
    elapsed_ms: float
    skipped: bool
    input_image_size: tuple[int, int] | None = None
    detection_roi: tuple[int, int, int, int] | None = None

    @property
    def current_count(self) -> int:
        return self.detection.object_count if self.detection is not None else 0


class ConveyorSession:
    def __init__(
        self,
        matcher: FrameDetector,
        tracking_config: TrackingConfig | None = None,
        quality_config: ImageQualityConfig | None = None,
        reject_bad_frames: bool = True,
    ):
        self.matcher = matcher
        self.tracker = ConveyorTracker(tracking_config)
        self.quality_config = quality_config or ImageQualityConfig()
        self.reject_bad_frames = bool(reject_bad_frames)

    def reset_counts(self) -> None:
        self.tracker.reset()

    def process_frame(
        self,
        image: np.ndarray,
        detection_roi: tuple[int, int, int, int] | None = None,
    ) -> ConveyorFrameResult:
        if image is None or image.size == 0:
            raise ValueError("Input frame is empty")
        started = perf_counter()
        frame_size = (int(image.shape[1]), int(image.shape[0]))
        roi = _normalize_roi(detection_roi, frame_size)
        if roi is None:
            detection_image = image
        else:
            x, y, width, height = roi
            detection_image = image[y : y + height, x : x + width]
        quality = evaluate_image_quality(detection_image, self.quality_config)
        if self.reject_bad_frames and not quality.passed:
            tracking = self.tracker.update((), frame_size)
            return ConveyorFrameResult(
                None,
                quality,
                tracking,
                (perf_counter() - started) * 1000.0,
                True,
                frame_size,
                roi,
            )
        detection = self.matcher.match(detection_image)
        if roi is not None:
            detection = _restore_roi_coordinates(detection, roi, frame_size)
        tracking = self.tracker.update(detection.objects, frame_size)
        return ConveyorFrameResult(
            detection,
            quality,
            tracking,
            (perf_counter() - started) * 1000.0,
            False,
            frame_size,
            roi,
        )


def _normalize_roi(
    roi: tuple[int, int, int, int] | None,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    frame_width, frame_height = frame_size
    x, y, width, height = (int(round(value)) for value in roi)
    x = min(max(x, 0), frame_width - 1)
    y = min(max(y, 0), frame_height - 1)
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    if width < 32 or height < 32:
        raise ValueError("Detection ROI must be at least 32 x 32 pixels")
    return x, y, width, height


def _restore_roi_coordinates(
    result: MultiTemplateResult,
    roi: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> MultiTemplateResult:
    offset_x, offset_y, _, _ = roi
    objects = tuple(
        replace(
            item,
            center_x=item.center_x + offset_x,
            center_y=item.center_y + offset_y,
            pick_point_x=(item.pick_point_x + offset_x) if item.pick_point_x is not None else None,
            pick_point_y=(item.pick_point_y + offset_y) if item.pick_point_y is not None else None,
            box_points=tuple((x + offset_x, y + offset_y) for x, y in item.box_points),
            contour_points=tuple((x + offset_x, y + offset_y) for x, y in item.contour_points),
        )
        for item in result.objects
    )
    return replace(result, objects=objects, source_image_size=frame_size)


def draw_conveyor_frame(
    image: np.ndarray,
    result: ConveyorFrameResult,
    tracking_config: TrackingConfig,
) -> np.ndarray:
    canvas = draw_multi_template_matches(image, result.detection) if result.detection else image.copy()
    height, width = canvas.shape[:2]
    if result.detection_roi is not None:
        x, y, roi_width, roi_height = result.detection_roi
        cv2.rectangle(
            canvas,
            (x, y),
            (x + roi_width - 1, y + roi_height - 1),
            (90, 235, 140),
            max(2, width // 800),
            cv2.LINE_AA,
        )
    line_color = (40, 220, 255)
    if tracking_config.line_axis == "x":
        coordinate = int(round(width * tracking_config.line_position))
        cv2.line(canvas, (coordinate, 0), (coordinate, height - 1), line_color, max(2, width // 700), cv2.LINE_AA)
    else:
        coordinate = int(round(height * tracking_config.line_position))
        cv2.line(canvas, (0, coordinate), (width - 1, coordinate), line_color, max(2, width // 700), cv2.LINE_AA)
    scale = max(0.65, max(width, height) / 1800.0)
    thickness = max(2, int(round(scale * 1.6)))
    for observation in result.tracking.observations:
        label = f"T{observation.track_id}{' COUNT' if observation.counted_now else ''}"
        point = (int(round(observation.center_x + 10)), int(round(observation.center_y - 10)))
        cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 120), thickness, cv2.LINE_AA)
    quality_text = "QUALITY OK" if result.quality.passed else "QUALITY: " + ",".join(result.quality.issues)
    status = (
        f"{quality_text}  {result.elapsed_ms:.0f} ms  "
        f"NOW={result.current_count}  LINE={result.tracking.cumulative_total}"
    )
    text_size, baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y0 = max(0, height - text_size[1] - baseline - 12)
    cv2.rectangle(canvas, (0, y0), (min(width - 1, text_size[0] + 16), height - 1), (18, 24, 34), -1)
    cv2.putText(canvas, status, (8, height - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas
