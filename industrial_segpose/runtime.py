"""Production frame processing: quality gate, matching, tracking and counting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .quality import ImageQualityConfig, ImageQualityReport, evaluate_image_quality
from .template_matching import MultiTemplateMatcher, MultiTemplateResult, draw_multi_template_matches
from .tracking import ConveyorTracker, TrackingConfig, TrackingSnapshot


@dataclass(frozen=True)
class ConveyorFrameResult:
    detection: MultiTemplateResult | None
    quality: ImageQualityReport
    tracking: TrackingSnapshot
    elapsed_ms: float
    skipped: bool


class ConveyorSession:
    def __init__(
        self,
        matcher: MultiTemplateMatcher,
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

    def process_frame(self, image: np.ndarray) -> ConveyorFrameResult:
        started = perf_counter()
        quality = evaluate_image_quality(image, self.quality_config)
        frame_size = (int(image.shape[1]), int(image.shape[0]))
        if self.reject_bad_frames and not quality.passed:
            tracking = self.tracker.update((), frame_size)
            return ConveyorFrameResult(None, quality, tracking, (perf_counter() - started) * 1000.0, True)
        detection = self.matcher.match(image)
        tracking = self.tracker.update(detection.objects, frame_size)
        return ConveyorFrameResult(
            detection,
            quality,
            tracking,
            (perf_counter() - started) * 1000.0,
            False,
        )


def draw_conveyor_frame(
    image: np.ndarray,
    result: ConveyorFrameResult,
    tracking_config: TrackingConfig,
) -> np.ndarray:
    canvas = draw_multi_template_matches(image, result.detection) if result.detection else image.copy()
    height, width = canvas.shape[:2]
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
    status = f"{quality_text}  {result.elapsed_ms:.0f} ms  COUNT={result.tracking.cumulative_total}"
    text_size, baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y0 = max(0, height - text_size[1] - baseline - 12)
    cv2.rectangle(canvas, (0, y0), (min(width - 1, text_size[0] + 16), height - 1), (18, 24, 34), -1)
    cv2.putText(canvas, status, (8, height - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas
