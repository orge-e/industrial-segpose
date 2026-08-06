import cv2
import numpy as np

from industrial_segpose.camera import OpenCVCameraSource
from industrial_segpose.quality import ImageQualityConfig, evaluate_image_quality
from industrial_segpose.runtime import ConveyorSession, draw_conveyor_frame
from industrial_segpose.template_matching.multi_matcher import (
    MultiTemplateResult,
    RecognizedObject,
)
from industrial_segpose.tracking import ConveyorTracker, TrackingConfig


def recognized(object_id, x, y, name="A", status="confirmed"):
    return RecognizedObject(
        object_id,
        status,
        "template-a" if status == "confirmed" else None,
        name,
        float(x),
        float(y),
        12.0,
        0.9,
        0.7,
        1.0,
        (0, 220, 0),
        ((x - 10, y - 6), (x + 10, y - 6), (x + 10, y + 6), (x - 10, y + 6)),
        ((x - 10, y - 6), (x + 10, y - 6), (x + 10, y + 6), (x - 10, y + 6)),
        (),
    )


def detection(*objects):
    counts = {}
    for item in objects:
        if item.classification_status == "confirmed":
            counts[item.template_name] = counts.get(item.template_name, 0) + 1
    return MultiTemplateResult(tuple(objects), counts, 0, {}, ())


def textured_frame(width=320, height=200):
    yy, xx = np.indices((height, width))
    gray = ((xx * 3 + yy * 5) % 150 + 45).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_quality_gate_reports_good_overexposed_and_blurred_frames():
    config = ImageQualityConfig(min_sharpness=8.0)
    good = evaluate_image_quality(textured_frame(), config)
    white = evaluate_image_quality(np.full((160, 240, 3), 255, np.uint8), config)
    blurred = evaluate_image_quality(np.full((160, 240, 3), 100, np.uint8), config)
    assert good.passed
    assert "overexposed" in white.issues
    assert "blurred" in blurred.issues
    assert 0.0 <= good.score <= 1.0


def test_conveyor_tracker_counts_a_crossing_only_once():
    config = TrackingConfig(
        max_match_distance_px=80,
        min_confirmed_frames=2,
        line_axis="x",
        line_position=0.5,
        direction="positive",
        crossing_hysteresis_px=3,
    )
    tracker = ConveyorTracker(config)
    snapshots = []
    for x in (55, 80, 96, 104, 125, 145):
        snapshots.append(tracker.update((recognized(1, x, 50),), (200, 100)))
    assert snapshots[-1].cumulative_total == 1
    assert snapshots[-1].counts_by_template == {"A": 1}
    assert sum(item.counted_now for snapshot in snapshots for item in snapshot.observations) == 1
    assert len({snapshot.observations[0].track_id for snapshot in snapshots}) == 1


def test_tracker_does_not_count_wrong_direction_or_ambiguous_objects():
    tracker = ConveyorTracker(TrackingConfig(line_position=0.5, direction="positive", crossing_hysteresis_px=2))
    tracker.update((recognized(1, 150, 50),), (200, 100))
    result = tracker.update((recognized(1, 50, 50), recognized(2, 30, 30, status="ambiguous")), (200, 100))
    assert result.cumulative_total == 0
    assert len(result.observations) == 1


class FakeMatcher:
    def __init__(self):
        self.calls = 0

    def match(self, _image):
        self.calls += 1
        return detection(recognized(1, 60 + self.calls * 30, 50))


def test_conveyor_session_skips_bad_frames_and_draws_overlay():
    matcher = FakeMatcher()
    tracking = TrackingConfig(line_position=0.5, crossing_hysteresis_px=2)
    session = ConveyorSession(matcher, tracking, ImageQualityConfig(min_sharpness=8.0))
    bad = session.process_frame(np.full((100, 200, 3), 255, np.uint8))
    assert bad.skipped
    assert matcher.calls == 0
    first = session.process_frame(textured_frame(200, 100))
    second = session.process_frame(textured_frame(200, 100))
    assert not first.skipped and not second.skipped
    assert matcher.calls == 2
    canvas = draw_conveyor_frame(textured_frame(200, 100), second, tracking)
    assert canvas.shape == (100, 200, 3)


def test_camera_source_text_parsing():
    assert OpenCVCameraSource.from_text("0").source == 0
    assert OpenCVCameraSource.from_text("video.mp4").source == "video.mp4"


def test_camera_source_reads_video_file(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60))
    if not writer.isOpened():
        import pytest
        pytest.skip("MJPG video writer is unavailable")
    writer.write(np.full((60, 80, 3), (20, 80, 160), np.uint8))
    writer.release()
    source = OpenCVCameraSource(str(path))
    with source:
        frame = source.read()
    assert frame.shape == (60, 80, 3)
    assert float(frame[:, :, 2].mean()) > 140
