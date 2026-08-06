import math

from k230_runtime.detector import BlobTemplateDetector, robust_lab_threshold
from k230_runtime.tracking import CentroidCounter


class FakeBlob:
    def __init__(self, x, y, width, height, pixels, angle_deg):
        self.values = [x, y, width, height, pixels, x + width / 2, y + height / 2, math.radians(angle_deg)]

    def __getitem__(self, index):
        return self.values[index]

    def x(self): return self.values[0]
    def y(self): return self.values[1]
    def w(self): return self.values[2]
    def h(self): return self.values[3]
    def pixels(self): return self.values[4]
    def cx(self): return self.values[5]
    def cy(self): return self.values[6]
    def rotation(self): return self.values[7]


class CanMVAxisBlob(FakeBlob):
    def major_axis_line(self):
        return (40, 100, 240, 100)

    def minor_axis_line(self):
        return (140, 50, 140, 150)


class FakeFrame:
    def __init__(self, blobs):
        self.blobs = blobs
        self.calls = []

    def find_blobs(self, thresholds, **kwargs):
        self.calls.append((thresholds, kwargs))
        return self.blobs


class FakeLibrary:
    def __init__(self, templates, ambiguity_margin=0.08):
        self.templates = templates
        self.manifest = {"ambiguity_margin": ambiguity_margin}


def _template(template_id="pink", name="Pink"):
    return {
        "template_id": template_id,
        "name": name,
        "enabled": True,
        "width": 200,
        "height": 100,
        "fill_ratio": 0.55,
        "reference_angle_deg": 10.0,
        "reference_center_xy": [100.0, 50.0],
        "pick_point_xy": [110.0, 55.0],
        "parameters": {"scale_min": 0.5, "scale_max": 1.5},
        "segmentation": {
            "lab_thresholds": [[30, 100, 5, 70, -20, 50]],
            "pixels_threshold": 80,
            "area_threshold": 120,
        },
    }


def test_blob_detector_returns_center_relative_angle_and_pick_point():
    detector = BlobTemplateDetector(FakeLibrary([_template()]))
    frame = FakeFrame([FakeBlob(40, 50, 200, 100, 11000, 40)])

    detections = detector.detect(frame)

    assert len(detections) == 1
    result = detections[0]
    assert result["image_center"] == [140.0, 100.0]
    assert result["angle_deg"] == 30.0
    assert result["confidence"] > 0.9
    assert result["pick_point"] != result["image_center"]
    assert frame.calls[0][1]["pixels_threshold"] == 80


def test_blob_detector_handles_canmv_axis_line_methods_without_none_values():
    detector = BlobTemplateDetector(FakeLibrary([_template()]))
    frame = FakeFrame([CanMVAxisBlob(40, 50, 200, 100, 11000, 40)])

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0]["image_center"] == [140.0, 100.0]


def test_multi_template_close_scores_are_ambiguous():
    detector = BlobTemplateDetector(FakeLibrary([_template("a", "A"), _template("b", "B")], 0.08))
    frame = FakeFrame([FakeBlob(10, 20, 200, 100, 11000, 20)])

    results = detector.detect(frame)

    assert len(results) == 1
    assert results[0]["status"] == "ambiguous"
    assert len(results[0]["candidate_templates"]) >= 2


def test_low_shape_score_is_rejected_before_reaching_ui():
    detector = BlobTemplateDetector(FakeLibrary([_template()]))
    frame = FakeFrame([FakeBlob(10, 20, 200, 200, 2000, 20)])

    assert detector.detect(frame) == []


def test_color_threshold_keeps_exposure_tolerance_but_excludes_neutral_background():
    segmentation = {
        "mode": "color",
        "foreground_lab_median": [62.0, 18.0, 7.0],
        "exposure_tolerance_l": 10,
        "chroma_tolerance": 3,
    }

    threshold = robust_lab_threshold(segmentation, [28, 76, -2, 39, -8, 24])

    assert threshold[:2] == (18, 86)
    assert threshold[2] >= 5
    assert not (threshold[2] <= 0 <= threshold[3])


def test_exported_discriminative_channel_restores_background_boundary():
    segmentation = {
        "mode": "color",
        "foreground_lab_median": [50.0, 10.0, 4.0],
        "exposure_tolerance_l": 10,
        "chroma_tolerance": 3,
        "discriminative_channel": {
            "enabled": True,
            "channel": "a",
            "foreground_side": "above",
            "cutoff": 5.0,
        },
    }

    threshold = robust_lab_threshold(segmentation, [30, 70, -2, 30, -8, 20])

    assert threshold[2] == 5


def test_invalid_discriminative_interval_is_rejected():
    segmentation = {
        "mode": "dark_texture",
        "discriminative_channel": {
            "enabled": True,
            "channel": "l",
            "foreground_side": "above",
            "cutoff": 90,
        },
    }

    assert robust_lab_threshold(segmentation, [20, 60, -10, 10, -10, 10]) is None


def test_dark_template_is_skipped_when_scene_has_opposite_background_polarity():
    detector = BlobTemplateDetector(FakeLibrary([]))
    segmentation = {
        "mode": "dark_texture",
        "polarity": "brighter",
        "foreground_l_median": 45.0,
    }

    assert detector._scene_is_compatible(segmentation, 31.0) is True
    assert detector._scene_is_compatible(segmentation, 52.0) is False


def test_tracker_counts_once_when_crossing_line():
    counter = CentroidCounter(max_distance=50, line_axis="x", line_position=100, direction=1)
    first = {"template_id": "a", "template_name": "A", "image_center": [80.0, 40.0]}
    second = {"template_id": "a", "template_name": "A", "image_center": [105.0, 42.0]}
    third = {"template_id": "a", "template_name": "A", "image_center": [120.0, 43.0]}

    assert counter.update([first])[0]["emit"] is False
    assert counter.update([second])[0]["emit"] is True
    assert counter.update([third])[0]["emit"] is False
    assert counter.total_count == 1
    assert counter.counts == {"A": 1}
