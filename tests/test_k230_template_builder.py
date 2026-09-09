import json
from pathlib import Path

from k230_runtime.template_builder import (
    assisted_threshold_candidates,
    contrast_threshold_candidates,
    suggested_lab_threshold,
)


class Statistics:
    def l_mean(self): return 62
    def l_stdev(self): return 8
    def a_mean(self): return 24
    def a_stdev(self): return 6
    def b_mean(self): return 5
    def b_stdev(self): return 4


def test_suggested_lab_threshold_is_bounded_and_uses_variance():
    threshold = suggested_lab_threshold(Statistics(), sigma=2.5)
    assert threshold == [42, 82, 9, 39, -5, 15]


class ExtremeStatistics:
    def l_mean(self): return 99
    def l_stdev(self): return 20
    def a_mean(self): return -125
    def a_stdev(self): return 20
    def b_mean(self): return 126
    def b_stdev(self): return 20


def test_suggested_lab_threshold_clamps_to_canmv_ranges():
    assert suggested_lab_threshold(ExtremeStatistics()) == [49, 100, -128, -75, 76, 127]


def test_assisted_candidates_include_neutral_excluding_color_directions():
    candidates = assisted_threshold_candidates([20, 90, -10, 10, -8, 12])

    assert candidates[0][0] == "assisted_roi"
    assert any(mode == "color" and threshold[2] >= 4 for mode, threshold in candidates)
    assert any(mode == "color" and threshold[3] <= -4 for mode, threshold in candidates)


class FixedStatistics:
    def __init__(self, l, a, b):
        self.values = {"l": l, "a": a, "b": b}

    def l_mean(self): return self.values["l"]
    def a_mean(self): return self.values["a"]
    def b_mean(self): return self.values["b"]


class ContrastFrame:
    def get_statistics(self, roi=None):
        # The large centre seed is a light, slightly red workpiece. Corner
        # samples represent a darker neutral conveyor/background.
        return FixedStatistics(72, 8, 0) if roi[2] > 24 else FixedStatistics(42, 0, 0)


def test_contrast_candidates_adapt_to_foreground_and_background_without_fixed_colour():
    candidates = contrast_threshold_candidates(ContrastFrame(), (10, 20, 160, 120))

    assert any(mode == "contrast_l" and threshold[0] > 42 for mode, threshold in candidates)
    assert any(mode == "contrast_a" and threshold[2] > 0 for mode, threshold in candidates)


class PreviewBlob:
    def __init__(self, pixels=3000, rect=(40, 35, 80, 45)):
        self._pixels = pixels
        self._rect = rect

    def pixels(self): return self._pixels
    def rect(self): return self._rect


class PreviewImage:
    def binary(self, thresholds):
        self.thresholds = thresholds

    def to_grayscale(self, copy=False):
        return self


class PreviewFrame:
    def __init__(self, blob):
        self.blob = blob

    def get_statistics(self, roi=None):
        return Statistics()

    def width(self): return 200
    def height(self): return 140

    def find_blobs(self, thresholds, **kwargs):
        return [self.blob]

    def copy(self, roi=None):
        return PreviewImage()


def test_template_preview_exposes_mask_coverage_and_quality():
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    preview = OnDeviceTemplateBuilder("/templates").preview(
        PreviewFrame(PreviewBlob()), (20, 20, 140, 80)
    )

    assert preview["valid"] is True
    assert preview["quality"] == "good"
    assert 0.20 < preview["coverage"] < 0.30
    assert preview["mask"] is not None


def test_template_preview_rejects_background_flood():
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    preview = OnDeviceTemplateBuilder("/templates").preview(
        PreviewFrame(PreviewBlob(pixels=10500, rect=(20, 20, 140, 80))), (20, 20, 140, 80)
    )

    assert preview["valid"] is False
    assert preview["quality"] == "invalid"


def test_template_preview_rejects_fragmented_candidate_touching_all_roi_edges():
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    preview = OnDeviceTemplateBuilder("/templates").preview(
        PreviewFrame(PreviewBlob(pixels=5600, rect=(20, 20, 140, 80))), (20, 20, 140, 80)
    )

    assert preview["valid"] is False
    assert preview["reason"] in ("touches_roi_edges", "background_flood")


def test_template_preview_rejects_small_highlight_or_thread_candidate():
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    preview = OnDeviceTemplateBuilder("/templates").preview(
        PreviewFrame(PreviewBlob(pixels=420, rect=(60, 45, 70, 14))), (20, 20, 140, 80)
    )

    assert preview["valid"] is False
    assert preview["reason"] == "foreground_too_small"


def test_template_preview_rejects_real_q1_fragment_from_device_diagnostics():
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    preview = OnDeviceTemplateBuilder("/templates").preview(
        PreviewFrame(PreviewBlob(pixels=2444, rect=(210, 132, 91, 199))),
        (210, 120, 190, 239),
    )

    assert preview["valid"] is False
    assert preview["reason"] == "foreground_too_small"


class DiagnosticImage:
    def __init__(self, width=40, height=30):
        self._width = width
        self._height = height

    def width(self): return self._width
    def height(self): return self._height

    def save(self, path):
        Path(path).write_bytes(b"diagnostic-image")

    def copy(self, roi=None):
        if roi is None:
            return DiagnosticImage(self._width, self._height)
        return DiagnosticImage(roi[2], roi[3])

    def binary(self, _thresholds): return self
    def to_grayscale(self, copy=False): return self


class DiagnosticFrame(DiagnosticImage):
    def get_statistics(self, roi=None): return Statistics()
    def find_blobs(self, _thresholds, **_kwargs): return [PreviewBlob(pixels=80, rect=(2, 3, 12, 8))]


def test_template_diagnostics_keep_complete_latest_segmentation_trail(tmp_path):
    from k230_runtime.template_builder import OnDeviceTemplateBuilder

    root = tmp_path / "diagnostics" / "latest_template"
    root.mkdir(parents=True)
    (root / "obsolete.txt").write_text("old", encoding="utf-8")
    builder = OnDeviceTemplateBuilder(
        str(tmp_path / "templates"), diagnostic_root=str(root)
    )
    frame = DiagnosticFrame(64, 48)
    intermediate = DiagnosticImage(24, 16)
    selected = suggested_lab_threshold(Statistics())

    result = builder._save_diagnostics(
        frame, (5, 7, 24, 16), intermediate, intermediate, intermediate,
        intermediate, intermediate, selected, "assisted_roi",
    )

    assert result == str(root)
    assert not (root / "obsolete.txt").exists()
    assert (root / "00_frozen_frame.jpg").is_file()
    assert (root / "01_roi_original.bmp").is_file()
    assert (root / "21_final_mask.bmp").is_file()
    payload = json.loads((root / "segmentation_debug.json").read_text(encoding="utf-8"))
    assert payload["source_size"] == [64, 48]
    assert payload["roi_xywh"] == [5, 7, 24, 16]
    assert payload["candidate_count"] == 5
    assert sum(candidate["selected"] for candidate in payload["candidates"]) == 1
