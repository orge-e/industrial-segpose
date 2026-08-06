from k230_runtime.template_builder import assisted_threshold_candidates, suggested_lab_threshold


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
