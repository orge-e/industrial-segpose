import cv2
import numpy as np

from industrial_segpose.template_matching import (
    MatchParameters,
    MultiTemplateMatcher,
    TemplateLibraryEntry,
    TemplateMatcher,
    TemplateModel,
    recommend_feature_mode,
)
from industrial_segpose.template_matching.library import LoadedTemplateEntry


def _polygon_mask(shape=(160, 220)):
    mask = np.zeros(shape, np.uint8)
    points = np.array(
        [[28, 62], [68, 31], [116, 45], [185, 26], [202, 82],
         [166, 112], [124, 101], [91, 140], [43, 123]],
        np.int32,
    )
    cv2.fillPoly(mask, [points], 255)
    return mask


def test_feature_router_selects_template_adaptive_colour_mode():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 170, np.uint8)
    image[mask > 0] = (185, 155, 218)

    recommendation = recommend_feature_mode(TemplateModel("colour", image, mask=mask))

    assert recommendation.mode == "pose_tolerant"
    assert recommendation.confidence > 0.6


def test_feature_router_selects_cut_seam_for_similar_light_material():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 222, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (125, 125, 125), 3, cv2.LINE_AA)

    recommendation = recommend_feature_mode(TemplateModel("white", image, mask=mask))

    assert recommendation.mode == "white_cut_seam"
    assert recommendation.confidence > 0.6


def test_feature_router_keeps_dim_yellowish_white_material_in_cut_seam_mode():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), (137, 145, 143), np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (92, 96, 95), 3, cv2.LINE_AA)

    recommendation = recommend_feature_mode(TemplateModel("dim-white", image, mask=mask))

    assert recommendation.mode == "white_cut_seam"


def test_feature_router_selects_texture_for_dark_neutral_material():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 52, np.uint8)
    yy, xx = np.indices(mask.shape)
    weave = np.where(((xx // 2 + yy // 2) % 2) == 0, 30, 78).astype(np.uint8)
    for channel in range(3):
        image[:, :, channel][mask > 0] = weave[mask > 0]

    recommendation = recommend_feature_mode(TemplateModel("dark", image, mask=mask))

    assert recommendation.mode == "dark_textile"
    assert recommendation.foreground_texture > recommendation.background_texture


def test_feature_router_selects_dark_mode_for_smooth_black_part_on_light_support():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 145, np.uint8)
    image[mask > 0] = 18

    recommendation = recommend_feature_mode(TemplateModel("smooth-dark", image, mask=mask))

    assert recommendation.mode == "dark_textile"


def test_feature_router_selects_white_contour_for_neutral_part_on_gray_support():
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 125, np.uint8)
    image[mask > 0] = 165

    recommendation = recommend_feature_mode(TemplateModel("detached-white", image, mask=mask))

    assert recommendation.mode == "white_cut_seam"


def test_feature_router_falls_back_to_edges_without_background_pixels():
    image = np.full((80, 120, 3), 180, np.uint8)
    model = TemplateModel("full-mask", image)

    assert recommend_feature_mode(model).mode == "edges"
    assert TemplateMatcher(model, MatchParameters(feature_mode="auto")).resolved_feature_mode == "edges"


def test_multi_template_matcher_shares_white_seam_candidates(monkeypatch):
    mask = _polygon_mask()
    image = np.full((*mask.shape, 3), 222, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (112, 112, 112), 4, cv2.LINE_AA)
    params = MatchParameters(
        score_threshold=0.45,
        feature_mode="white_cut_seam",
        angle_min=-180,
        angle_max=178,
        angle_step=2,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=0.05,
    )
    first_model = TemplateModel("white-a", image, mask=mask)
    second_model = TemplateModel("white-b", image.copy(), mask=mask.copy())
    loaded = [
        LoadedTemplateEntry(TemplateLibraryEntry("a", "white-a", "a.json", parameters=params), first_model),
        LoadedTemplateEntry(TemplateLibraryEntry("b", "white-b", "b.json", parameters=params), second_model),
    ]
    calls = 0
    original = TemplateMatcher._pose_candidate_mask

    def counted(self, scene):
        nonlocal calls
        calls += 1
        return original(self, scene)

    monkeypatch.setattr(TemplateMatcher, "_pose_candidate_mask", counted)

    result = MultiTemplateMatcher(loaded).match(image)

    assert calls == 1
    assert all(item["resolved_feature_mode"] == "white_cut_seam" for item in result.used_templates)
