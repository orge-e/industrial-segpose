"""Rotation-aware image template creation and matching."""

from .matcher import MatchParameters, MatchSceneContext, TemplateMatch, TemplateMatcher
from .feature_router import FeatureModeRecommendation, recommend_feature_mode
from .library import LoadedTemplateEntry, TemplateLibrary, TemplateLibraryEntry
from .mask_assist import AutomaticTemplateSelection, AssistedMaskResult, METHOD_LABELS, TemplateMaskQuality, analyze_template_mask, build_assisted_mask, locate_assisted_template
from .model import TemplateModel
from .multi_matcher import MultiTemplateMatcher, MultiTemplateResult, RecognizedObject, TemplateCandidateScore, draw_multi_template_matches
from .result_writer import write_multi_template_result
from .white_textile import WhiteTextileParameters, WhiteTextileResult, contour_template_feature, save_white_textile_diagnostics, segment_white_textile, white_cut_seam_feature

__all__ = ["AutomaticTemplateSelection", "AssistedMaskResult", "FeatureModeRecommendation", "LoadedTemplateEntry", "METHOD_LABELS", "MatchParameters", "MatchSceneContext", "MultiTemplateMatcher", "MultiTemplateResult", "RecognizedObject", "TemplateCandidateScore", "TemplateLibrary", "TemplateLibraryEntry", "TemplateMaskQuality", "TemplateMatch", "TemplateMatcher", "TemplateModel", "WhiteTextileParameters", "WhiteTextileResult", "analyze_template_mask", "build_assisted_mask", "contour_template_feature", "locate_assisted_template", "draw_multi_template_matches", "recommend_feature_mode", "save_white_textile_diagnostics", "segment_white_textile", "white_cut_seam_feature", "write_multi_template_result"]
