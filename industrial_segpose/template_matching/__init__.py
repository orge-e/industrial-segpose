"""Rotation-aware image template creation and matching."""

from .matcher import MatchParameters, TemplateMatch, TemplateMatcher
from .library import LoadedTemplateEntry, TemplateLibrary, TemplateLibraryEntry
from .mask_assist import AutomaticTemplateSelection, AssistedMaskResult, METHOD_LABELS, build_assisted_mask, locate_assisted_template
from .model import TemplateModel
from .multi_matcher import MultiTemplateMatcher, MultiTemplateResult, RecognizedObject, TemplateCandidateScore, draw_multi_template_matches
from .result_writer import write_multi_template_result

__all__ = ["AutomaticTemplateSelection", "AssistedMaskResult", "LoadedTemplateEntry", "METHOD_LABELS", "MatchParameters", "MultiTemplateMatcher", "MultiTemplateResult", "RecognizedObject", "TemplateCandidateScore", "TemplateLibrary", "TemplateLibraryEntry", "TemplateMatch", "TemplateMatcher", "TemplateModel", "build_assisted_mask", "locate_assisted_template", "draw_multi_template_matches", "write_multi_template_result"]
