"""Adapter over the existing template domain and persistence APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...io.image_reader import read_image
from ...template_matching.library import TemplateLibraryEntry
from ...template_matching.mask_assist import (
    TemplateMaskQuality,
    analyze_template_mask,
    build_assisted_mask,
)
from ...template_matching.matcher import MatchParameters
from ...template_matching.model import TemplateModel
from .template_repository import LibraryTemplateRepository


@dataclass(frozen=True)
class SegmentationOutput:
    mask: np.ndarray
    method: str
    method_label: str
    quality: TemplateMaskQuality
    algorithm_score: float


class TemplateService:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.repository = LibraryTemplateRepository(self.project_root / "templates")
        # Kept as a read-only compatibility handle for diagnostics and tests.
        self.library = self.repository.library

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        return read_image(path)

    @staticmethod
    def segment_roi(image: np.ndarray, roi_xywh: tuple[int, int, int, int], method: str = "auto") -> SegmentationOutput:
        x, y, width, height = map(int, roi_xywh)
        if x < 0 or y < 0 or width < 5 or height < 5 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError("ROI无效或超出图像范围")
        crop = image[y : y + height, x : x + width].copy()
        result = build_assisted_mask(crop, method)
        quality = analyze_template_mask(result.mask)
        return SegmentationOutput(result.mask, result.method, result.method_label, quality, result.quality_score)

    @staticmethod
    def analyze_mask(mask: np.ndarray) -> TemplateMaskQuality:
        return analyze_template_mask(mask)

    def save_template(
        self,
        *,
        name: str,
        image: np.ndarray,
        source_path: str | Path | None,
        roi_xywh: tuple[int, int, int, int],
        mask: np.ndarray,
        parameters: MatchParameters,
    ) -> TemplateLibraryEntry:
        normalized = name.strip()
        if not normalized:
            raise ValueError("模板名称不能为空")
        if len(normalized) > 48:
            raise ValueError("模板名称最多48个字符")
        parameters.validate()
        model = TemplateModel.from_roi(
            image,
            roi_xywh,
            normalized,
            str(source_path) if source_path else None,
            mask,
            tighten_mask=True,
        )
        entry = self.repository.add(model, parameters)
        return entry

    def template_count(self) -> int:
        return self.repository.count()
