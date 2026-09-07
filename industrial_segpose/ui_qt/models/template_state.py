"""Qt 状态模型：描述模板建立页面的步骤、图像和参数状态。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np

from ...template_matching.matcher import MatchParameters
from ...template_matching.mask_assist import TemplateMaskQuality


class TemplateStep(IntEnum):
    SOURCE = 0
    ROI = 1
    MASK = 2
    PARAMETERS = 3
    SAVE = 4


class ViewMode(StrEnum):
    ORIGINAL = "original"
    MASK = "mask"
    OVERLAY = "overlay"


@dataclass(frozen=True)
class TemplatePageState:
    step: TemplateStep = TemplateStep.SOURCE
    name: str = ""
    source_path: Path | None = None
    image: np.ndarray | None = field(default=None, compare=False, repr=False)
    roi_xywh: tuple[int, int, int, int] | None = None
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    mask_method: str = ""
    mask_quality: TemplateMaskQuality | None = None
    parameters: MatchParameters = field(default_factory=MatchParameters)
    view_mode: ViewMode = ViewMode.ORIGINAL
    busy: bool = False
    busy_message: str = ""
    error: str = ""
    saved_template_id: str | None = None

    @property
    def has_image(self) -> bool:
        return self.image is not None and self.image.size > 0

    @property
    def has_roi(self) -> bool:
        return self.roi_xywh is not None and self.roi_xywh[2] >= 5 and self.roi_xywh[3] >= 5

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and int(np.count_nonzero(self.mask)) >= 25

    @property
    def can_segment(self) -> bool:
        return self.has_image and self.has_roi and not self.busy

    @property
    def can_save(self) -> bool:
        return bool(self.name.strip()) and self.has_image and self.has_roi and self.has_mask and not self.busy

    @property
    def image_size_text(self) -> str:
        if not self.has_image:
            return "— × —"
        height, width = self.image.shape[:2]
        return f"{width} × {height}"

    def evolve(self, **changes) -> "TemplatePageState":
        return replace(self, **changes)

    def completed_steps(self) -> set[TemplateStep]:
        completed: set[TemplateStep] = set()
        if self.has_image:
            completed.add(TemplateStep.SOURCE)
        if self.has_roi:
            completed.add(TemplateStep.ROI)
        if self.has_mask:
            completed.add(TemplateStep.MASK)
        if self.has_mask and self.parameters:
            completed.add(TemplateStep.PARAMETERS)
        if self.saved_template_id:
            completed.add(TemplateStep.SAVE)
        return completed
"""Qt 状态模型：描述模板建立页面的步骤、图像和参数状态。"""
