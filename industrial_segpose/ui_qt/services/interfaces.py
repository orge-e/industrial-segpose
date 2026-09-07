"""Presentation-facing service contracts.

Only template authoring has a concrete adapter in this migration stage. The
other contracts make future pages depend on stable
boundaries instead of importing cameras or algorithms directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ...template_matching.library import TemplateLibraryEntry
from ...template_matching.matcher import MatchParameters
from ...template_matching.model import TemplateModel


class TemplateRepository(Protocol):
    def add(self, model: TemplateModel, parameters: MatchParameters) -> TemplateLibraryEntry: ...
    def count(self) -> int: ...


class CameraService(Protocol):
    def open(self, source: object) -> None: ...
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class DetectionService(Protocol):
    def detect(self, image: np.ndarray) -> object: ...


class CalibrationService(Protocol):
    def load(self, path: str | Path) -> object: ...
    def image_to_world(self, point_xy: tuple[float, float]) -> tuple[float, float]: ...
