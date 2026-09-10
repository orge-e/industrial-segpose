"""Qt 服务：封装模板资产和模板库记录的持久化访问。"""

from __future__ import annotations

from pathlib import Path

from ...template_matching.library import TemplateLibrary, TemplateLibraryEntry
from ...template_matching.matcher import MatchParameters
from ...template_matching.model import TemplateModel


class LibraryTemplateRepository:
    """Adapter for the existing JSON/PNG persistent template library."""

    def __init__(self, root: str | Path):
        self.library = TemplateLibrary(root).load()

    def add(self, model: TemplateModel, parameters: MatchParameters) -> TemplateLibraryEntry:
        return self.library.add_model(model, parameters)

    def count(self) -> int:
        return len(self.library.entries)

    def reload(self) -> None:
        self.library.load()
"""Qt 服务：封装模板资产和模板库记录的持久化访问。"""
