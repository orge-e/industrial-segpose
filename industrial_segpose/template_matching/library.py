"""Persistent multi-template library and per-template matching settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from uuid import uuid4

from .matcher import MatchParameters
from .model import TemplateModel


_COLORS = [
    (0, 220, 0), (255, 140, 0), (255, 0, 180), (0, 180, 255),
    (180, 80, 255), (255, 220, 0), (80, 220, 180), (220, 120, 80),
]


def _safe_stem(name: str) -> str:
    value = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return value[:48] or "template"


def _parameters_from_dict(payload: dict | None) -> MatchParameters:
    allowed = MatchParameters.__dataclass_fields__
    values = {key: value for key, value in (payload or {}).items() if key in allowed}
    result = MatchParameters(**values)
    result.validate()
    return result


@dataclass
class TemplateLibraryEntry:
    template_id: str
    name: str
    template_file: str
    enabled: bool = True
    color_bgr: tuple[int, int, int] = (0, 220, 0)
    parameters: MatchParameters = field(default_factory=MatchParameters)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template_file": self.template_file,
            "enabled": bool(self.enabled),
            "color_bgr": list(map(int, self.color_bgr)),
            "parameters": asdict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TemplateLibraryEntry":
        color = payload.get("color_bgr", (0, 220, 0))
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            color = (0, 220, 0)
        return cls(
            template_id=str(payload["template_id"]),
            name=str(payload["name"]),
            template_file=str(payload["template_file"]),
            enabled=bool(payload.get("enabled", True)),
            color_bgr=tuple(max(0, min(255, int(value))) for value in color),
            parameters=_parameters_from_dict(payload.get("parameters")),
        )


@dataclass(frozen=True)
class LoadedTemplateEntry:
    entry: TemplateLibraryEntry
    model: TemplateModel | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.model is not None and self.error is None


class TemplateLibrary:
    FORMAT_VERSION = 1

    def __init__(self, root: str | Path = "templates"):
        self.root = Path(root)
        self.manifest_path = self.root / "template_library.json"
        self.entries: list[TemplateLibraryEntry] = []
        self.ambiguity_margin = 0.08
        self.cross_template_iou = 0.25

    def load(self) -> "TemplateLibrary":
        self.entries = []
        if not self.manifest_path.is_file():
            return self
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported template library format version")
        self.ambiguity_margin = float(payload.get("ambiguity_margin", 0.08))
        self.cross_template_iou = float(payload.get("cross_template_iou", 0.25))
        self.entries = [TemplateLibraryEntry.from_dict(item) for item in payload.get("entries", [])]
        self._validate_unique()
        return self

    def save(self) -> Path:
        self._validate_unique()
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "ambiguity_margin": self.ambiguity_margin,
            "cross_template_iou": self.cross_template_iou,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)
        return self.manifest_path

    def _validate_unique(self) -> None:
        names = [entry.name.casefold() for entry in self.entries]
        ids = [entry.template_id for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Template names must be unique")
        if len(ids) != len(set(ids)):
            raise ValueError("Template IDs must be unique")

    def _assert_name_available(self, name: str, excluding_id: str | None = None) -> None:
        if any(entry.template_id != excluding_id and entry.name.casefold() == name.casefold() for entry in self.entries):
            raise ValueError(f"Template name already exists: {name}")

    def add_model(
        self,
        model: TemplateModel,
        parameters: MatchParameters | None = None,
        enabled: bool = True,
    ) -> TemplateLibraryEntry:
        self._assert_name_available(model.name)
        template_id = uuid4().hex
        filename = f"{template_id[:8]}_{_safe_stem(model.name)}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        model.save(self.root / filename)
        entry = TemplateLibraryEntry(
            template_id=template_id,
            name=model.name,
            template_file=filename,
            enabled=enabled,
            color_bgr=_COLORS[len(self.entries) % len(_COLORS)],
            parameters=parameters or MatchParameters(),
        )
        self.entries.append(entry)
        self.save()
        return entry

    def import_template(
        self,
        metadata_path: str | Path,
        parameters: MatchParameters | None = None,
    ) -> TemplateLibraryEntry:
        return self.add_model(TemplateModel.load(metadata_path), parameters)

    def get(self, template_id: str) -> TemplateLibraryEntry:
        for entry in self.entries:
            if entry.template_id == template_id:
                return entry
        raise KeyError(f"Unknown template ID: {template_id}")

    def remove(self, template_id: str) -> TemplateLibraryEntry:
        entry = self.get(template_id)
        self.entries.remove(entry)
        self.save()
        return entry

    def set_enabled(self, template_id: str, enabled: bool) -> None:
        self.get(template_id).enabled = bool(enabled)
        self.save()

    def update_parameters(self, template_id: str, parameters: MatchParameters) -> None:
        parameters.validate()
        self.get(template_id).parameters = parameters
        self.save()

    def rename(self, template_id: str, name: str) -> None:
        """Rename the display class while keeping the stable internal ID."""

        normalized = name.strip()
        if not normalized:
            raise ValueError("Template name cannot be empty")
        if len(normalized) > 48:
            raise ValueError("Template name must not exceed 48 characters")
        entry = self.get(template_id)
        self._assert_name_available(normalized, excluding_id=template_id)
        model_path = self.root / entry.template_file
        model = TemplateModel.load(model_path)
        previous_entry_name = entry.name
        previous_model_name = model.name
        try:
            model.name = normalized
            model.save(model_path)
            entry.name = normalized
            self.save()
        except Exception:
            entry.name = previous_entry_name
            model.name = previous_model_name
            model.save(model_path)
            raise

    def load_entries(self) -> list[LoadedTemplateEntry]:
        loaded: list[LoadedTemplateEntry] = []
        for entry in self.entries:
            path = self.root / entry.template_file
            try:
                model = TemplateModel.load(path)
                if model.name != entry.name:
                    raise ValueError(f"Template name mismatch: manifest={entry.name}, file={model.name}")
                loaded.append(LoadedTemplateEntry(entry, model))
            except Exception as exc:
                loaded.append(LoadedTemplateEntry(entry, None, str(exc)))
        return loaded

