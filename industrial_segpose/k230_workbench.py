"""Desktop workflow helpers for K230 captures, validation, and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .io.image_reader import SUPPORTED_EXTENSIONS, read_image
from .k230_deploy import build_k230_deployment
from .k230_validation import validate_paths
from .template_matching import TemplateLibrary


_SESSION_PATTERN = re.compile(r"^session[_-]?(\d+)$", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class K230CaptureImage:
    """A captured image discovered in a K230 capture directory."""

    path: Path
    session: str
    width: int
    height: int
    size_bytes: int

    @property
    def dimensions(self) -> str:
        return f"{self.width} x {self.height}"


@dataclass(frozen=True)
class DeploymentCheck:
    """Preflight result for a desktop template library."""

    exported_templates: int
    enabled_templates: int
    disabled_templates: int
    invalid_templates: tuple[str, ...]


def _natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NUMBER_PATTERN.split(str(path)))


def resolve_capture_root(source: str | Path) -> Path:
    """Resolve an SD root, app root, capture root, or copied session folder."""

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"采集目录不存在：{root}")
    candidates = (root / "industrial_vision" / "captures", root / "captures", root)
    for candidate in candidates:
        if candidate.is_dir() and any(
            item.is_dir() and _SESSION_PATTERN.match(item.name) for item in candidate.iterdir()
        ):
            return candidate
    for candidate in candidates:
        if candidate.is_dir() and any(
            item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS for item in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(f"未在目录中找到K230采集图像：{root}")


def scan_k230_captures(source: str | Path) -> list[K230CaptureImage]:
    """Scan K230 captures without copying them into the desktop project."""

    capture_root = resolve_capture_root(source)
    paths = sorted(
        (item for item in capture_root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=_natural_key,
    )
    captures: list[K230CaptureImage] = []
    for path in paths:
        try:
            image = read_image(path)
        except (OSError, ValueError):
            continue
        relative = path.relative_to(capture_root)
        session = relative.parts[0] if len(relative.parts) > 1 else "未分组"
        captures.append(
            K230CaptureImage(
                path=path,
                session=session,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                size_bytes=path.stat().st_size,
            )
        )
    if not captures:
        raise FileNotFoundError(f"采集目录内没有可读取的图像：{capture_root}")
    return captures


def check_template_library(template_root: str | Path) -> DeploymentCheck:
    """Fail early when no usable enabled template can be exported."""

    library = TemplateLibrary(template_root).load()
    exported = 0
    enabled = 0
    disabled = 0
    invalid: list[str] = []
    for loaded in library.load_entries():
        if loaded.valid:
            exported += 1
            if loaded.entry.enabled:
                enabled += 1
            else:
                disabled += 1
        else:
            invalid.append(f"{loaded.entry.name}: {loaded.error}")
    if enabled == 0:
        detail = "；".join(invalid) if invalid else "模板库为空或全部已停用"
        raise ValueError(f"无法生成K230部署包：{detail}")
    return DeploymentCheck(exported, enabled, disabled, tuple(invalid))


def build_workbench_deployment(
    project_root: str | Path,
    output_root: str | Path | None = None,
) -> tuple[Path, DeploymentCheck]:
    """Build the complete SD-card package after validating desktop templates."""

    project = Path(project_root).resolve()
    check = check_template_library(project / "templates")
    output = Path(output_root).resolve() if output_root else project / "build" / "k230_sdcard"
    app_root = build_k230_deployment(project, output, overwrite=True)
    return app_root, check


def validate_workbench_images(
    deployment_app_root: str | Path,
    image_paths: Iterable[str | Path],
    output_root: str | Path,
) -> Path:
    """Replay selected captures using the exported K230 bundle."""

    app_root = Path(deployment_app_root)
    manifest = app_root / "templates" / "template_library.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"部署包模板清单不存在：{manifest}")
    return validate_paths(manifest, image_paths, Path(output_root))
