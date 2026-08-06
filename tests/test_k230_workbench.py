from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_segpose.k230_workbench import (
    check_template_library,
    resolve_capture_root,
    scan_k230_captures,
)


def _write_image(path: Path, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), 128, np.uint8))


def test_scan_k230_captures_accepts_sd_root_and_natural_order(tmp_path):
    captures = tmp_path / "industrial_vision" / "captures"
    _write_image(captures / "session_0002" / "IMG_000010.jpg", 80, 60)
    _write_image(captures / "session_0002" / "IMG_000002.jpg", 64, 48)
    (captures / "session_0002" / "notes.txt").write_text("ignore", encoding="utf-8")

    result = scan_k230_captures(tmp_path)

    assert resolve_capture_root(tmp_path) == captures
    assert [item.path.name for item in result] == ["IMG_000002.jpg", "IMG_000010.jpg"]
    assert result[0].session == "session_0002"
    assert result[0].dimensions == "64 x 48"


def test_scan_k230_captures_accepts_single_copied_session(tmp_path):
    session = tmp_path / "session_0012"
    _write_image(session / "IMG_000001.jpg")

    result = scan_k230_captures(session)

    assert len(result) == 1
    assert result[0].session == "未分组"


def test_template_preflight_rejects_empty_library(tmp_path):
    with pytest.raises(ValueError, match="模板库为空"):
        check_template_library(tmp_path / "templates")
