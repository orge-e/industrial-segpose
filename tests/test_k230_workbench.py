from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_segpose.k230_workbench import (
    check_template_library,
    resolve_capture_root,
    scan_k230_captures,
)
from industrial_segpose.template_ui import K230TemplateWorkbench


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


def test_template_preflight_counts_enabled_and_disabled_valid_templates(tmp_path):
    from industrial_segpose.template_matching import TemplateLibrary, TemplateModel

    image = np.full((60, 100, 3), 160, np.uint8)
    mask = np.full((60, 100), 255, np.uint8)
    library = TemplateLibrary(tmp_path / "templates")
    library.add_model(TemplateModel("enabled", image, mask=mask))
    disabled = library.add_model(TemplateModel("disabled", image, mask=mask))
    library.set_enabled(disabled.template_id, False)

    check = check_template_library(library.root)

    assert check.exported_templates == 2
    assert check.enabled_templates == 1
    assert check.disabled_templates == 1


def test_use_as_reference_hides_transient_workbench_instead_of_iconifying(tmp_path):
    image_path = tmp_path / "IMG_000001.jpg"

    class Notebook:
        selected = None

        def select(self, tab):
            self.selected = tab

    class Parent:
        def __init__(self):
            self.notebook = Notebook()
            self.template_tab = object()
            self.loaded = None
            self.lifted = False
            self.focused = False

        def load_reference_path(self, path, auto_extract=False):
            self.loaded = (path, auto_extract)

        def lift(self):
            self.lifted = True

        def focus_force(self):
            self.focused = True

    class Workbench:
        def __init__(self):
            self.parent = Parent()
            self.hidden = False
            self.destroyed = False

        def _selected_captures(self):
            return [type("Capture", (), {"path": image_path})()]

        def withdraw(self):
            self.hidden = True

        def destroy(self):
            self.destroyed = True

    workbench = Workbench()

    K230TemplateWorkbench._use_as_reference(workbench)

    assert workbench.hidden is True
    assert workbench.destroyed is True
    assert workbench.parent.loaded == (image_path, True)
    assert workbench.parent.notebook.selected is workbench.parent.template_tab
    assert workbench.parent.lifted is True
    assert workbench.parent.focused is True
