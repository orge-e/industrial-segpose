from industrial_segpose.io.image_reader import read_image
from industrial_segpose.template_demo import prepare_template_demo
from industrial_segpose.template_matching import MultiTemplateMatcher, TemplateLibrary
from industrial_segpose.template_ui import _application_data_root


def test_template_demo_prepares_assets_and_detects_expected_counts(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    demo = prepare_template_demo(library, tmp_path / "demo")
    assert demo.detection_path.is_file()
    assert len(library.entries) == 2
    result = MultiTemplateMatcher.from_library(library).match(read_image(demo.detection_path))
    assert result.object_count == 5
    assert result.ambiguous_count == 0
    assert result.counts_by_template == demo.expected_counts


def test_template_demo_is_idempotent(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    first = prepare_template_demo(library, tmp_path / "demo")
    second = prepare_template_demo(library, tmp_path / "demo")
    assert first.template_ids == second.template_ids
    assert len(library.entries) == 2


def test_ui_data_root_does_not_depend_on_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = _application_data_root()
    assert (root / "pyproject.toml").is_file()
    assert root.name == "industrial-segpose"
