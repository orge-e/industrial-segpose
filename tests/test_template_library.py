from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_segpose.template_matching import MatchParameters, TemplateLibrary, TemplateModel


def model(name, shape="rect"):
    image = np.zeros((50, 70, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    if shape == "rect":
        cv2.rectangle(mask, (8, 12), (62, 38), 255, -1)
    else:
        points = np.array([[35, 5], [64, 42], [8, 42]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    image[mask > 0] = 220
    return TemplateModel(name, image, mask=mask)


def test_library_roundtrip_enable_parameters_and_remove(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    first = library.add_model(model("工件A"), MatchParameters(score_threshold=0.8))
    second = library.add_model(model("工件B", "triangle"), MatchParameters(angle_step=5))
    assert library.manifest_path.is_file()
    assert (library.root / first.template_file).is_file()

    library.set_enabled(second.template_id, False)
    library.update_parameters(first.template_id, replace(first.parameters, score_threshold=0.85))
    reloaded = TemplateLibrary(library.root).load()
    assert len(reloaded.entries) == 2
    assert reloaded.get(first.template_id).parameters.score_threshold == pytest.approx(0.85)
    assert reloaded.get(second.template_id).enabled is False
    assert all(item.valid for item in reloaded.load_entries())

    asset = reloaded.root / second.template_file
    reloaded.remove(second.template_id)
    assert asset.is_file()
    assert len(TemplateLibrary(reloaded.root).load().entries) == 1


def test_library_import_copies_assets_and_rejects_duplicate_name(tmp_path):
    external = tmp_path / "external" / "part.json"
    model("同名工件").save(external)
    library = TemplateLibrary(tmp_path / "library")
    entry = library.import_template(external)
    external.unlink()
    loaded = library.load_entries()[0]
    assert loaded.valid
    assert loaded.entry.template_id == entry.template_id
    with pytest.raises(ValueError, match="already exists"):
        library.add_model(model("同名工件"))


def test_library_reports_missing_template_without_blocking_others(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    good = library.add_model(model("正常模板"))
    missing = library.add_model(model("丢失模板", "triangle"))
    (library.root / missing.template_file).unlink()
    loaded = library.load_entries()
    assert next(item for item in loaded if item.entry.template_id == good.template_id).valid
    bad = next(item for item in loaded if item.entry.template_id == missing.template_id)
    assert not bad.valid
    assert "does not exist" in bad.error
