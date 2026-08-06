import json

import cv2
import numpy as np
import pytest

from industrial_segpose.k230_export import export_k230_bundle
from industrial_segpose.template_matching import MatchParameters, TemplateLibrary, TemplateModel
from k230_runtime.template_library import K230TemplateLibrary, _dirname, _join


def test_k230_path_helpers_do_not_require_os_path():
    assert _dirname("/sdcard/industrial_vision/templates/template_library.json") == "/sdcard/industrial_vision/templates"
    assert _dirname("metadata.json") == ""
    assert _join("/sdcard/templates", "part/metadata.json") == "/sdcard/templates/part/metadata.json"
    assert _join("/sdcard/templates", "/absolute/metadata.json") == "/absolute/metadata.json"


def test_export_compact_k230_template_bundle(tmp_path):
    image = np.zeros((420, 760, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    polygon = np.array([[90, 90], [680, 120], [610, 310], [210, 350], [120, 260]])
    cv2.fillPoly(mask, [polygon], 255)
    image[mask > 0] = (130, 180, 240)
    model = TemplateModel("textile-a", image, source_image=r"D:\private\reference.jpg", mask=mask)
    library = TemplateLibrary(tmp_path / "templates")
    library.add_model(
        model,
        MatchParameters(
            score_threshold=0.66,
            angle_min=-90,
            angle_max=90,
            angle_step=5,
            scale_min=0.8,
            scale_max=1.2,
            scale_step=0.1,
        ),
    )

    manifest_path = export_k230_bundle(library.root, tmp_path / "bundle", max_edge=256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = K230TemplateLibrary(str(manifest_path)).load()
    template = loaded.templates[0]

    assert manifest["format_version"] == 1
    assert len(manifest["templates"]) == 1
    assert max(template["width"], template["height"]) <= 256
    assert template["parameters"]["score_threshold"] == 0.66
    assert template["pick_radius_px"] > 10
    assert template["segmentation"]["lab_thresholds"]
    assert "discriminative_channel" in template["segmentation"]
    assert template["shape_features"]["elongation"] >= 0.0
    assert 0.0 < template["shape_features"]["solidity"] <= 1.0
    assert template["reference_angle_deg"] != 0.0
    assert cv2.imread(template["asset_paths"]["pose"], cv2.IMREAD_GRAYSCALE).shape == (192, 192)
    mask_image = cv2.imread(template["asset_paths"]["mask"], cv2.IMREAD_GRAYSCALE)
    pick_x, pick_y = map(lambda value: int(round(value)), template["pick_point_xy"])
    assert mask_image[pick_y, pick_x] == 255
    exported_json = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "bundle").rglob("*.json")
    )
    assert "D:\\private" not in exported_json
    assert "source_image" not in exported_json


def test_export_requires_explicit_overwrite(tmp_path):
    library = TemplateLibrary(tmp_path / "templates")
    library.save()
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_k230_bundle(library.root, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_k230_library_can_toggle_and_remove_without_deleting_assets(tmp_path):
    image = np.full((100, 180, 3), 180, np.uint8)
    mask = np.zeros((100, 180), np.uint8)
    cv2.rectangle(mask, (20, 20), (160, 80), 255, -1)
    library = TemplateLibrary(tmp_path / "templates")
    entry = library.add_model(TemplateModel("part", image, mask=mask))
    manifest_path = export_k230_bundle(library.root, tmp_path / "bundle")
    board_library = K230TemplateLibrary(str(manifest_path)).load()
    metadata_path = board_library.templates[0]["asset_paths"]["gray"]

    board_library.set_enabled(entry.template_id, False)
    assert board_library.templates[0]["enabled"] is False
    board_library.remove(entry.template_id)

    assert board_library.templates == []
    assert (tmp_path / "bundle" / entry.template_id).is_dir()
    assert metadata_path.endswith("template_gray.png")
