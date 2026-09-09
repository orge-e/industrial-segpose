import json

import pytest

from k230_runtime.calibration import K230PlanarCalibration


def test_k230_calibration_loads_desktop_file_and_applies_axis_offset(tmp_path):
    payload = {
        "calibration_id": "plane-a",
        "homography": [[0.5, 0.0, 10.0], [0.0, 0.4, -5.0], [0.0, 0.0, 1.0]],
        "tool_offset_mm": [3.0, -2.0],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    calibration = K230PlanarCalibration.load(path)

    assert calibration.pixel_to_global((40, 25), (100, 200)) == pytest.approx([133, 203])
    assert calibration.calibration_id == "plane-a"
