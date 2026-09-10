import json

import cv2
import numpy as np
import pytest

from industrial_segpose.calibration import PlanarCalibration
from industrial_segpose.calibration.tool import run_calibration_job
from industrial_segpose.measurement.pick_points import plan_pick_points
from industrial_segpose.production.models import Point2D, TargetRecord
from shared_protocol import (
    FLAG_PICK_AREA_INSUFFICIENT,
    FLAG_TOUCHES_BORDER,
    auto_pick_allowed,
    flag_names,
)


def test_pick_point_planner_avoids_outer_edge_and_internal_hole():
    mask = np.zeros((180, 300), np.uint8)
    cv2.rectangle(mask, (15, 20), (284, 164), 255, -1)
    cv2.circle(mask, (150, 92), 35, 0, -1)

    points = plan_pick_points(mask, maximum_points=3, minimum_safe_radius_px=8)

    assert 2 <= len(points) <= 3
    assert all(mask[int(point.y), int(point.x)] == 255 for point in points)
    assert all(point.safe_radius_px >= 8 for point in points)
    assert points[0].safe_radius_px >= points[-1].safe_radius_px
    assert all((point.x - 150) ** 2 + (point.y - 92) ** 2 > 35**2 for point in points)


def test_quality_flags_control_auto_pick_permission():
    flags = FLAG_TOUCHES_BORDER | FLAG_PICK_AREA_INSUFFICIENT

    assert flag_names(flags) == ["touches_border", "pick_area_insufficient"]
    assert auto_pick_allowed(flags) is False
    target = TargetRecord(
        1, "shoe-part", "template-a", "鞋面A", 3, 2,
        Point2D(120.0, 80.0), 15.0, 0.91, (50, 40, 140, 80), flags=flags,
    )
    payload = target.to_dict()
    assert payload["auto_pick_allowed"] is False
    assert payload["flag_names"] == ["touches_border", "pick_area_insufficient"]


def test_planar_calibration_converts_local_and_global_coordinates(tmp_path):
    pixels = np.array([[0, 0], [200, 0], [200, 100], [0, 100], [100, 50]], np.float64)
    local = np.column_stack((pixels[:, 0] * 0.5 + 10.0, pixels[:, 1] * 0.4 - 5.0))
    calibration = PlanarCalibration.fit(
        "workplane-a", pixels, local, tool_offset_mm=(3.0, -2.0), operator="tester"
    )

    assert calibration.report is not None
    assert calibration.report.max_error_mm < 1e-6
    assert np.allclose(calibration.pixel_to_local([[40, 25]])[0], [30, 5], atol=1e-6)
    assert np.allclose(
        calibration.pixel_to_global([[40, 25]], axis_snapshot_mm=(100, 200))[0],
        [133, 203], atol=1e-6,
    )

    path = calibration.save(tmp_path / "calibration.json")
    restored = PlanarCalibration.load(path)
    assert np.allclose(restored.homography, calibration.homography)
    assert restored.report.max_error_mm < 1e-6

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tool_offset_mm"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        PlanarCalibration.load(path)


def test_calibration_job_exports_machine_file_and_reports(tmp_path):
    points = {
        "calibration_id": "grid-01",
        "pixel_points": [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        "local_points_mm": [[10, 20], [60, 20], [60, 70], [10, 70], [35, 45]],
        "tool_offset_mm": [2, -3],
    }
    point_file = tmp_path / "points.json"
    point_file.write_text(json.dumps(points), encoding="utf-8")

    outputs = run_calibration_job(point_file, tmp_path / "output")

    assert all(path.is_file() for path in outputs.values())
    assert "P95误差" in outputs["report"].read_text(encoding="utf-8")
    assert "error_mm" in outputs["csv"].read_text(encoding="utf-8-sig")
    assert PlanarCalibration.load(outputs["calibration"]).tool_offset_mm == (2.0, -3.0)
