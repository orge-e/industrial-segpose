"""Calibration point-file workflow and durable report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .homography import PlanarCalibration


def run_calibration_job(point_file: str | Path, output_directory: str | Path) -> dict[str, Path]:
    source = Path(point_file)
    payload = json.loads(source.read_text(encoding="utf-8"))
    pixels = np.asarray(payload.get("pixel_points", []), dtype=np.float64).reshape(-1, 2)
    local = np.asarray(payload.get("local_points_mm", []), dtype=np.float64).reshape(-1, 2)
    calibration = PlanarCalibration.fit(
        str(payload.get("calibration_id", source.stem)),
        pixels,
        local,
        tool_offset_mm=tuple(payload.get("tool_offset_mm", (0.0, 0.0))),
        material_height_mm=float(payload.get("material_height_mm", 0.0)),
        operator=str(payload.get("operator", "")),
        use_ransac=bool(payload.get("use_ransac", False)),
        ransac_threshold_mm=float(payload.get("ransac_threshold_mm", 2.0)),
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration.save(output / "calibration.json")
    predicted = calibration.pixel_to_local(pixels)
    errors = np.linalg.norm(predicted - local, axis=1)
    csv_path = output / "calibration_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "pixel_x", "pixel_y", "expected_x_mm", "expected_y_mm", "predicted_x_mm", "predicted_y_mm", "error_mm"))
        for index, (pixel, expected, observed, error) in enumerate(zip(pixels, local, predicted, errors), start=1):
            writer.writerow((index, *pixel, *expected, *observed, float(error)))
    report = calibration.report
    markdown_path = output / "calibration_report.md"
    markdown_path.write_text(
        "\n".join((
            "# 平面视觉标定报告",
            "",
            f"- 标定配方：`{calibration.calibration_id}`",
            f"- 标定点数：{report.point_count}",
            f"- 平均误差：{report.mean_error_mm:.4f} mm",
            f"- P95误差：{report.p95_error_mm:.4f} mm",
            f"- 最大误差：{report.max_error_mm:.4f} mm",
            f"- 吸头偏置：X={calibration.tool_offset_mm[0]:.3f} mm，Y={calibration.tool_offset_mm[1]:.3f} mm",
            f"- 料面高度：{calibration.material_height_mm:.3f} mm",
            "",
            "建议正式验收使用不少于25个覆盖全视野的点，并分别检查平均、P95及最大误差。",
        )),
        encoding="utf-8",
    )
    return {"calibration": calibration_path, "csv": csv_path, "report": markdown_path}
