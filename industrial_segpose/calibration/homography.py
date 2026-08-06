"""Planar homography, tool-offset conversion and validation reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationErrorReport:
    point_count: int
    mean_error_mm: float
    p95_error_mm: float
    max_error_mm: float
    errors_mm: tuple[float, ...]


@dataclass
class PlanarCalibration:
    calibration_id: str
    homography: np.ndarray
    tool_offset_mm: tuple[float, float] = (0.0, 0.0)
    material_height_mm: float = 0.0
    created_at: str = ""
    operator: str = ""
    report: CalibrationErrorReport | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        matrix = np.asarray(self.homography, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("homography must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography is singular")
        self.homography = matrix
        self.tool_offset_mm = tuple(map(float, self.tool_offset_mm))
        if not self.calibration_id.strip():
            raise ValueError("calibration_id cannot be empty")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def fit(
        cls,
        calibration_id: str,
        pixel_points: list[tuple[float, float]] | np.ndarray,
        local_points_mm: list[tuple[float, float]] | np.ndarray,
        tool_offset_mm: tuple[float, float] = (0.0, 0.0),
        material_height_mm: float = 0.0,
        operator: str = "",
        use_ransac: bool = False,
        ransac_threshold_mm: float = 2.0,
    ) -> "PlanarCalibration":
        source = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 2)
        destination = np.asarray(local_points_mm, dtype=np.float64).reshape(-1, 2)
        if len(source) != len(destination) or len(source) < 4:
            raise ValueError("at least four paired calibration points are required")
        method = cv2.RANSAC if use_ransac else 0
        matrix, _ = cv2.findHomography(source, destination, method, float(ransac_threshold_mm))
        if matrix is None:
            raise ValueError("homography estimation failed")
        calibration = cls(
            calibration_id,
            matrix,
            tool_offset_mm=tool_offset_mm,
            material_height_mm=float(material_height_mm),
            operator=operator,
        )
        calibration.report = calibration.evaluate(source, destination)
        return calibration

    def pixel_to_local(self, points: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, self.homography).reshape(-1, 2)

    def pixel_to_global(
        self,
        points: list[tuple[float, float]] | np.ndarray,
        axis_snapshot_mm: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        result = self.pixel_to_local(points)
        result[:, 0] += float(axis_snapshot_mm[0]) + self.tool_offset_mm[0]
        result[:, 1] += float(axis_snapshot_mm[1]) + self.tool_offset_mm[1]
        return result

    def evaluate(
        self,
        pixel_points: list[tuple[float, float]] | np.ndarray,
        expected_local_mm: list[tuple[float, float]] | np.ndarray,
    ) -> CalibrationErrorReport:
        expected = np.asarray(expected_local_mm, dtype=np.float64).reshape(-1, 2)
        predicted = self.pixel_to_local(pixel_points)
        if len(predicted) != len(expected) or not len(expected):
            raise ValueError("evaluation points must be non-empty paired coordinates")
        errors = np.linalg.norm(predicted - expected, axis=1)
        return CalibrationErrorReport(
            len(errors),
            float(np.mean(errors)),
            float(np.percentile(errors, 95)),
            float(np.max(errors)),
            tuple(float(value) for value in errors),
        )

    def _payload(self) -> dict:
        return {
            "format_version": self.format_version,
            "calibration_id": self.calibration_id,
            "homography": self.homography.tolist(),
            "tool_offset_mm": list(self.tool_offset_mm),
            "material_height_mm": float(self.material_height_mm),
            "created_at": self.created_at,
            "operator": self.operator,
            "report": asdict(self.report) if self.report is not None else None,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PlanarCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = str(payload.pop("sha256", ""))
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if not checksum or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != checksum:
            raise ValueError("calibration checksum mismatch")
        report_data = payload.get("report")
        report = None
        if report_data:
            report = CalibrationErrorReport(
                int(report_data["point_count"]),
                float(report_data["mean_error_mm"]),
                float(report_data["p95_error_mm"]),
                float(report_data["max_error_mm"]),
                tuple(map(float, report_data["errors_mm"])),
            )
        return cls(
            str(payload["calibration_id"]),
            np.asarray(payload["homography"], dtype=np.float64),
            tuple(payload.get("tool_offset_mm", (0.0, 0.0))),
            float(payload.get("material_height_mm", 0.0)),
            str(payload.get("created_at", "")),
            str(payload.get("operator", "")),
            report,
            int(payload.get("format_version", 1)),
        )
