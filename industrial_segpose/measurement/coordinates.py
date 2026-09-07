"""测量模块：转换 OpenCV 工作坐标与项目输出坐标。

OpenCV algorithms keep their native top-left origin and downward-positive Y
axis.  Business-facing vector coordinates use the original input image with a
bottom-left origin, right-positive X and upward-positive Y.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CoordinateTransform:
    """A reversible resize/crop transform for one original image."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    roi_xywh: tuple[int, int, int, int]

    @classmethod
    def identity(cls, width: int, height: int) -> "CoordinateTransform":
        return cls(width, height, width, height, (0, 0, width, height))

    @property
    def scale_x(self) -> float:
        return self.resized_width / float(self.original_width)

    @property
    def scale_y(self) -> float:
        return self.resized_height / float(self.original_height)

    @property
    def processed_size(self) -> tuple[int, int]:
        return self.roi_xywh[2], self.roi_xywh[3]

    def processed_to_original_cv(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        values[:, 0] = (values[:, 0] + self.roi_xywh[0]) / self.scale_x
        values[:, 1] = (values[:, 1] + self.roi_xywh[1]) / self.scale_y
        return values

    def original_cv_to_processed(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        values[:, 0] = values[:, 0] * self.scale_x - self.roi_xywh[0]
        values[:, 1] = values[:, 1] * self.scale_y - self.roi_xywh[1]
        return values

    def original_cv_to_output(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        values[:, 1] = (self.original_height - 1) - values[:, 1]
        return values

    def output_to_original_cv(self, points: np.ndarray) -> np.ndarray:
        return self.original_cv_to_output(points)

    def processed_to_output(self, points: np.ndarray) -> np.ndarray:
        return self.original_cv_to_output(self.processed_to_original_cv(points))

    def output_to_processed(self, points: np.ndarray) -> np.ndarray:
        return self.original_cv_to_processed(self.output_to_original_cv(points))

    def point_to_output(self, x: float, y: float) -> tuple[float, float]:
        point = self.processed_to_output(np.asarray([[x, y]], np.float64))[0]
        return float(point[0]), float(point[1])

    def contour_to_output(self, contour: np.ndarray) -> np.ndarray:
        points = self.processed_to_output(np.asarray(contour).reshape(-1, 2))
        return points.reshape(-1, 1, 2)

    def bbox_to_output(self, bbox_xywh: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x, y, width, height = bbox_xywh
        corners = self.processed_to_original_cv(
            np.asarray([[x, y], [x + width, y + height]], np.float64)
        )
        left, top = corners[0]
        right, bottom = corners[1]
        return (
            float(left),
            float(self.original_height - bottom),
            float(right - left),
            float(bottom - top),
        )

    def roi_to_output(self) -> tuple[float, float, float, float]:
        return self.bbox_to_output((0, 0, self.roi_xywh[2], self.roi_xywh[3]))

    def restore_mask(self, processed_mask: np.ndarray) -> np.ndarray:
        """Restore a processed mask to the original raster (OpenCV row order)."""
        width, height = self.processed_size
        if processed_mask.shape[:2] != (height, width):
            raise ValueError("Processed mask shape does not match coordinate transform")
        resized_mask = np.zeros((self.resized_height, self.resized_width), np.uint8)
        x, y, roi_width, roi_height = self.roi_xywh
        resized_mask[y : y + roi_height, x : x + roi_width] = processed_mask
        if (self.resized_width, self.resized_height) == (self.original_width, self.original_height):
            return resized_mask
        return cv2.resize(
            resized_mask,
            (self.original_width, self.original_height),
            interpolation=cv2.INTER_NEAREST,
        )


def output_axis_angle_from_cv(angle_deg: float) -> float:
    """Convert an undirected OpenCV-raster angle to output coordinates."""
    return float((-angle_deg) % 180.0)


def output_directed_angle_from_cv(angle_deg: float) -> float:
    """Convert a directed OpenCV-raster angle to output coordinates."""
    return float((-angle_deg) % 360.0)
