"""Stable undirected-axis orientation calculations."""

import cv2
import numpy as np


def normalize_axis_angle(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def min_area_rect_orientation(contour: np.ndarray) -> tuple[float, float]:
    (_, _), (width, height), angle = cv2.minAreaRect(contour)
    long_angle = angle if width >= height else angle + 90.0
    aspect = max(width, height) / max(min(width, height), 1e-9)
    confidence = float(np.clip(1.0 - 1.0 / aspect, 0.0, 1.0))
    return normalize_axis_angle(long_angle), confidence


def pca_orientation(contour: np.ndarray) -> tuple[float, float]:
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 2:
        return min_area_rect_orientation(contour)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=np.empty((0)))
    vector = eigenvectors[0]
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    values = eigenvalues.reshape(-1)
    confidence = float((values[0] - values[1]) / max(values[0] + values[1], 1e-9)) if len(values) > 1 else 0.0
    return normalize_axis_angle(float(angle)), float(np.clip(confidence, 0.0, 1.0))


def calculate_orientation(contour: np.ndarray, method: str) -> tuple[float, float]:
    return pca_orientation(contour) if method == "pca" else min_area_rect_orientation(contour)
