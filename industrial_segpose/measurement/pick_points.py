"""Suction-safe pick-point planning inside an irregular binary mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PickPointCandidate:
    x: float
    y: float
    safe_radius_px: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "safe_radius_px": float(self.safe_radius_px),
            "score": float(self.score),
        }


def plan_pick_points(
    mask: np.ndarray,
    maximum_points: int = 3,
    minimum_safe_radius_px: float = 3.0,
    minimum_spacing_px: float | None = None,
) -> list[PickPointCandidate]:
    """Return well-separated local distance-transform maxima.

    Holes and exterior pixels are zero, so candidates are automatically kept
    away from both the outer contour and internal holes. The first item is the
    primary point and remaining items are ordered fallbacks.
    """
    if maximum_points < 1:
        raise ValueError("maximum_points must be at least 1")
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return []
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0:
        return []
    spacing = float(minimum_spacing_px) if minimum_spacing_px is not None else max(6.0, maximum * 0.85)
    kernel_size = max(3, int(round(maximum * 0.55)) | 1)
    local_max = distance >= cv2.dilate(distance, np.ones((kernel_size, kernel_size), np.uint8)) - 1e-6
    ys, xs = np.where(local_max & (distance >= float(minimum_safe_radius_px)))
    ranked = sorted(
        ((float(distance[y, x]), int(x), int(y)) for x, y in zip(xs, ys)),
        key=lambda item: (-item[0], item[2], item[1]),
    )
    selected: list[PickPointCandidate] = []
    for radius, x, y in ranked:
        if any((x - item.x) ** 2 + (y - item.y) ** 2 < spacing * spacing for item in selected):
            continue
        selected.append(PickPointCandidate(float(x), float(y), radius, radius / maximum))
        if len(selected) >= maximum_points:
            break
    return selected
