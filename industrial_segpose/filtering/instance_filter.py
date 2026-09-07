"""实例过滤：依据面积、尺寸、形状和边界条件剔除无效目标。"""

import cv2
import numpy as np
from ..measurement.geometry import touches_border


def evaluate_instance(mask: np.ndarray, contour: np.ndarray, config: dict) -> tuple[list[str], dict]:
    cfg = config["filter"]
    area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    perimeter = float(cv2.arcLength(contour, True))
    aspect = width / max(height, 1)
    circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-9)
    border = touches_border(mask)
    reasons: list[str] = []
    if area < float(cfg["min_area_px"]): reasons.append("area_too_small")
    if cfg["max_area_px"] is not None and area > float(cfg["max_area_px"]): reasons.append("area_too_large")
    if width < int(cfg["min_width_px"]): reasons.append("width_too_small")
    if height < int(cfg["min_height_px"]): reasons.append("height_too_small")
    if not float(cfg["min_aspect_ratio"]) <= aspect <= float(cfg["max_aspect_ratio"]): reasons.append("invalid_geometry")
    if circularity < float(cfg.get("min_circularity", 0.0)): reasons.append("invalid_geometry")
    if cfg["reject_border_objects"] and border: reasons.append("touching_border")
    return reasons, {"area": area, "bbox": (x, y, width, height), "perimeter": perimeter, "aspect": aspect, "circularity": circularity, "touches_border": border}
"""实例过滤：依据面积、尺寸、形状和边界条件剔除无效目标。"""
