"""目标测量：根据实例 Mask 计算目标中心坐标。"""

import cv2
import numpy as np


def calculate_center(contour: np.ndarray, method: str = "moments") -> tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    if method == "min_area_rect":
        return float(rect[0][0]), float(rect[0][1])
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    return float(rect[0][0]), float(rect[0][1])
"""目标测量：根据实例 Mask 计算目标中心坐标。"""
