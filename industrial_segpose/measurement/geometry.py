"""目标测量：计算轮廓、外接框和旋转矩形等几何结果。"""

import cv2
import numpy as np


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def touches_border(mask: np.ndarray) -> bool:
    return bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
"""目标测量：计算轮廓、外接框和旋转矩形等几何结果。"""
