"""Configurable image preprocessing."""

import cv2
import numpy as np


def preprocess(image: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cfg = config["preprocessing"]
    max_dim = int(config["input"]["resize_max_dimension"] or 0)
    work = image.copy()
    if max_dim and max(work.shape[:2]) > max_dim:
        scale = max_dim / max(work.shape[:2])
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    roi = config["input"].get("roi")
    if roi:
        x, y, width, height = map(int, roi)
        if min(x, y, width, height) < 0 or width == 0 or height == 0 or x + width > work.shape[1] or y + height > work.shape[0]:
            raise ValueError("input.roi is outside image bounds")
        work = work[y:y + height, x:x + width]
    space = cfg["color_space"].lower()
    if space == "gray":
        processed = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    elif space == "hsv":
        processed = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    elif space == "lab":
        processed = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    else:
        raise ValueError(f"Unsupported preprocessing color space: {space}")
    if processed.ndim == 2:
        alpha = float(cfg.get("contrast_alpha", 1.0))
        if alpha != 1.0:
            processed = cv2.convertScaleAbs(processed, alpha=alpha)
        if cfg["clahe_enabled"]:
            processed = cv2.createCLAHE(float(cfg["clahe_clip_limit"]), (8, 8)).apply(processed)
    gaussian = int(cfg["gaussian_blur_kernel"])
    median = int(cfg["median_blur_kernel"])
    if gaussian:
        processed = cv2.GaussianBlur(processed, (gaussian, gaussian), 0)
    if median:
        processed = cv2.medianBlur(processed, median)
    return work, {"preprocessed": processed}
