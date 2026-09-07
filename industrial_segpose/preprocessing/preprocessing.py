"""Configurable image preprocessing."""

import cv2
import numpy as np
from ..measurement.coordinates import CoordinateTransform


def preprocess(image: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    work, debug, _ = preprocess_with_transform(image, config)
    return work, debug


def preprocess_with_transform(
    image: np.ndarray, config: dict
) -> tuple[np.ndarray, dict[str, np.ndarray], CoordinateTransform]:
    cfg = config["preprocessing"]
    max_dim = int(config["input"]["resize_max_dimension"] or 0)
    original_height, original_width = image.shape[:2]
    work = image.copy()
    if max_dim and max(work.shape[:2]) > max_dim:
        scale = max_dim / max(work.shape[:2])
        target_width = max(1, int(round(original_width * scale)))
        target_height = max(1, int(round(original_height * scale)))
        work = cv2.resize(work, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_height, resized_width = work.shape[:2]
    roi = config["input"].get("roi")
    if roi:
        x, y, width, height = map(int, roi)
        if min(x, y, width, height) < 0 or width == 0 or height == 0 or x + width > work.shape[1] or y + height > work.shape[0]:
            raise ValueError("input.roi is outside image bounds")
        work = work[y:y + height, x:x + width]
        roi_xywh = (x, y, width, height)
    else:
        roi_xywh = (0, 0, resized_width, resized_height)
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
    transform = CoordinateTransform(
        original_width,
        original_height,
        resized_width,
        resized_height,
        roi_xywh,
    )
    return work, {"preprocessed": processed}, transform
