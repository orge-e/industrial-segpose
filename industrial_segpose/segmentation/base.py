"""Segmentation backend protocol and shared helpers."""

from typing import Protocol
import cv2
import numpy as np
from ..types import SegmentationResult


class SegmentationBackend(Protocol):
    name: str
    def segment(self, image: np.ndarray) -> SegmentationResult: ...


def clean_binary(binary: np.ndarray, config: dict) -> np.ndarray:
    if binary.ndim != 2:
        raise ValueError("Segmentation binary image must have one channel")
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)
    cfg = config["morphology"]
    kernel_size = int(cfg["kernel_size"])
    if kernel_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if int(cfg["open_iterations"]):
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=int(cfg["open_iterations"]))
        if int(cfg["close_iterations"]):
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(cfg["close_iterations"]))
    return binary


def connected_instances(binary: np.ndarray, debug: dict[str, np.ndarray] | None = None) -> SegmentationResult:
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    masks = [(labels == label).astype(np.uint8) * 255 for label in range(1, count)]
    return SegmentationResult(labels.astype(np.int32), masks, debug or {"binary": binary})


def as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3:
        return image[:, :, 0]
    raise ValueError("Unsupported image array shape")
