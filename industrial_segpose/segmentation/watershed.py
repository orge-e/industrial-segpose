"""Marker-based watershed for lightly touching foreground objects."""

import cv2
import numpy as np
from .base import as_gray, clean_binary
from ..types import SegmentationResult


class WatershedBackend:
    name = "watershed"
    def __init__(self, config: dict): self.config = config

    def segment(self, image: np.ndarray) -> SegmentationResult:
        gray = as_gray(image)
        threshold_cfg = self.config["threshold"]
        flag = cv2.THRESH_BINARY_INV if self.config["segmentation"]["invert"] else cv2.THRESH_BINARY
        if threshold_cfg["method"] == "otsu": flag |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(gray, float(threshold_cfg["value"]), 255, flag)
        binary = clean_binary(binary, self.config)
        cfg = self.config["watershed"]
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peak = float(distance.max())
        if peak <= 0:
            empty = np.zeros(binary.shape, np.int32)
            return SegmentationResult(empty, [], {"gray": gray, "binary": binary, "distance": distance})
        _, foreground = cv2.threshold(distance, float(cfg["distance_threshold_ratio"]) * peak, 255, cv2.THRESH_BINARY)
        foreground = foreground.astype(np.uint8)
        count, markers = cv2.connectedComponents(foreground)
        min_marker = int(cfg["min_marker_area"])
        for label in range(1, count):
            if np.count_nonzero(markers == label) < min_marker:
                markers[markers == label] = 0
        _, markers = cv2.connectedComponents((markers > 0).astype(np.uint8))
        kernel = np.ones((3, 3), np.uint8)
        background = cv2.dilate(binary, kernel, iterations=int(cfg["background_dilate_iterations"]))
        unknown = cv2.subtract(background, foreground)
        markers = markers + 1
        markers[unknown > 0] = 0
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        watershed_labels = cv2.watershed(color, markers.astype(np.int32))
        output = np.zeros(binary.shape, np.int32)
        masks: list[np.ndarray] = []
        next_label = 1
        for label in sorted(v for v in np.unique(watershed_labels) if v > 1):
            mask = ((watershed_labels == label) & (binary > 0)).astype(np.uint8) * 255
            if np.count_nonzero(mask) == 0: continue
            output[mask > 0] = next_label
            masks.append(mask)
            next_label += 1
        distance_vis = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return SegmentationResult(output, masks, {"gray": gray, "binary": binary, "distance": distance_vis, "markers": markers.astype(np.uint16)})
