"""分割后端：使用自适应阈值应对局部光照变化。"""

import cv2
import numpy as np
from .base import as_gray, clean_binary, connected_instances


class AdaptiveThresholdBackend:
    name = "adaptive_threshold"
    def __init__(self, config: dict): self.config = config

    def segment(self, image: np.ndarray):
        gray = as_gray(image)
        cfg = self.config["adaptive_threshold"]
        mode = cv2.THRESH_BINARY_INV if self.config["segmentation"]["invert"] else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, int(cfg["block_size"]), float(cfg["c"]))
        binary = clean_binary(binary, self.config)
        return connected_instances(binary, {"gray": gray, "binary": binary})
"""分割后端：使用自适应阈值应对局部光照变化。"""
