import cv2
import numpy as np
from .base import as_gray, clean_binary, connected_instances


class ThresholdBackend:
    name = "threshold"
    def __init__(self, config: dict): self.config = config

    def segment(self, image: np.ndarray):
        gray = as_gray(image)
        cfg = self.config["threshold"]
        flag = cv2.THRESH_BINARY_INV if self.config["segmentation"]["invert"] else cv2.THRESH_BINARY
        if cfg["method"] == "otsu":
            flag |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(gray, float(cfg["value"]), 255, flag)
        binary = clean_binary(binary, self.config)
        return connected_instances(binary, {"gray": gray, "binary": binary})
