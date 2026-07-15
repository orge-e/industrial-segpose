import cv2
import numpy as np
from .base import clean_binary, connected_instances


class ColorRangeBackend:
    name = "color_range"
    def __init__(self, config: dict): self.config = config

    def segment(self, image: np.ndarray):
        cfg = self.config["color_range"]
        space = cfg.get("space", "hsv").lower()
        if image.ndim == 2:
            converted = image
        elif space == "hsv":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif space == "lab":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            converted = image
        lower, upper = np.array(cfg["lower"], np.uint8), np.array(cfg["upper"], np.uint8)
        binary = cv2.inRange(converted, lower, upper)
        if self.config["segmentation"]["invert"]:
            binary = cv2.bitwise_not(binary)
        binary = clean_binary(binary, self.config)
        return connected_instances(binary, {"color_space": converted, "binary": binary})
