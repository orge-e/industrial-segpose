"""End-to-end segmentation and measurement pipeline."""

from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .filtering import evaluate_instance
from .measurement.center import calculate_center
from .measurement.geometry import largest_contour
from .measurement.orientation import calculate_orientation
from .preprocessing import preprocess
from .segmentation import create_backend
from .types import ImageResult, ObjectResult, RejectedInstance, SegmentationResult


class SegPosePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.backend_name = config["segmentation"]["backend"]
        self.backend = create_backend(self.backend_name, config)
        self.last_segmentation: SegmentationResult | None = None
        self.last_work_image: np.ndarray | None = None

    def process(self, image: np.ndarray, image_name: str = "image") -> ImageResult:
        started = perf_counter()
        work, debug = preprocess(image, self.config)
        source = debug["preprocessed"]
        segmentation = self.backend.segment(source if self.backend_name != "color_range" else work)
        segmentation.debug_images = {**debug, **segmentation.debug_images}
        self.last_segmentation, self.last_work_image = segmentation, work
        objects: list[ObjectResult] = []
        rejected: list[RejectedInstance] = []
        measurement = self.config["measurement"]
        for source_label, mask in enumerate(segmentation.masks, start=1):
            contour = largest_contour(mask)
            if contour is None or len(contour) < 3:
                rejected.append(RejectedInstance(source_label, ["invalid_contour"], float(np.count_nonzero(mask))))
                continue
            reasons, geometry = evaluate_instance(mask, contour, self.config)
            if reasons:
                rejected.append(RejectedInstance(source_label, reasons, geometry["area"]))
                continue
            center_x, center_y = calculate_center(contour, measurement["center_method"])
            angle, confidence = calculate_orientation(contour, measurement["angle_method"])
            rect = cv2.minAreaRect(contour)
            width, height = map(float, rect[1])
            long_side, short_side = max(width, height), min(width, height)
            box = cv2.boxPoints(rect).astype(float).tolist()
            objects.append(ObjectResult(0, center_x, center_y, angle, measurement["angle_method"], confidence >= float(measurement["min_orientation_confidence"]), confidence, geometry["area"], geometry["perimeter"], geometry["bbox"], box, long_side, short_side, long_side / max(short_side, 1e-9), geometry["touches_border"], _mask=mask))
        objects.sort(key=lambda item: (item.center_y, item.center_x))
        for object_id, obj in enumerate(objects, start=1): obj.object_id = object_id
        elapsed = (perf_counter() - started) * 1000.0
        return ImageResult(str(image_name), work.shape[1], work.shape[0], len(objects), objects, elapsed, self.backend_name, rejected_instances=rejected)
