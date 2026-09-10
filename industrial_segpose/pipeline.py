"""End-to-end segmentation and measurement pipeline."""

from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .filtering import evaluate_instance
from .measurement.center import calculate_center
from .measurement.geometry import largest_contour
from .measurement.orientation import calculate_orientation
from .preprocessing.preprocessing import preprocess_with_transform
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
        work, debug, transform = preprocess_with_transform(image, self.config)
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
            center_x_cv, center_y_cv = calculate_center(contour, measurement["center_method"])
            center_x, center_y = transform.point_to_output(center_x_cv, center_y_cv)
            angle, confidence = calculate_orientation(contour, measurement["angle_method"])
            rect = cv2.minAreaRect(contour)
            width = float(rect[1][0]) / transform.scale_x
            height = float(rect[1][1]) / transform.scale_y
            long_side, short_side = max(width, height), min(width, height)
            box = transform.processed_to_output(cv2.boxPoints(rect)).astype(float).tolist()
            output_contour = transform.contour_to_output(contour).reshape(-1, 2)
            area_scale = transform.scale_x * transform.scale_y
            length_scale = (transform.scale_x + transform.scale_y) / 2.0
            objects.append(ObjectResult(
                0, center_x, center_y, angle, measurement["angle_method"],
                confidence >= float(measurement["min_orientation_confidence"]), confidence,
                geometry["area"] / area_scale, geometry["perimeter"] / length_scale,
                transform.bbox_to_output(geometry["bbox"]), box, long_side, short_side,
                long_side / max(short_side, 1e-9), geometry["touches_border"],
                contour_points=output_contour.astype(float).tolist(),
                axis_angle_deg=angle,
                directed_angle_deg=None,
                direction_confidence=None,
                _mask=transform.restore_mask(mask),
            ))
        objects.sort(key=lambda item: (item.center_y, item.center_x))
        for object_id, obj in enumerate(objects, start=1): obj.object_id = object_id
        elapsed = (perf_counter() - started) * 1000.0
        return ImageResult(str(image_name), image.shape[1], image.shape[0], len(objects), objects, elapsed, self.backend_name, rejected_instances=rejected)
