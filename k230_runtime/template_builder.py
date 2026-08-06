"""On-device assisted template creation backend.

The touch wizard can call this service after the operator freezes a frame and
selects an ROI.  Asset creation uses only APIs available in CanMV/OpenMV.
"""

import math

try:
    import ujson as json
except ImportError:
    import json

try:
    import uos as os
except ImportError:
    import os


def suggested_lab_threshold(statistics, sigma=2.5):
    """Create a robust CanMV LAB threshold from ROI statistics."""
    sigma = max(1.0, float(sigma))
    values = []
    for channel, low, high in (("l", 0, 100), ("a", -128, 127), ("b", -128, 127)):
        mean = float(getattr(statistics, channel + "_mean")())
        deviation_method = getattr(statistics, channel + "_stdev", None)
        deviation = float(deviation_method()) if callable(deviation_method) else 10.0
        margin = max(5.0, deviation * sigma)
        values.extend([int(max(low, mean - margin)), int(min(high, mean + margin))])
    return values


def assisted_threshold_candidates(base_threshold):
    """Add signed-chroma candidates so a neutral ROI background cannot dominate."""
    candidates = [("assisted_roi", list(base_threshold))]
    candidates.extend([
        ("color", [base_threshold[0], base_threshold[1], 4, 127, -128, 127]),
        ("color", [base_threshold[0], base_threshold[1], -128, -4, -128, 127]),
        ("color", [base_threshold[0], base_threshold[1], -128, 127, 4, 127]),
        ("color", [base_threshold[0], base_threshold[1], -128, 127, -128, -4]),
    ])
    return candidates


def _ensure_dir(path):
    parts = path.replace("\\", "/").split("/")
    current = "/" if path.startswith("/") else ""
    for part in parts:
        if not part:
            continue
        current = current + part if current in ("", "/") else current + "/" + part
        try:
            os.stat(current)
        except OSError:
            os.mkdir(current)


def _read_json(path, default):
    try:
        with open(path, "r") as stream:
            return json.loads(stream.read())
    except (OSError, ValueError):
        return default


def _write_json_atomic(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(json.dumps(payload))
    try:
        os.remove(path)
    except OSError:
        pass
    os.rename(temporary, path)


def _axis_length(blob, method, fallback):
    value = getattr(blob, method, None)
    if not callable(value):
        return float(fallback)
    try:
        x1, y1, x2, y2 = value()
        return max(1.0, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    except Exception:
        return float(fallback)


class OnDeviceTemplateBuilder:
    def __init__(self, template_root, pose_size=192):
        self.template_root = template_root.rstrip("/")
        self.pose_size = int(pose_size)

    def analyze_roi(self, frame, roi):
        statistics = frame.get_statistics(roi=roi)
        base_threshold = suggested_lab_threshold(statistics)
        roi_area = max(1, int(roi[2]) * int(roi[3]))
        best = None
        for mode, threshold in assisted_threshold_candidates(base_threshold):
            blobs = frame.find_blobs([tuple(threshold)], roi=roi, pixels_threshold=80, area_threshold=120, merge=True)
            for blob in blobs or []:
                fraction = float(blob.pixels()) / roi_area
                if fraction < 0.015:
                    continue
                # A result covering almost the complete ROI is normally the
                # table/conveyor, not the selected irregular workpiece.
                flood_penalty = 0.02 if fraction > 0.82 else 1.0
                score = float(blob.pixels()) * flood_penalty
                if best is None or score > best[0]:
                    best = (score, threshold, blob, mode)
        if best is None:
            return base_threshold, None, "assisted_roi"
        return best[1], best[2], best[3]

    def preview(self, frame, roi):
        """Build a non-persistent visual preview for the confirmation page."""
        threshold, blob, segmentation_mode = self.analyze_roi(frame, roi)
        if blob is None:
            return {
                "valid": False,
                "quality": "invalid",
                "message": "未提取到工件，请重新框选",
                "threshold": list(threshold),
                "mode": segmentation_mode,
                "coverage": 0.0,
                "roi": tuple(roi),
                "mask": None,
                "blob_rect": None,
            }
        roi_area = max(1, int(roi[2]) * int(roi[3]))
        coverage = float(blob.pixels()) / roi_area
        blob_rect = tuple(blob.rect())
        margin = 3
        edge_touch = (
            blob_rect[0] <= roi[0] + margin
            or blob_rect[1] <= roi[1] + margin
            or blob_rect[0] + blob_rect[2] >= roi[0] + roi[2] - margin
            or blob_rect[1] + blob_rect[3] >= roi[1] + roi[3] - margin
        )
        quality = "good"
        message = "绿色框/白色区域应只覆盖工件"
        valid = True
        if coverage > 0.82:
            quality = "invalid"
            message = "前景覆盖过大，疑似选中了背景"
            valid = False
        elif coverage < 0.03:
            quality = "invalid"
            message = "前景覆盖过小，请扩大或重新框选"
            valid = False
        elif edge_touch:
            quality = "warning"
            message = "工件接触ROI边缘，建议留出背景边距"
        mask = frame.copy(roi=roi)
        mask.binary([tuple(threshold)])
        mask = mask.to_grayscale(copy=False)
        return {
            "valid": valid,
            "quality": quality,
            "message": message,
            "threshold": list(threshold),
            "mode": segmentation_mode,
            "coverage": coverage,
            "roi": tuple(roi),
            "mask": mask,
            "blob_rect": blob_rect,
            "edge_touch": edge_touch,
        }

    @staticmethod
    def _padded_rect(frame, blob, padding=8):
        x, y, width, height = blob.rect()
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(frame.width(), x + width + padding)
        bottom = min(frame.height(), y + height + padding)
        return left, top, right - left, bottom - top

    @staticmethod
    def _blob_nearest_rect(blobs, expected_rect):
        if not blobs:
            return None
        if not expected_rect:
            return max(blobs, key=lambda item: item.pixels())
        expected = tuple(int(value) for value in expected_rect)
        return min(
            blobs,
            key=lambda item: sum(abs(int(value) - expected[index]) for index, value in enumerate(item.rect())),
        )

    def _pose_image(self, gray):
        import image

        canvas = image.Image(self.pose_size, self.pose_size, image.GRAYSCALE)
        canvas.clear()
        scale = min((self.pose_size - 16) / gray.width(), (self.pose_size - 16) / gray.height())
        x = int((self.pose_size - gray.width() * scale) / 2)
        y = int((self.pose_size - gray.height() * scale) / 2)
        canvas.draw_image(gray, x, y, x_scale=scale, y_scale=scale)
        return canvas

    def create(self, frame, roi, name, template_id=None, threshold=None, segmentation_mode=None, confirmed_preview=None):
        confirmed_preview = confirmed_preview or {}
        if threshold is None and confirmed_preview.get("threshold") is not None:
            threshold = confirmed_preview.get("threshold")
        if segmentation_mode is None:
            segmentation_mode = confirmed_preview.get("mode")
        if threshold is None:
            threshold, blob, segmentation_mode = self.analyze_roi(frame, roi)
        else:
            blobs = frame.find_blobs([tuple(threshold)], roi=roi, pixels_threshold=80, area_threshold=120, merge=True)
            blob = self._blob_nearest_rect(blobs or [], confirmed_preview.get("blob_rect"))
            segmentation_mode = segmentation_mode or "assisted_roi"
        if blob is None:
            raise ValueError("No valid workpiece was found in the selected ROI")
        if template_id is None:
            checksum = 2166136261
            for character in str(name):
                checksum = ((checksum ^ ord(character)) * 16777619) & 0xFFFFFFFF
            template_id = "template_%08x" % checksum
        directory = self.template_root + "/" + template_id
        _ensure_dir(directory)
        rect = self._padded_rect(frame, blob)
        crop = frame.copy(roi=rect)
        gray = crop.to_grayscale(copy=True)
        mask = crop.copy()
        mask.binary([tuple(threshold)])
        mask = mask.to_grayscale(copy=False)
        edge = gray.copy()
        try:
            import image
            edge.find_edges(image.EDGE_CANNY, threshold=(45, 135))
        except Exception:
            edge = gray.copy()
        pose = self._pose_image(gray)
        assets = {
            "gray": "template_gray.pgm",
            "mask": "template_mask.pgm",
            "edge": "template_edge.pgm",
            "pose": "template_pose.pgm",
        }
        gray.save(directory + "/" + assets["gray"])
        mask.save(directory + "/" + assets["mask"])
        edge.save(directory + "/" + assets["edge"])
        pose.save(directory + "/" + assets["pose"])

        angle = math.degrees(blob.rotation())
        blob_width = max(1, int(blob.w()))
        blob_height = max(1, int(blob.h()))
        major_axis = _axis_length(blob, "major_axis_line", max(blob_width, blob_height))
        minor_axis = _axis_length(blob, "minor_axis_line", min(blob_width, blob_height))
        if minor_axis > major_axis:
            major_axis, minor_axis = minor_axis, major_axis
        axis_fill_ratio = float(blob.pixels()) / max(major_axis * minor_axis, 1.0)
        expected_pixels = int(blob.pixels())
        metadata = {
            "format_version": 1,
            "template_id": template_id,
            "name": str(name),
            "enabled": True,
            "width": int(rect[2]),
            "height": int(rect[3]),
            "reference_center_xy": [rect[2] / 2.0, rect[3] / 2.0],
            "pick_point_xy": [rect[2] / 2.0, rect[3] / 2.0],
            "pick_radius_px": min(rect[2], rect[3]) * 0.15,
            "mask_area_px": int(blob.pixels()),
            "fill_ratio": float(blob.density()),
            "shape_features": {
                "major_axis_length_px": major_axis,
                "minor_axis_length_px": minor_axis,
                "axis_fill_ratio": axis_fill_ratio,
            },
            "reference_angle_deg": float(angle),
            "pose_canvas_size": self.pose_size,
            "segmentation": {
                "mode": segmentation_mode,
                "lab_thresholds": [list(threshold)],
                "foreground_lab_median": [
                    (threshold[0] + threshold[1]) / 2.0,
                    (threshold[2] + threshold[3]) / 2.0,
                    (threshold[4] + threshold[5]) / 2.0,
                ],
                "exposure_tolerance_l": 10,
                "chroma_tolerance": 3,
                "pixels_threshold": max(40, min(800, int(expected_pixels * 0.04))),
                "area_threshold": max(60, min(1200, int(expected_pixels * 0.06))),
                "merge": True,
                "margin": 4,
            },
            "parameters": {
                "score_threshold": 0.68,
                "scale_min": 0.50,
                "scale_max": 1.80,
                "nms_iou": 0.25,
            },
            "assets": assets,
            "sha256": {},
        }
        metadata_file = template_id + "/metadata.json"
        _write_json_atomic(directory + "/metadata.json", metadata)
        manifest_path = self.template_root + "/template_library.json"
        manifest = _read_json(manifest_path, {"format_version": 1, "ambiguity_margin": 0.08, "templates": []})
        manifest["templates"] = [item for item in manifest.get("templates", []) if item.get("template_id") != template_id]
        manifest["templates"].append({
            "template_id": template_id,
            "name": str(name),
            "metadata_file": metadata_file,
            "metadata_sha256": "",
        })
        _write_json_atomic(manifest_path, manifest)
        return metadata
