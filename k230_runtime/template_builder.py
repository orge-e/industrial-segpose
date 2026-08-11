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


def _stat_value(statistics, channel, suffix="mean", fallback=0.0):
    method = getattr(statistics, channel + "_" + suffix, None)
    try:
        return float(method()) if callable(method) else float(fallback)
    except Exception:
        return float(fallback)


def _channel_contrast_threshold(channel, foreground, background):
    """Return a one-sided LAB threshold when foreground differs from background."""
    limits = {"l": (0, 100, 4.0), "a": (-128, 127, 2.5), "b": (-128, 127, 2.5)}
    low, high, minimum_delta = limits[channel]
    foreground_value = _stat_value(foreground, channel)
    background_value = _stat_value(background, channel)
    delta = foreground_value - background_value
    if abs(delta) < minimum_delta:
        return None
    midpoint = int(round((foreground_value + background_value) / 2.0))
    if delta > 0:
        return max(low, midpoint), high
    return low, min(high, midpoint)


def contrast_threshold_candidates(frame, roi):
    """Build generic candidates from a centre foreground seed and ROI corners."""
    x, y, width, height = [int(value) for value in roi]
    if width < 32 or height < 32:
        return []
    center_roi = (x + width // 4, y + height // 4, max(8, width // 2), max(8, height // 2))
    sample_w = max(6, min(width // 5, 24))
    sample_h = max(6, min(height // 5, 24))
    corner_rois = (
        (x, y, sample_w, sample_h),
        (x + width - sample_w, y, sample_w, sample_h),
        (x, y + height - sample_h, sample_w, sample_h),
        (x + width - sample_w, y + height - sample_h, sample_w, sample_h),
    )
    foreground = frame.get_statistics(roi=center_roi)
    backgrounds = [frame.get_statistics(roi=item) for item in corner_rois]

    class AveragedStatistics:
        pass

    background = AveragedStatistics()
    for channel in ("l", "a", "b"):
        value = sum(_stat_value(item, channel) for item in backgrounds) / len(backgrounds)
        setattr(background, channel + "_mean", lambda value=value: value)

    candidates = []
    indexes = {"l": 0, "a": 2, "b": 4}
    for channel in ("l", "a", "b"):
        bounds = _channel_contrast_threshold(channel, foreground, background)
        if bounds is None:
            continue
        threshold = [0, 100, -128, 127, -128, 127]
        index = indexes[channel]
        threshold[index:index + 2] = list(bounds)
        candidates.append(("contrast_" + channel, threshold))
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
    def __init__(self, template_root, pose_size=192, diagnostic_root=None):
        self.template_root = template_root.rstrip("/")
        self.pose_size = int(pose_size)
        application_root = self.template_root.rsplit("/", 1)[0]
        self.diagnostic_root = (
            str(diagnostic_root).rstrip("/")
            if diagnostic_root is not None
            else application_root + "/diagnostics/latest_template"
        )

    @staticmethod
    def _edge_touch_count(blob_rect, roi, margin=3):
        bx, by, bw, bh = [int(value) for value in blob_rect]
        rx, ry, rw, rh = [int(value) for value in roi]
        return sum((
            bx <= rx + margin,
            by <= ry + margin,
            bx + bw >= rx + rw - margin,
            by + bh >= ry + rh - margin,
        ))

    @staticmethod
    def _blob_density(blob):
        method = getattr(blob, "density", None)
        if callable(method):
            try:
                return max(0.0, min(1.0, float(method())))
            except Exception:
                pass
        rect = blob.rect()
        return max(0.0, min(1.0, float(blob.pixels()) / max(1, rect[2] * rect[3])))

    def _score_candidate(self, blob, blobs, roi):
        roi_area = max(1, int(roi[2]) * int(roi[3]))
        pixels = float(blob.pixels())
        coverage = pixels / roi_area
        density = self._blob_density(blob)
        edge_touches = self._edge_touch_count(blob.rect(), roi)
        total_pixels = sum(max(0, int(item.pixels())) for item in blobs) or 1
        dominance = pixels / total_pixels
        valid = True
        reason = "ok"
        # A manually confirmed template ROI should contain a meaningful part
        # of the workpiece. Highlights, shadows and loose threads commonly
        # occupy only a few percent of the ROI and must not become templates.
        if coverage < 0.08:
            valid, reason = False, "foreground_too_small"
        elif coverage > 0.78:
            valid, reason = False, "background_flood"
        elif edge_touches >= 3:
            valid, reason = False, "touches_roi_edges"
        elif density < 0.18:
            valid, reason = False, "fragmented"

        size_score = min(1.0, coverage / 0.18)
        edge_score = (1.0, 0.72, 0.36, 0.0, 0.0)[min(edge_touches, 4)]
        fragment_score = 1.0 / (1.0 + max(0, len(blobs) - 1) * 0.18)
        score = (
            0.30 * size_score + 0.28 * density + 0.22 * dominance
            + 0.12 * edge_score + 0.08 * fragment_score
        )
        if not valid:
            score *= 0.02
        return {
            "score": float(score), "valid": bool(valid), "reason": reason,
            "coverage": float(coverage), "density": float(density),
            "edge_touches": int(edge_touches), "component_count": int(len(blobs)),
            "dominance": float(dominance),
        }

    def _analyze_roi_details(self, frame, roi):
        statistics = frame.get_statistics(roi=roi)
        base_threshold = suggested_lab_threshold(statistics)
        candidates = assisted_threshold_candidates(base_threshold)
        try:
            candidates.extend(contrast_threshold_candidates(frame, roi))
        except Exception:
            pass
        unique = []
        seen = set()
        for mode, threshold in candidates:
            key = tuple(int(value) for value in threshold)
            if key not in seen:
                unique.append((mode, list(key)))
                seen.add(key)

        best = None
        rejected = []
        for mode, threshold in unique:
            blobs = frame.find_blobs(
                [tuple(threshold)], roi=roi, pixels_threshold=80,
                area_threshold=120, merge=True,
            ) or []
            for blob in blobs:
                metrics = self._score_candidate(blob, blobs, roi)
                record = (metrics["score"], threshold, blob, mode, metrics)
                if metrics["valid"] and (best is None or record[0] > best[0]):
                    best = record
                elif not metrics["valid"]:
                    rejected.append(record)
        if best is None:
            reason = "no_candidate"
            if rejected:
                reason = max(rejected, key=lambda item: item[0])[4]["reason"]
            return {
                "threshold": base_threshold, "blob": None, "mode": "assisted_roi",
                "metrics": {"valid": False, "reason": reason, "score": 0.0},
            }
        return {
            "threshold": best[1], "blob": best[2], "mode": best[3], "metrics": best[4],
        }

    def analyze_roi(self, frame, roi):
        result = self._analyze_roi_details(frame, roi)
        return result["threshold"], result["blob"], result["mode"]

    @staticmethod
    def _clean_binary_mask(mask):
        try:
            mask.dilate(1)
            mask.erode(1)
            mask.erode(1)
            mask.dilate(1)
        except Exception:
            pass
        return mask

    def _mask_for_rect(self, frame, rect, threshold):
        mask = frame.copy(roi=rect)
        mask.binary([tuple(threshold)])
        mask = mask.to_grayscale(copy=False)
        return self._clean_binary_mask(mask)

    @staticmethod
    def _mask_overlay_layer(mask):
        """Create a green RGB565 layer that can be drawn through ``mask``."""
        try:
            import image
            layer = image.Image(mask.width(), mask.height(), image.RGB565)
            layer.clear()
            layer.draw_rectangle(
                0, 0, mask.width(), mask.height(),
                color=(0, 255, 80), thickness=1, fill=True,
            )
            return layer
        except Exception:
            return None

    def preview(self, frame, roi):
        """Build a non-persistent visual preview for the confirmation page."""
        analysis = self._analyze_roi_details(frame, roi)
        threshold = analysis["threshold"]
        blob = analysis["blob"]
        segmentation_mode = analysis["mode"]
        metrics = analysis["metrics"]
        if blob is None:
            return {
                "valid": False,
                "quality": "invalid",
                "message": "分割失败：请调整ROI、背景或光照",
                "threshold": list(threshold),
                "mode": segmentation_mode,
                "coverage": 0.0,
                "roi": tuple(roi),
                "mask": None,
                "blob_rect": None,
                "reason": metrics.get("reason", "no_candidate"),
            }
        coverage = metrics["coverage"]
        blob_rect = tuple(blob.rect())
        edge_touches = metrics["edge_touches"]
        edge_touch = edge_touches > 0
        quality = "good"
        message = "绿色框/白色区域应只覆盖工件"
        valid = True
        if not metrics.get("valid", False):
            quality = "invalid"
            message = "分割无效：" + metrics.get("reason", "unknown")
            valid = False
        elif edge_touches >= 2:
            quality = "invalid"
            message = "工件接触多个ROI边缘，请扩大ROI并留出背景"
            valid = False
        elif edge_touches == 1:
            quality = "warning"
            message = "工件接触ROI边缘，建议留出背景边距"
        rect = self._padded_rect(frame, blob)
        mask = self._mask_for_rect(frame, rect, threshold)
        mask_overlay = self._mask_overlay_layer(mask)
        return {
            "valid": valid,
            "quality": quality,
            "message": message,
            "threshold": list(threshold),
            "mode": segmentation_mode,
            "coverage": coverage,
            "roi": tuple(roi),
            "mask": mask,
            "mask_overlay": mask_overlay,
            "blob_rect": blob_rect,
            "edge_touch": edge_touch,
            "edge_touches": edge_touches,
            "score": metrics.get("score", 0.0),
            "density": metrics.get("density", 0.0),
            "component_count": metrics.get("component_count", 0),
            "reason": metrics.get("reason", "ok"),
            "mask_rect": tuple(rect),
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

    def _reset_diagnostics(self):
        """Keep only the latest template diagnostic package on the SD card."""
        _ensure_dir(self.diagnostic_root)
        try:
            names = os.listdir(self.diagnostic_root)
        except OSError:
            names = []
        for name in names:
            try:
                os.remove(self.diagnostic_root + "/" + name)
            except OSError:
                pass

    @staticmethod
    def _blob_debug_record(blob, roi_area):
        pixels = int(blob.pixels())
        return {
            "rect": [int(value) for value in blob.rect()],
            "pixels": pixels,
            "coverage": float(pixels / max(1, roi_area)),
        }

    def _save_diagnostics(self, frame, roi, crop, gray, mask, edge, pose, selected_threshold, selected_mode):
        """Save the complete segmentation trail without depending on an RTC."""
        self._reset_diagnostics()
        root = self.diagnostic_root
        frame.save(root + "/00_frozen_frame.jpg")
        roi_image = frame.copy(roi=roi)
        roi_image.save(root + "/01_roi_original.bmp")

        statistics = frame.get_statistics(roi=roi)
        base_threshold = suggested_lab_threshold(statistics)
        roi_area = max(1, int(roi[2]) * int(roi[3]))
        candidate_records = []
        diagnostic_candidates = assisted_threshold_candidates(base_threshold)
        try:
            diagnostic_candidates.extend(contrast_threshold_candidates(frame, roi))
        except Exception:
            pass
        unique_candidates = []
        seen = set()
        for mode, threshold in diagnostic_candidates:
            key = tuple(int(value) for value in threshold)
            if key not in seen:
                unique_candidates.append((mode, list(key)))
                seen.add(key)
        for index, (mode, threshold) in enumerate(unique_candidates):
            candidate_mask = frame.copy(roi=roi)
            candidate_mask.binary([tuple(threshold)])
            candidate_mask = candidate_mask.to_grayscale(copy=False)
            candidate_mask.save(root + "/10_candidate_%02d_%s.bmp" % (index, mode))
            blobs = frame.find_blobs(
                [tuple(threshold)], roi=roi,
                pixels_threshold=80, area_threshold=120, merge=True,
            )
            blob_records = []
            for blob in (blobs or []):
                record = self._blob_debug_record(blob, roi_area)
                record.update(self._score_candidate(blob, blobs or [], roi))
                blob_records.append(record)
            candidate_records.append({
                "index": index,
                "mode": mode,
                "threshold": list(threshold),
                "selected": list(threshold) == list(selected_threshold),
                "blobs": blob_records,
            })

        crop.save(root + "/20_template_crop.bmp")
        mask.save(root + "/21_final_mask.bmp")
        gray.save(root + "/22_gray.bmp")
        edge.save(root + "/23_edge.bmp")
        pose.save(root + "/24_pose.bmp")
        payload = {
            "format_version": 1,
            "source_size": [int(frame.width()), int(frame.height())],
            "roi_xywh": [int(value) for value in roi],
            "selected_mode": selected_mode,
            "selected_threshold": list(selected_threshold),
            "base_threshold": list(base_threshold),
            "candidate_count": len(candidate_records),
            "candidates": candidate_records,
            "files": {
                "frozen_frame": "00_frozen_frame.jpg",
                "roi_original": "01_roi_original.bmp",
                "template_crop": "20_template_crop.bmp",
                "final_mask": "21_final_mask.bmp",
                "gray": "22_gray.bmp",
                "edge": "23_edge.bmp",
                "pose": "24_pose.bmp",
            },
        }
        _write_json_atomic(root + "/segmentation_debug.json", payload)
        return root

    def create(self, frame, roi, name, template_id=None, threshold=None, segmentation_mode=None, confirmed_preview=None):
        confirmed_preview = confirmed_preview or {}
        if confirmed_preview and not confirmed_preview.get("valid", False):
            raise ValueError("Segmentation preview is invalid")
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
        mask = self._mask_for_rect(frame, rect, threshold)
        # Use the binary silhouette for edge extraction. It remains stable on
        # pale workpieces where grayscale contrast is nearly absent.
        edge = mask.copy()
        try:
            import image
            edge.find_edges(image.EDGE_CANNY, threshold=(45, 135))
        except Exception:
            edge = gray.copy()
        pose = self._pose_image(mask)
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

        blob_width = max(1, int(blob.w()))
        blob_height = max(1, int(blob.h()))
        major_axis = _axis_length(blob, "major_axis_line", max(blob_width, blob_height))
        minor_axis = _axis_length(blob, "minor_axis_line", min(blob_width, blob_height))
        if minor_axis > major_axis:
            major_axis, minor_axis = minor_axis, major_axis
        axis_fill_ratio = float(blob.pixels()) / max(major_axis * minor_axis, 1.0)
        expected_pixels = int(blob.pixels())
        base_angle = float(math.degrees(blob.rotation()))
        center_x = float(blob.cx())
        center_y = float(blob.cy())
        bbox_center_x = float(blob.x()) + float(blob.w()) / 2.0
        bbox_center_y = float(blob.y()) + float(blob.h()) / 2.0
        unit_x = math.cos(math.radians(base_angle))
        unit_y = math.sin(math.radians(base_angle))
        direction_projection = ((bbox_center_x - center_x) * unit_x + (bbox_center_y - center_y) * unit_y)
        if direction_projection < 0.0:
            base_angle += 180.0
            direction_projection = -direction_projection
        reference_angle = base_angle % 360.0
        direction_confidence = min(1.0, direction_projection / max(major_axis * 0.25, 1.0))
        metadata = {
            "format_version": 1,
            "template_id": template_id,
            "name": str(name),
            "enabled": True,
            "width": int(rect[2]),
            "height": int(rect[3]),
            "reference_center_xy": [float(blob.cx() - rect[0]), float(blob.cy() - rect[1])],
            "pick_point_xy": [float(blob.cx() - rect[0]), float(blob.cy() - rect[1])],
            "pick_radius_px": min(rect[2], rect[3]) * 0.15,
            "pick_points": [{
                "x": float(blob.cx() - rect[0]),
                "y": float(blob.cy() - rect[1]),
                "safe_radius_px": min(rect[2], rect[3]) * 0.15,
                "score": 1.0,
            }],
            "pick_planning": {
                "method": "board_template_fallback",
                "minimum_safe_radius_px": 3.0,
                "candidate_count": 1,
            },
            "mask_area_px": int(blob.pixels()),
            "fill_ratio": float(blob.density()),
            "shape_features": {
                "major_axis_length_px": major_axis,
                "minor_axis_length_px": minor_axis,
                "axis_fill_ratio": axis_fill_ratio,
            },
            "reference_angle_deg": float(reference_angle),
            "orientation_period_deg": 360 if direction_confidence >= 0.08 else 180,
            "orientation_direction_confidence": float(direction_confidence),
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
        try:
            diagnostic_path = self._save_diagnostics(
                frame, roi, crop, gray, mask, edge, pose, threshold, segmentation_mode,
            )
            metadata["diagnostics"] = {
                "status": "saved",
                "path": diagnostic_path,
                "retention": "latest_only",
            }
        except Exception as diagnostic_error:
            # Diagnostics must never make an otherwise valid template fail.
            print("template diagnostics failed", diagnostic_error)
            metadata["diagnostics"] = {
                "status": "failed",
                "error": str(diagnostic_error),
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
