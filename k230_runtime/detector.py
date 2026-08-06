"""CanMV-friendly multi-template candidate detector and pose estimator.

The detector deliberately performs segmentation first.  Full-frame exhaustive
rotation/scale correlation is too expensive for a MicroPython production loop.
"""

import math


def normalize_angle(angle):
    value = float(angle)
    while value >= 180.0:
        value -= 360.0
    while value < -180.0:
        value += 360.0
    return value


def normalize_gripper_angle(angle):
    """Normalize principal-axis pose because a blob axis is 180-degree periodic."""
    value = normalize_angle(angle)
    if value >= 90.0:
        value -= 180.0
    elif value < -90.0:
        value += 180.0
    return value


def _blob_value(blob, method, index):
    value = getattr(blob, method, None)
    if callable(value):
        return value()
    return blob[index]


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _axis_length(blob, method, fallback):
    value = getattr(blob, method, None)
    if not callable(value):
        return float(fallback)
    try:
        x1, y1, x2, y2 = value()
        return max(1.0, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    except Exception:
        return float(fallback)


def robust_lab_threshold(segmentation, item):
    """Expand exposure range while keeping colored targets away from neutral backgrounds."""
    values = list(item)
    if len(values) != 6:
        return None
    exposure_tolerance = int(segmentation.get("exposure_tolerance_l", 10))
    chroma_tolerance = int(segmentation.get("chroma_tolerance", 3))
    output = [
        max(0, int(values[0]) - exposure_tolerance),
        min(100, int(values[1]) + exposure_tolerance),
        max(-128, int(values[2]) - chroma_tolerance),
        min(127, int(values[3]) + chroma_tolerance),
        max(-128, int(values[4]) - chroma_tolerance),
        min(127, int(values[5]) + chroma_tolerance),
    ]
    if segmentation.get("mode") != "color":
        return _apply_discriminative_channel(segmentation, output)

    foreground = segmentation.get("foreground_lab_median", [])
    channel_medians = []
    for offset, low_index in ((1, 2), (2, 4)):
        if len(foreground) > offset:
            channel_medians.append(float(foreground[offset]))
        else:
            channel_medians.append((float(values[low_index]) + float(values[low_index + 1])) * 0.5)

    # A rectangular LAB threshold cannot express chroma magnitude.  Constrain
    # the dominant signed channel so the rectangle cannot cross neutral gray.
    dominant = 0 if abs(channel_medians[0]) >= abs(channel_medians[1]) else 1
    signed_median = channel_medians[dominant]
    low_index = 2 + dominant * 2
    if signed_median >= 4.0:
        cutoff = max(3, int(round(signed_median * 0.30)))
        output[low_index] = max(int(values[low_index]), cutoff)
    elif signed_median <= -4.0:
        cutoff = min(-3, int(round(signed_median * 0.30)))
        output[low_index + 1] = min(int(values[low_index + 1]), cutoff)
    return _apply_discriminative_channel(segmentation, output)


def _apply_discriminative_channel(segmentation, output):
    """Keep tolerant LAB ranges from swallowing the local background.

    Desktop template export records the single LAB channel that most clearly
    separates the masked workpiece from its immediate surroundings.  Applying
    this compact decision on K230 costs virtually nothing and is much more
    stable than relying on absolute luminance alone.
    """
    discriminator = segmentation.get("discriminative_channel", {})
    if not discriminator.get("enabled", False):
        return tuple(output)
    index = {"l": 0, "a": 2, "b": 4}.get(discriminator.get("channel"))
    if index is None:
        return tuple(output)
    cutoff = int(round(float(discriminator.get("cutoff", 0.0))))
    if discriminator.get("foreground_side") == "above":
        output[index] = max(output[index], cutoff)
    else:
        output[index + 1] = min(output[index + 1], cutoff)
    if output[index] > output[index + 1]:
        return None
    return tuple(output)


def _statistics_value(statistics, name):
    method = getattr(statistics, name, None)
    if callable(method):
        try:
            return float(method())
        except Exception:
            return None
    return None


def _blob_metric(blob, name):
    method = getattr(blob, name, None)
    if not callable(method):
        return None
    try:
        return float(method())
    except Exception:
        return None


def _similarity(observed, expected, tolerance):
    if observed is None or expected is None:
        return None
    return _clamp(1.0 - abs(float(observed) - float(expected)) / max(float(tolerance), 0.01))


class BlobTemplateDetector:
    """Detect candidates with ``find_blobs`` and classify by template geometry."""

    def __init__(self, template_library, ambiguity_margin=None):
        self.library = template_library
        manifest = template_library.manifest or {}
        if ambiguity_margin is None:
            ambiguity_margin = manifest.get("ambiguity_margin", 0.08)
        self.ambiguity_margin = float(ambiguity_margin)

    def _candidate(self, blob, template):
        x = int(_blob_value(blob, "x", 0))
        y = int(_blob_value(blob, "y", 1))
        width = max(1, int(_blob_value(blob, "w", 2)))
        height = max(1, int(_blob_value(blob, "h", 3)))
        pixels = max(1, int(_blob_value(blob, "pixels", 4)))
        center_x = float(_blob_value(blob, "cx", 5))
        center_y = float(_blob_value(blob, "cy", 6))
        rotation = float(_blob_value(blob, "rotation", 7))

        shape = template.get("shape_features", {})
        observed_long = _axis_length(blob, "major_axis_line", max(width, height))
        observed_short = _axis_length(blob, "minor_axis_line", min(width, height))
        if observed_short > observed_long:
            observed_long, observed_short = observed_short, observed_long
        template_long = float(shape.get("major_axis_length_px", max(template.get("width", 1), template.get("height", 1))))
        template_short = float(shape.get("minor_axis_length_px", max(1, min(template.get("width", 1), template.get("height", 1)))))
        bbox_scale = math.sqrt((observed_long / max(template_long, 1.0)) * (observed_short / max(template_short, 1.0)))
        reference_area = float(template.get("mask_area_px", max(1.0, template.get("fill_ratio", 0.5) * template.get("width", 1) * template.get("height", 1))))
        area_scale = math.sqrt(pixels / max(reference_area, 1.0))
        scale = 0.55 * bbox_scale + 0.45 * area_scale
        observed_ratio = observed_long / observed_short
        template_ratio = template_long / template_short
        ratio_score = _clamp(1.0 - abs(observed_ratio - template_ratio) / max(template_ratio, 0.01))

        parameters = template.get("parameters", {})
        scale_min = float(parameters.get("scale_min", 0.45))
        scale_max = float(parameters.get("scale_max", 1.80))
        if scale < scale_min or scale > scale_max:
            return None
        scale_consistency = _clamp(1.0 - abs(bbox_scale - area_scale) / max(bbox_scale, area_scale, 0.1))

        density = pixels / max(observed_long * observed_short, 1.0)
        fill_ratio = float(shape.get("axis_fill_ratio", template.get("fill_ratio", 0.5)))
        density_score = _clamp(1.0 - abs(density - fill_ratio) / max(fill_ratio, 0.15))
        # Geometry remains the inexpensive base classifier.  On firmware that
        # exposes richer blob descriptors, solidity/roundness/elongation make
        # similarly sized but differently shaped textile pieces separable.
        scores = [
            (ratio_score, 0.34),
            (density_score, 0.24),
            (scale_consistency, 0.18),
        ]
        optional_scores = (
            (_similarity(_blob_metric(blob, "elongation"), shape.get("elongation"), 0.35), 0.10),
            (_similarity(_blob_metric(blob, "solidity"), shape.get("solidity"), 0.30), 0.08),
            (_similarity(_blob_metric(blob, "roundness"), shape.get("roundness"), 0.30), 0.06),
        )
        scores.extend((score, weight) for score, weight in optional_scores if score is not None)
        total_weight = sum(weight for _, weight in scores)
        confidence = sum(score * weight for score, weight in scores) / max(total_weight, 0.01)

        score_threshold = float(parameters.get("score_threshold", 0.60))
        if confidence < score_threshold:
            return None

        reference_angle = float(template.get("reference_angle_deg", 0.0))
        angle = normalize_gripper_angle(math.degrees(rotation) - reference_angle)
        pick = self._pick_point(template, center_x, center_y, angle, scale)
        return {
            "template_id": template["template_id"],
            "template_name": template.get("name", template["template_id"]),
            "confidence": _clamp(confidence),
            "image_center": [center_x, center_y],
            "pick_point": pick,
            "angle_deg": angle,
            "scale": scale,
            "bbox": [x, y, width, height],
            "density": density,
            "score_threshold": score_threshold,
            "status": "ready",
            "candidate_templates": [],
        }

    def _pick_point(self, template, center_x, center_y, angle_deg, scale):
        reference = template.get("reference_center_xy", [template.get("width", 1) / 2, template.get("height", 1) / 2])
        pick = template.get("pick_point_xy", reference)
        dx = (float(pick[0]) - float(reference[0])) * scale
        dy = (float(pick[1]) - float(reference[1])) * scale
        radians = math.radians(angle_deg)
        return [
            center_x + dx * math.cos(radians) - dy * math.sin(radians),
            center_y + dx * math.sin(radians) + dy * math.cos(radians),
        ]

    @staticmethod
    def _scene_l_median(frame):
        method = getattr(frame, "get_statistics", None)
        if not callable(method):
            return None
        try:
            return _statistics_value(method(), "l_median")
        except Exception:
            return None

    @staticmethod
    def _scene_is_compatible(segmentation, scene_l):
        if scene_l is None or segmentation.get("mode") != "dark_texture":
            return True
        foreground_l = float(segmentation.get("foreground_l_median", scene_l))
        polarity = segmentation.get("polarity", "similar")
        if polarity == "brighter":
            return scene_l < foreground_l - 2.0
        if polarity == "darker":
            return scene_l > foreground_l + 2.0
        return True

    @staticmethod
    def _floods_frame(frame, blob):
        width_method = getattr(frame, "width", None)
        height_method = getattr(frame, "height", None)
        if not callable(width_method) or not callable(height_method):
            return False
        try:
            frame_width = max(1, int(width_method()))
            frame_height = max(1, int(height_method()))
            blob_width = int(_blob_value(blob, "w", 2))
            blob_height = int(_blob_value(blob, "h", 3))
            pixels = int(_blob_value(blob, "pixels", 4))
            return (
                blob_width >= frame_width * 0.90 and blob_height >= frame_height * 0.90
            ) or pixels >= frame_width * frame_height * 0.85
        except Exception:
            return False

    def _find_template_candidates(self, frame, template, scene_l=None):
        segmentation = template.get("segmentation", {})
        thresholds = segmentation.get("lab_thresholds", [])
        if not thresholds or not self._scene_is_compatible(segmentation, scene_l):
            return []
        robust_thresholds = []
        for item in thresholds:
            threshold = robust_lab_threshold(segmentation, item)
            if threshold is not None:
                robust_thresholds.append(threshold)
        if not robust_thresholds:
            return []
        blobs = frame.find_blobs(
            robust_thresholds,
            pixels_threshold=int(segmentation.get("pixels_threshold", 80)),
            area_threshold=int(segmentation.get("area_threshold", 120)),
            merge=bool(segmentation.get("merge", True)),
            margin=int(segmentation.get("margin", 4)),
        )
        output = []
        for blob in blobs or []:
            if self._floods_frame(frame, blob):
                continue
            candidate = self._candidate(blob, template)
            if candidate is not None:
                output.append(candidate)
        return output

    @staticmethod
    def _conflicts(first, second):
        ax, ay = first["image_center"]
        bx, by = second["image_center"]
        distance = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        short_side = min(first["bbox"][2], first["bbox"][3], second["bbox"][2], second["bbox"][3])
        return distance < max(12.0, short_side * 0.60)

    def _resolve_conflicts(self, candidates):
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        resolved = []
        consumed = set()
        for index, best in enumerate(candidates):
            if index in consumed:
                continue
            group = [best]
            for other_index in range(index + 1, len(candidates)):
                if other_index in consumed:
                    continue
                other = candidates[other_index]
                if self._conflicts(best, other):
                    group.append(other)
                    consumed.add(other_index)
            group.sort(key=lambda item: item["confidence"], reverse=True)
            winner = group[0]
            winner["candidate_templates"] = [
                {"template_id": item["template_id"], "name": item["template_name"], "score": item["confidence"]}
                for item in group
            ]
            if len(group) > 1 and group[0]["template_id"] != group[1]["template_id"]:
                if group[0]["confidence"] - group[1]["confidence"] < self.ambiguity_margin:
                    winner["status"] = "ambiguous"
            resolved.append(winner)
        resolved.sort(key=lambda item: (item["image_center"][1], item["image_center"][0]))
        return resolved

    def detect(self, frame, timestamp_ms=0):
        candidates = []
        scene_l = self._scene_l_median(frame)
        for template in self.library.templates:
            if template.get("enabled", True):
                candidates.extend(self._find_template_candidates(frame, template, scene_l))
        return self._resolve_conflicts(candidates)
