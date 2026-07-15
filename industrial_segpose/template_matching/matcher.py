"""Multi-angle, multi-scale template matching with rotated-box NMS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, cos, radians, sin
from typing import Callable, Iterable

import cv2
import numpy as np

from .model import TemplateModel
from .mask_assist import extract_dark_textile_candidates


@dataclass(frozen=True)
class MatchParameters:
    score_threshold: float = 0.72
    angle_min: float = -180.0
    angle_max: float = 180.0
    angle_step: float = 2.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    scale_step: float = 0.1
    nms_iou_threshold: float = 0.25
    use_edges: bool = True
    feature_mode: str = "auto"
    max_candidates_per_transform: int = 20
    max_results: int = 200

    def validate(self) -> None:
        if not 0.0 < self.score_threshold <= 1.0:
            raise ValueError("Score threshold must be in (0, 1]")
        if self.angle_max < self.angle_min or self.angle_step <= 0:
            raise ValueError("Invalid angle range or step")
        if self.scale_min <= 0 or self.scale_max < self.scale_min or self.scale_step <= 0:
            raise ValueError("Invalid scale range or step")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("NMS IoU threshold must be in [0, 1]")
        if self.max_candidates_per_transform < 1 or self.max_results < 1:
            raise ValueError("Candidate and result limits must be positive")
        if self.feature_mode not in {"auto", "edges", "gray", "textile_chroma", "pose_tolerant", "dark_textile"}:
            raise ValueError("Feature mode must be auto, edges, gray, textile_chroma, pose_tolerant, or dark_textile")
        angle_count = int((self.angle_max - self.angle_min) / self.angle_step) + 1
        scale_count = int((self.scale_max - self.scale_min) / self.scale_step) + 1
        if angle_count * scale_count > 5000:
            raise ValueError("Angle/scale search contains more than 5000 transforms")


@dataclass(frozen=True)
class TemplateMatch:
    object_id: int
    template_name: str
    center_x: float
    center_y: float
    angle_deg: float
    score: float
    scale: float
    box_points: tuple[tuple[float, float], ...]
    contour_points: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["box_points"] = [list(point) for point in self.box_points]
        payload["contour_points"] = [list(point) for point in self.contour_points]
        return payload


@dataclass(frozen=True)
class _Candidate:
    center_x: float
    center_y: float
    angle_deg: float
    score: float
    scale: float
    box: np.ndarray
    contour: np.ndarray
    suppression_radius: float


@dataclass(frozen=True)
class _TemplateVariant:
    image: np.ndarray
    mask: np.ndarray
    center: np.ndarray
    contour: np.ndarray
    angle: float
    scale: float


def _values(start: float, stop: float, step: float) -> list[float]:
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = [start + index * step for index in range(count)]
    if values and values[-1] < stop - 1e-9:
        values.append(stop)
    return values


def _normalize_angle(angle: float) -> float:
    value = (angle + 180.0) % 360.0 - 180.0
    return 0.0 if abs(value) < 1e-9 else float(value)


def _rotation_geometry(width: int, height: int, angle: float) -> tuple[np.ndarray, int, int]:
    theta = radians(angle)
    new_width = max(1, int(ceil(abs(width * cos(theta)) + abs(height * sin(theta)))))
    new_height = max(1, int(ceil(abs(height * cos(theta)) + abs(width * sin(theta)))))
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[0, 2] += (new_width - width) / 2.0
    matrix[1, 2] += (new_height - height) / 2.0
    return matrix, new_width, new_height


def _rotate_expanded(
    image: np.ndarray,
    angle: float,
    border_value: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix, new_width, new_height = _rotation_geometry(width, height, angle)
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _rotated_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_first = abs(float(cv2.contourArea(first.astype(np.float32))))
    area_second = abs(float(cv2.contourArea(second.astype(np.float32))))
    if area_first <= 0 or area_second <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(
        first.astype(np.float32), second.astype(np.float32)
    )
    return float(intersection) / max(area_first + area_second - float(intersection), 1e-9)


class TemplateMatcher:
    def __init__(self, model: TemplateModel, parameters: MatchParameters | None = None):
        self.model = model
        self.parameters = parameters or MatchParameters()
        self.parameters.validate()
        self._variant_cache_key: MatchParameters | None = None
        self._variant_cache: list[_TemplateVariant] = []

    @staticmethod
    def _prepare(image: np.ndarray, use_edges: bool, feature_mode: str = "auto") -> np.ndarray:
        if feature_mode in {"textile_chroma", "pose_tolerant"}:
            bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            red_chroma = np.clip(lab[:, :, 1].astype(np.int16) - 128, 0, 127).astype(np.uint8)
            saturation = hsv[:, :, 1]
            feature = cv2.max(red_chroma, (saturation // 2).astype(np.uint8))
            if feature_mode == "pose_tolerant":
                border = np.concatenate((feature[0], feature[-1], feature[:, 0], feature[:, -1]))
                baseline = float(np.median(border)) if border.size else 0.0
                threshold = int(np.clip(round(baseline + 5.0), 8, 48))
                foreground = np.where(feature >= threshold, 255, 0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
                edges = cv2.Canny(foreground, 40, 120)
                return cv2.GaussianBlur(edges, (0, 0), 2.2)
            return cv2.GaussianBlur(feature, (5, 5), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if feature_mode == "edges" or (feature_mode == "auto" and use_edges):
            return cv2.Canny(gray, 50, 150)
        return gray

    def match(
        self,
        image: np.ndarray,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[TemplateMatch]:
        if image is None or image.size == 0:
            raise ValueError("Detection image is empty")
        params = self.parameters
        params.validate()
        if params.feature_mode in {"pose_tolerant", "dark_textile"}:
            matches = self._match_pose_tolerant(image)
            if progress:
                progress(1, 1)
            return matches
        scene = self._prepare(image, params.use_edges, params.feature_mode)
        variants = self._variants()
        total = len(variants)
        candidates: list[_Candidate] = []
        for done, variant in enumerate(variants, 1):
            rotated = variant.image
            rotated_mask = variant.mask
            rotated_height, rotated_width = rotated.shape[:2]
            if rotated_height <= scene.shape[0] and rotated_width <= scene.shape[1]:
                response = cv2.matchTemplate(scene, rotated, cv2.TM_CCORR_NORMED, mask=rotated_mask)
                response[~np.isfinite(response)] = -1.0
                local_max = response == cv2.dilate(response, np.ones((3, 3), np.uint8))
                ys, xs = np.where(local_max & (response >= params.score_threshold))
                if len(xs):
                    scores = response[ys, xs]
                    order = np.argsort(scores)[::-1][: params.max_candidates_per_transform]
                    for index in order:
                        offset = np.asarray([float(xs[index]), float(ys[index])], dtype=np.float32)
                        contour = variant.contour + offset
                        center_x = float(variant.center[0] + offset[0])
                        center_y = float(variant.center[1] + offset[1])
                        rect = cv2.minAreaRect(contour.astype(np.float32))
                        box = cv2.boxPoints(rect)
                        hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2)
                        radius = 0.6 * min(float(rect[1][0]), float(rect[1][1]))
                        candidates.append(_Candidate(center_x, center_y, _normalize_angle(variant.angle), float(scores[index]), float(variant.scale), box, hull, radius))
            if progress:
                progress(done, total)
        return self._finalize(candidates)

    @staticmethod
    def _directed_mask_pose(mask: np.ndarray) -> tuple[float, np.ndarray]:
        """Return a 360-degree PCA pose, resolving the usual 180-degree ambiguity.

        The sign of the major axis is selected from the third moment along that
        axis.  This is stable for the asymmetric textile parts used here and
        keeps the reported angle compatible with OpenCV's rotation convention.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < 25:
            raise ValueError("Pose mask contains too few pixels")
        points = np.column_stack((xs, ys)).astype(np.float64)
        center = points.mean(axis=0)
        centered = points - center
        covariance = np.cov(centered.T)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        projection = centered @ axis
        if float(np.mean(projection ** 3)) < 0.0:
            axis = -axis
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return angle, center.astype(np.float32)

    def _pose_candidate_mask(self, image: np.ndarray) -> np.ndarray:
        """Segment a lightly coloured workpiece from a dark/neutral conveyor.

        Instead of a fixed RGB threshold, the chroma direction is learned from
        pixels inside the stored irregular template mask.  This tolerates large
        exposure changes while rejecting neutral black/gray textile distractors.
        """
        if self.parameters.feature_mode == "dark_textile":
            return extract_dark_textile_candidates(image)
        denoised = cv2.medianBlur(image, 3)
        template_lab = cv2.cvtColor(self.model.image, cv2.COLOR_BGR2LAB)
        selected = self.model.mask > 0
        median = np.median(template_lab[selected], axis=0).astype(np.float32)
        chroma_vector = median[1:] - 128.0
        chroma_length = float(np.linalg.norm(chroma_vector))
        scene_lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        scene_lightness = scene_lab[:, :, 0].astype(np.float32)
        low, high = np.percentile(scene_lightness, (2.0, 98.0))
        adaptive_floor = max(15.0, float(low + 0.05 * (high - low)))
        lightness_ok = scene_lightness >= adaptive_floor
        if chroma_length >= 3.0:
            chroma = scene_lab[:, :, 1:].astype(np.float32) - 128.0
            projection = np.tensordot(chroma, chroma_vector, axes=([2], [0])) / chroma_length
            chroma_ok = projection >= max(2.5, 0.35 * chroma_length)
            # Strong exposure can wash pale pink nearly to neutral.  On a dark
            # conveyor, retain a complementary high-lightness route so the
            # silhouette survives colour clipping; shape verification later
            # rejects broad glare regions and unrelated bright objects.
            bright_threshold = max(float(np.percentile(scene_lightness, 90.0)), float(low + 0.45 * (high - low)))
            # The achromatic highlight fallback is useful only when the scene is
            # predominantly dark (the intended conveyor).  Enabling it on a
            # pale table would classify the whole background as foreground.
            dark_scene = float(np.median(scene_lightness)) < 170.0
            bright_ok = (scene_lightness >= bright_threshold) if dark_scene else np.zeros_like(lightness_ok)
            # Reliable template-directed chroma must not be gated by scene
            # brightness: the textile can be darker than a pale table or sit
            # inside a local shadow.  The achromatic fallback remains
            # brightness-gated because it is intentionally less selective.
            raw = chroma_ok | (lightness_ok & bright_ok)
        else:
            # Neutral templates cannot provide a useful colour direction.
            # Retain the light foreground path for dark conveyor applications.
            raw = lightness_ok & (scene_lab[:, :, 0].astype(np.float32) >= float(median[0]) * 0.70)
        binary = np.where(raw, 255, 0).astype(np.uint8)
        size = max(3, int(round(min(image.shape[:2]) * 0.0035)))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    def _angle_near_search_range(self, angle: float) -> float | None:
        params = self.parameters
        options = [angle - 360.0, angle, angle + 360.0]
        tolerance = max(1.0, params.angle_step)
        eligible = [
            min(params.angle_max, max(params.angle_min, value))
            for value in options
            if params.angle_min - tolerance <= value <= params.angle_max + tolerance
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda value: abs(value - angle))

    def _match_pose_tolerant(self, image: np.ndarray) -> list[TemplateMatch]:
        """Fast component-guided silhouette matching for flexible textile parts."""
        params = self.parameters
        binary = self._pose_candidate_mask(image)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        dark_texture_strength: np.ndarray | None = None
        if params.feature_mode == "dark_textile":
            gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            median = cv2.medianBlur(gray_u8, 3)
            cleaned = np.where(cv2.absdiff(gray_u8, median) > 28, median, gray_u8).astype(np.float32)
            local_illumination = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
            dark_texture_strength = np.abs(cleaned - local_illumination) / (local_illumination + 18.0) * 110.0
        template_mask = self.model.mask > 0
        template_area = float(np.count_nonzero(template_mask))
        template_angle, _ = self._directed_mask_pose(template_mask)
        reference_center = np.asarray(self.model.reference_center_xy, dtype=np.float32)
        minimum_area = max(25.0, template_area * params.scale_min * params.scale_min * 0.15)
        candidates: list[_Candidate] = []

        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component = labels == label
            if dark_texture_strength is not None:
                x, y, width, height = map(int, stats[label, :4])
                area_fraction = area / max(float(image.shape[0] * image.shape[1]), 1.0)
                touches_border = x == 0 or y == 0 or x + width == image.shape[1] or y + height == image.shape[0]
                texture_median, texture_upper_quartile = np.percentile(dark_texture_strength[component], (50.0, 75.0))
                # Smooth conveyor gradients can form a large template-like
                # silhouette.  A true black textile has distributed weave
                # energy, while that background response is both weaker and
                # commonly connected to an image border.  Partial, very large
                # border objects are also unsafe pickup targets.
                if texture_median < 5.0 or texture_upper_quartile < 9.0:
                    continue
                if area_fraction > 0.55 or (touches_border and area_fraction > 0.12):
                    continue
            try:
                component_angle, component_center = self._directed_mask_pose(component)
            except ValueError:
                continue
            raw_angle = _normalize_angle(-(component_angle - template_angle))
            angle_hypotheses: list[float] = []
            for hypothesis in (raw_angle, _normalize_angle(raw_angle + 180.0)):
                adjusted = self._angle_near_search_range(hypothesis)
                if adjusted is not None and all(abs(adjusted - item) > 1e-6 for item in angle_hypotheses):
                    angle_hypotheses.append(adjusted)
            if not angle_hypotheses:
                continue
            estimated_scale = float(np.sqrt(area / max(template_area, 1.0)))
            if estimated_scale < params.scale_min * 0.82 or estimated_scale > params.scale_max * 1.18:
                continue

            x, y, width, height = map(int, stats[label, :4])
            padding = max(6, int(round(max(width, height) * 0.14)))
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1 = min(image.shape[1], x + width + padding)
            y1 = min(image.shape[0], y + height + padding)
            target = component[y0:y1, x0:x1]
            best_score, best_angle, best_scale = -1.0, angle_hypotheses[0], estimated_scale
            angle_offsets = (-4.0, -2.0, 0.0, 2.0, 4.0)
            scale_factors = (0.94, 0.97, 1.0, 1.03, 1.06)
            scale_options = sorted({
                min(params.scale_max, max(params.scale_min, estimated_scale * factor))
                for factor in scale_factors
            })
            for base_angle in angle_hypotheses:
                for angle_offset in angle_offsets:
                    angle = self._angle_near_search_range(base_angle + angle_offset)
                    if angle is None:
                        continue
                    for scale in scale_options:
                        matrix = cv2.getRotationMatrix2D(tuple(reference_center), angle, scale)
                        matrix[0, 2] += float(component_center[0] - reference_center[0] - x0)
                        matrix[1, 2] += float(component_center[1] - reference_center[1] - y0)
                        predicted = cv2.warpAffine(
                            self.model.mask,
                            matrix,
                            (x1 - x0, y1 - y0),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        ) > 0
                        intersection = float(np.count_nonzero(predicted & target))
                        score = 2.0 * intersection / max(float(np.count_nonzero(predicted) + np.count_nonzero(target)), 1.0)
                        if score > best_score:
                            best_score, best_angle, best_scale = score, angle, scale
            if best_score < params.score_threshold:
                continue

            contour_list, _ = cv2.findContours(
                np.where(component, 255, 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contour_list:
                continue
            contour = max(contour_list, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            hull = cv2.convexHull(contour).reshape(-1, 2)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            radius = 0.6 * min(float(rect[1][0]), float(rect[1][1]))
            candidates.append(
                _Candidate(
                    float(component_center[0]),
                    float(component_center[1]),
                    _normalize_angle(best_angle),
                    float(best_score),
                    float(best_scale),
                    box,
                    hull,
                    radius,
                )
            )
        return self._finalize(candidates)

    def _variants(self) -> list[_TemplateVariant]:
        params = self.parameters
        if self._variant_cache_key == params and self._variant_cache:
            return self._variant_cache
        template = self._prepare(self.model.image, params.use_edges, params.feature_mode)
        template_mask = self.model.mask
        reference_center = np.asarray([self.model.reference_center_xy], dtype=np.float32).reshape(-1, 1, 2)
        template_contour = self.model.contour
        angles = _values(params.angle_min, params.angle_max, params.angle_step)
        scales = _values(params.scale_min, params.scale_max, params.scale_step)
        variants: list[_TemplateVariant] = []
        for scale in scales:
            scaled_width = max(5, int(round(template.shape[1] * scale)))
            scaled_height = max(5, int(round(template.shape[0] * scale)))
            scaled = cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(template_mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
            scale_x = scaled_width / float(template.shape[1])
            scale_y = scaled_height / float(template.shape[0])
            scaled_center = reference_center.copy()
            scaled_center[:, :, 0] *= scale_x
            scaled_center[:, :, 1] *= scale_y
            scaled_contour = template_contour.copy()
            scaled_contour[:, 0] *= scale_x
            scaled_contour[:, 1] *= scale_y
            border = 0
            for angle in angles:
                matrix, rotated_width, rotated_height = _rotation_geometry(scaled_width, scaled_height, angle)
                rotated = cv2.warpAffine(scaled, matrix, (rotated_width, rotated_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
                rotated_mask = cv2.warpAffine(scaled_mask, matrix, (rotated_width, rotated_height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_mask = np.where(rotated_mask > 0, 255, 0).astype(np.uint8)
                rotated_center = cv2.transform(scaled_center, matrix).reshape(2)
                rotated_contour = cv2.transform(scaled_contour.reshape(-1, 1, 2), matrix).reshape(-1, 2)
                rotated_height, rotated_width = rotated.shape[:2]
                valid_values = rotated[rotated_mask > 0]
                if valid_values.size and float(np.std(valid_values)) > 1e-6:
                    variants.append(_TemplateVariant(rotated, rotated_mask, rotated_center, rotated_contour, float(angle), float(scale)))
        self._variant_cache_key = params
        self._variant_cache = variants
        return variants

    def _finalize(self, candidates: list[_Candidate]) -> list[TemplateMatch]:
        params = self.parameters
        selected: list[_Candidate] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            def is_distinct(kept: _Candidate) -> bool:
                distance = float(np.hypot(candidate.center_x - kept.center_x, candidate.center_y - kept.center_y))
                too_close = distance < min(candidate.suppression_radius, kept.suppression_radius)
                overlaps = _rotated_iou(candidate.contour, kept.contour) > params.nms_iou_threshold
                return not (too_close or overlaps)

            if all(is_distinct(kept) for kept in selected):
                selected.append(candidate)
                if len(selected) >= params.max_results:
                    break
        selected.sort(key=lambda item: (item.center_y, item.center_x))
        return [
            TemplateMatch(
                index,
                self.model.name,
                candidate.center_x,
                candidate.center_y,
                candidate.angle_deg,
                candidate.score,
                candidate.scale,
                tuple((float(x), float(y)) for x, y in candidate.box),
                tuple((float(x), float(y)) for x, y in candidate.contour),
            )
            for index, candidate in enumerate(selected, 1)
        ]


def draw_template_matches(image: np.ndarray, matches: Iterable[TemplateMatch]) -> np.ndarray:
    canvas = image.copy()
    matches = list(matches)
    for match in matches:
        box = np.round(np.asarray(match.box_points)).astype(np.int32)
        contour = np.round(np.asarray(match.contour_points)).astype(np.int32)
        if len(contour) >= 3:
            cv2.polylines(canvas, [contour], True, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [box], True, (0, 220, 0), 2, cv2.LINE_AA)
        center = (int(round(match.center_x)), int(round(match.center_y)))
        cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        label = f"#{match.object_id} ({match.center_x:.1f},{match.center_y:.1f}) {match.angle_deg:.1f}deg {match.score:.3f}"
        anchor = (max(0, int(box[:, 0].min())), max(18, int(box[:, 1].min()) - 5))
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    header = f"Template matches: {len(matches)}"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 260), 27), (0, 0, 0), -1)
    cv2.putText(canvas, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
