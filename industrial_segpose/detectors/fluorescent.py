"""专用检测器：以颜色优先方式快速检测荧光纺织工件。

The detector intentionally exposes the same ``match`` interface as the
multi-template matcher, so production capture, tracking and result export can
reuse the existing desktop pipeline.  Colour segmentation finds candidate
ROIs; template matching can remain a later classification/refinement stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ..measurement.pick_points import plan_pick_points
from ..template_matching.multi_matcher import MultiTemplateResult, RecognizedObject


@dataclass(frozen=True)
class FluorescentTextileParameters:
    hue_min: int = 20
    hue_max: int = 100
    min_saturation: int = 62
    min_value: int = 55
    min_lab_chroma: float = 18.0
    min_area_ratio: float = 0.003
    # A very large colour component normally means that the cut pieces are
    # still connected to the parent sheet. It must not be reported as one
    # pickable workpiece.
    max_area_ratio: float = 0.12
    # Thin fluorescent cutting outlines can enclose a large empty area.  The
    # foreground must therefore occupy a meaningful fraction of its outer
    # contour before it is considered a physical workpiece.
    minimum_fill_ratio: float = 0.45
    # A detected workpiece is still counted when a hand overlaps it, but it is
    # explicitly blocked from automatic picking.
    occlusion_overlap_ratio: float = 0.025
    minimum_pick_radius_px: float = 10.0
    require_pickable: bool = True
    maximum_results: int = 100
    template_name: str = "Fluorescent_Textile"
    # When cut pieces remain tightly nested, printed colour regions and
    # resampled cutting seams can connect an entire layout into one component.
    # The dense-layout fallback finds thick interior cores and partitions the
    # parent component into repeated row/column cells.  It is deliberately
    # activated only for components rejected by ``max_area_ratio``.
    dense_layout_enabled: bool = True
    dense_min_rows: int = 3
    dense_max_rows: int = 10
    dense_min_columns: int = 3
    dense_max_columns: int = 12

    def validate(self) -> None:
        if not 0 <= self.hue_min <= 179 or not 0 <= self.hue_max <= 179:
            raise ValueError("Hue limits must be in [0, 179]")
        if self.hue_min > self.hue_max:
            raise ValueError("hue_min cannot exceed hue_max")
        if not 0 <= self.min_saturation <= 255 or not 0 <= self.min_value <= 255:
            raise ValueError("HSV thresholds must be in [0, 255]")
        if not 0.0 < self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("Area ratios must satisfy 0 < min < max <= 1")
        if not 0.0 <= self.minimum_fill_ratio <= 1.0:
            raise ValueError("minimum_fill_ratio must be in [0, 1]")
        if not 0.0 <= self.occlusion_overlap_ratio <= 1.0:
            raise ValueError("occlusion_overlap_ratio must be in [0, 1]")
        if self.minimum_pick_radius_px < 0:
            raise ValueError("minimum_pick_radius_px cannot be negative")
        if self.maximum_results < 1:
            raise ValueError("maximum_results must be positive")
        if not 2 <= self.dense_min_rows <= self.dense_max_rows:
            raise ValueError("Dense row limits are invalid")
        if not 2 <= self.dense_min_columns <= self.dense_max_columns:
            raise ValueError("Dense column limits are invalid")


class FluorescentTextileDetector:
    """Segment bright yellow/green textiles and plan suction-safe pick points."""

    def __init__(self, parameters: FluorescentTextileParameters | None = None):
        self.parameters = parameters or FluorescentTextileParameters()
        self.parameters.validate()
        self.max_processing_pixels = 3_000_000
        self.max_processing_edge = 2400

    @property
    def valid_enabled_count(self) -> int:
        return 1

    def match(self, image: np.ndarray) -> MultiTemplateResult:
        if image is None or image.size == 0:
            raise ValueError("Detection image is empty")
        processing, scale = self._prepare_image(image)
        mask, debug = self._segment(processing)
        skin_mask = self._skin_mask(processing)
        objects, dense_debug = self._measure(mask, processing, skin_mask)
        if scale < 0.999:
            objects = tuple(self._restore_object(item, scale) for item in objects)
        objects = tuple(
            RecognizedObject(
                index,
                item.classification_status,
                item.template_id,
                item.template_name,
                item.center_x,
                item.center_y,
                item.angle_deg,
                item.score,
                item.normalized_score,
                item.scale,
                item.color_bgr,
                item.box_points,
                item.contour_points,
                item.candidate_templates,
                item.pick_point_x,
                item.pick_point_y,
                item.safe_radius_px,
                item.orientation_confidence,
                item.auto_pick_allowed,
            )
            for index, item in enumerate(sorted(objects, key=lambda value: (value.center_y, value.center_x)), 1)
        )
        debug["fluorescent_mask"] = mask
        debug["fluorescent_skin_mask"] = skin_mask
        debug.update(dense_debug)
        return MultiTemplateResult(
            objects=objects,
            counts_by_template={self.parameters.template_name: len(objects)},
            ambiguous_count=0,
            template_errors={},
            used_templates=({
                "template_id": "fluorescent-colour-segmentation",
                "template_name": self.parameters.template_name,
                "feature_mode": "fluorescent_colour",
            },),
            source_image_size=(int(image.shape[1]), int(image.shape[0])),
            processing_image_size=(int(processing.shape[1]), int(processing.shape[0])),
            processing_scale=float(scale),
            debug_images=debug,
        )

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        pixel_scale = math.sqrt(self.max_processing_pixels / max(float(width * height), 1.0))
        edge_scale = self.max_processing_edge / max(float(width), float(height), 1.0)
        scale = float(min(1.0, pixel_scale, edge_scale))
        if scale >= 0.999:
            return image, 1.0
        target = (max(5, int(round(width * scale))), max(5, int(round(height * scale))))
        actual = min(target[0] / float(width), target[1] / float(height))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA), float(actual)

    def _segment(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # A small bilateral filter suppresses sensor/grain noise without moving
        # the cutting edge used for pose and pick-point planning.
        filtered = cv2.bilateralFilter(image, 5, 28, 28)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        p = self.parameters
        hsv_mask = cv2.inRange(
            hsv,
            np.array((p.hue_min, p.min_saturation, p.min_value), np.uint8),
            np.array((p.hue_max, 255, 255), np.uint8),
        )
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        chroma_mask = np.where(chroma >= p.min_lab_chroma, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(hsv_mask, chroma_mask)

        short_edge = min(image.shape[:2])
        open_size = max(3, (int(round(short_edge * 0.0015)) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Do not apply a global closing operation here.  On a dense cutting
        # layout the gaps between adjacent pieces are only a few pixels wide;
        # closing joined eight valid workpieces into one oversized component.
        # Interior print holes are filled per accepted component in _measure.
        return mask, {
            "fluorescent_filtered": filtered,
            "fluorescent_hsv_mask": hsv_mask,
            "fluorescent_chroma_mask": chroma_mask,
        }

    @staticmethod
    def _skin_mask(image: np.ndarray) -> np.ndarray:
        """Return a conservative hand/skin mask used only as a safety gate."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
        hsv_skin = cv2.inRange(hsv, (0, 20, 40), (28, 230, 255))
        skin = cv2.bitwise_and(ycrcb_skin, hsv_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)

    def _measure(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        skin_mask: np.ndarray | None = None,
    ) -> tuple[tuple[RecognizedObject, ...], dict[str, np.ndarray]]:
        # Connected-component foreground area is deliberately used for the
        # first filter.  cv2.contourArea would make a thin closed cutting line
        # look like a solid workpiece because it measures the enclosed area.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        image_area = float(mask.shape[0] * mask.shape[1])
        results: list[RecognizedObject] = []
        component_ids = [
            index
            for index in range(1, component_count)
            if float(stats[index, cv2.CC_STAT_AREA]) / max(image_area, 1.0)
            >= self.parameters.min_area_ratio
        ]
        component_areas = [float(stats[index, cv2.CC_STAT_AREA]) for index in component_ids]
        typical_area = float(np.median(component_areas)) if len(component_areas) >= 3 else 0.0
        component_masks: list[tuple[np.ndarray, bool]] = []
        dense_debug: dict[str, np.ndarray] = {}
        for component_id in component_ids:
            raw_component = np.where(labels == component_id, 255, 0).astype(np.uint8)
            foreground_area = float(stats[component_id, cv2.CC_STAT_AREA])
            area_ratio = foreground_area / max(image_area, 1.0)
            if self.parameters.dense_layout_enabled and area_ratio > self.parameters.max_area_ratio:
                parts, preview = self._split_dense_layout_component(raw_component, image)
                if parts:
                    component_masks.extend((part, True) for part in parts)
                    dense_debug["fluorescent_dense_layout"] = preview
                    continue
            size_multiple = foreground_area / max(typical_area, 1.0) if typical_area else 1.0
            split_count = int(round(size_multiple))
            if typical_area and 1.65 <= size_multiple <= 3.40 and 2 <= split_count <= 3:
                parts = self._split_spatial_component(raw_component, split_count)
                if len(parts) == split_count and all(
                    cv2.countNonZero(part) >= 0.48 * typical_area for part in parts
                ):
                    component_masks.extend((part, False) for part in parts)
                    continue
            component_masks.append((raw_component, False))
        component_masks.sort(key=lambda item: cv2.countNonZero(item[0]), reverse=True)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        short_edge = min(mask.shape[:2])
        safety_size = max(9, (int(round(short_edge * 0.012)) | 1))
        safety_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety_size, safety_size))
        for raw_component, dense_layout in component_masks:
            foreground_area = float(cv2.countNonZero(raw_component))
            ratio = foreground_area / max(image_area, 1.0)
            if ratio < self.parameters.min_area_ratio or ratio > self.parameters.max_area_ratio:
                continue
            contours, _ = cv2.findContours(raw_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            fill_ratio = foreground_area / max(contour_area, 1.0)
            if fill_ratio < self.parameters.minimum_fill_ratio:
                continue
            component = np.zeros_like(mask)
            cv2.drawContours(component, [contour], -1, 255, -1)
            points = plan_pick_points(
                component,
                maximum_points=3,
                minimum_safe_radius_px=self.parameters.minimum_pick_radius_px,
            )
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
            angle, orientation_confidence = self._directed_angle(component, center_x, center_y)
            rect = cv2.minAreaRect(contour)
            box = tuple(tuple(map(float, point)) for point in cv2.boxPoints(rect))
            epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            contour_points = tuple(tuple(map(float, point)) for point in simplified)
            x, y, width, height = cv2.boundingRect(contour)
            touches_border = x <= 1 or y <= 1 or x + width >= mask.shape[1] - 1 or y + height >= mask.shape[0] - 1
            occlusion_ratio = 0.0
            if skin_mask is not None:
                safety_region = cv2.dilate(component, safety_kernel)
                occlusion_pixels = cv2.countNonZero(cv2.bitwise_and(safety_region, skin_mask))
                occlusion_ratio = occlusion_pixels / max(foreground_area, 1.0)
            occluded = occlusion_ratio >= self.parameters.occlusion_overlap_ratio
            primary = points[0] if points else None
            if self.parameters.require_pickable and primary is None:
                continue
            mean_saturation = float(cv2.mean(saturation, mask=raw_component)[0])
            score = float(np.clip(0.55 + 0.45 * mean_saturation / 255.0, 0.0, 1.0))
            if occluded:
                score *= float(np.clip(1.0 - occlusion_ratio, 0.35, 0.9))
            results.append(RecognizedObject(
                object_id=len(results) + 1,
                classification_status="occluded" if occluded else (
                    "dense_layout" if dense_layout else "segmented"
                ),
                template_id=(
                    "fluorescent-dense-layout"
                    if dense_layout else "fluorescent-colour-segmentation"
                ),
                template_name=self.parameters.template_name,
                center_x=center_x,
                center_y=center_y,
                angle_deg=angle,
                score=score,
                normalized_score=score,
                scale=1.0,
                color_bgr=(40, 220, 255),
                box_points=box,
                contour_points=contour_points,
                candidate_templates=(),
                pick_point_x=primary.x if primary else None,
                pick_point_y=primary.y if primary else None,
                safe_radius_px=primary.safe_radius_px if primary else None,
                orientation_confidence=orientation_confidence * (0.35 if occluded else 1.0),
                auto_pick_allowed=bool(primary and not touches_border and not occluded),
            ))
            if len(results) >= self.parameters.maximum_results:
                break
        return tuple(results), dense_debug

    def _split_dense_layout_component(
        self,
        component: np.ndarray,
        image: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """Partition a repeated, tightly nested cutting layout.

        A single colour component is not sufficient evidence for one physical
        object in a nesting layout.  Thick distance-transform maxima provide
        scale-aware interior evidence while the regular row/column structure
        prevents the two lobes of an irregular piece from being double-counted.
        Thin empty cutting outlines have no sufficiently deep interior and are
        therefore excluded.
        """
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        maximum_distance = float(distance.max())
        minimum_core = max(
            self.parameters.minimum_pick_radius_px * 1.15,
            maximum_distance * 0.18,
            5.0,
        )
        short_edge = min(component.shape[:2])
        peak_size = max(15, int(round(short_edge * 0.025)) | 1)
        local_maximum = distance >= cv2.dilate(
            distance,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size)),
        ) - 1e-4
        peak_mask = np.where(local_maximum & (distance >= minimum_core), 255, 0).astype(np.uint8)
        peak_count, peak_labels, peak_stats, peak_centers = cv2.connectedComponentsWithStats(
            peak_mask, connectivity=8
        )
        peaks: list[tuple[float, float, float]] = []
        for index in range(1, peak_count):
            if peak_stats[index, cv2.CC_STAT_AREA] < 1:
                continue
            locations = np.where(peak_labels == index)
            values = distance[locations]
            best = int(np.argmax(values))
            peaks.append((
                float(locations[1][best]),
                float(locations[0][best]),
                float(values[best]),
            ))
        if len(peaks) < self.parameters.dense_min_rows * self.parameters.dense_min_columns:
            return (), image.copy()

        radii = np.asarray([item[2] for item in peaks], np.float32)
        row_tolerance = max(18.0, float(np.median(radii)) * 0.78)
        ordered = sorted(peaks, key=lambda item: item[1])
        rows: list[list[tuple[float, float, float]]] = []
        for peak in ordered:
            if not rows or peak[1] - max(item[1] for item in rows[-1]) > row_tolerance:
                rows.append([peak])
            else:
                rows[-1].append(peak)
        rows = [row for row in rows if len(row) >= self.parameters.dense_min_columns]
        if not self.parameters.dense_min_rows <= len(rows) <= self.parameters.dense_max_rows:
            return (), image.copy()

        column_count = self._infer_dense_column_count(rows, float(np.median(radii)))
        if column_count is None:
            return (), image.copy()

        row_centers = [float(np.median([item[1] for item in row])) for row in rows]
        row_boundaries = [0]
        row_boundaries.extend(
            int(round((row_centers[index - 1] + row_centers[index]) * 0.5))
            for index in range(1, len(row_centers))
        )
        if len(row_centers) > 1:
            tail = row_centers[-1] + (row_centers[-1] - row_centers[-2]) * 0.5
        else:
            tail = row_centers[-1] + row_tolerance
        row_boundaries.append(min(component.shape[0], int(round(tail))))

        preview = image.copy()
        parts: list[np.ndarray] = []
        for row_index, row_center in enumerate(row_centers):
            y0 = max(0, row_boundaries[row_index])
            y1 = min(component.shape[0], row_boundaries[row_index + 1])
            search_half = max(12, int(round((y1 - y0) * 0.32)))
            band_y0 = max(y0, int(round(row_center)) - search_half)
            band_y1 = min(y1, int(round(row_center)) + search_half + 1)
            band = distance[band_y0:band_y1]
            projection = np.max(band, axis=0) >= max(3.0, minimum_core * 0.42)
            indexes = np.flatnonzero(projection)
            if indexes.size == 0:
                continue
            runs = np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
            significant = [run for run in runs if len(run) >= peak_size]
            if not significant:
                continue
            x_min = int(min(run[0] for run in significant))
            x_max = int(max(run[-1] for run in significant)) + 1
            if x_max - x_min < column_count * peak_size:
                continue
            x_boundaries = np.linspace(x_min, x_max, column_count + 1).round().astype(int)
            row_peak = float(np.max(band))
            for column_index in range(column_count):
                x0 = int(x_boundaries[column_index])
                x1 = int(x_boundaries[column_index + 1])
                cell_distance = distance[y0:y1, x0:x1]
                if cell_distance.size == 0:
                    continue
                _, safe_radius, _, location = cv2.minMaxLoc(cell_distance)
                confirmation = max(minimum_core, row_peak * 0.34)
                if safe_radius < confirmation:
                    cv2.rectangle(preview, (x0, y0), (x1, y1), (90, 90, 90), 1)
                    continue
                part = np.zeros_like(component)
                part[y0:y1, x0:x1] = component[y0:y1, x0:x1]
                if cv2.countNonZero(part) < self.parameters.min_area_ratio * component.size:
                    continue
                pick_x = x0 + int(location[0])
                pick_y = y0 + int(location[1])
                cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 150, 30), 2)
                cv2.circle(preview, (pick_x, pick_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    str(len(parts) + 1),
                    (x0 + 4, max(16, y0 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                parts.append(part)
        if len(parts) < self.parameters.dense_min_rows * self.parameters.dense_min_columns:
            return (), preview
        return tuple(parts), preview

    def _infer_dense_column_count(
        self,
        rows: list[list[tuple[float, float, float]]],
        typical_radius: float,
    ) -> int | None:
        """Infer the repeated column count without hard-coding a factory layout."""
        candidates: list[tuple[int, float]] = []
        for count in range(self.parameters.dense_min_columns, self.parameters.dense_max_columns + 1):
            variations: list[float] = []
            minimum_gaps: list[float] = []
            for row in rows[-min(3, len(rows)):]:
                if len(row) < count:
                    continue
                samples = np.asarray([[item[0]] for item in row], np.float32)
                cv2.setRNGSeed(230)
                _, _, centers = cv2.kmeans(
                    samples,
                    count,
                    None,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
                    8,
                    cv2.KMEANS_PP_CENTERS,
                )
                gaps = np.diff(np.sort(centers[:, 0]))
                if gaps.size == 0 or float(np.mean(gaps)) <= 1e-6:
                    continue
                variations.append(float(np.std(gaps) / np.mean(gaps)))
                minimum_gaps.append(float(np.min(gaps)))
            if len(variations) < min(2, len(rows)):
                continue
            mean_variation = float(np.mean(variations))
            gap_is_physical = float(np.median(minimum_gaps)) >= max(20.0, typical_radius * 1.55)
            if mean_variation <= 0.19 and gap_is_physical:
                candidates.append((count, mean_variation))
        if not candidates:
            return None
        # The largest still-regular count avoids selecting a lower harmonic
        # (for example three columns for a seven-column nesting layout).
        return max(candidates, key=lambda item: item[0])[0]

    @staticmethod
    def _split_spatial_component(component: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
        """Split a merged repeated-layout component into spatial instances.

        Dense nesting layouts occasionally contain one-pixel bridges after
        camera resampling.  Area statistics reveal that such a component is
        approximately two or three normal pieces.  Spatial k-means removes the
        accidental bridge without using a large erosion that would damage the
        real cutting contour.
        """
        ys, xs = np.where(component > 0)
        if len(xs) < count * 20:
            return ()
        samples = np.column_stack((xs, ys)).astype(np.float32)
        cv2.setRNGSeed(230)
        _, labels, _ = cv2.kmeans(
            samples,
            count,
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25),
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        parts: list[np.ndarray] = []
        flat_labels = labels.reshape(-1)
        for index in range(count):
            selected = flat_labels == index
            part = np.zeros_like(component)
            part[ys[selected], xs[selected]] = 255
            parts.append(part)
        return tuple(parts)

    @staticmethod
    def _directed_angle(mask: np.ndarray, center_x: float, center_y: float) -> tuple[float, float]:
        ys, xs = np.where(mask > 0)
        if len(xs) < 5:
            return 0.0, 0.0
        samples = np.column_stack((xs - center_x, ys - center_y)).astype(np.float32)
        covariance = np.cov(samples, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        major = vectors[:, order[0]]
        major_value = float(values[order[0]])
        minor_value = max(float(values[order[1]]), 1e-9)
        projections = samples @ major
        negative = abs(float(projections.min()))
        positive = abs(float(projections.max()))
        # Resolve the 180-degree ambiguity deterministically toward the longer
        # extent. A later template/print classifier can provide semantic front.
        extent_confidence = abs(positive - negative) / max(positive + negative, 1e-9)
        if negative > positive:
            major = -major
        angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 360.0
        shape_confidence = float(np.clip(1.0 - minor_value / max(major_value, 1e-9), 0.0, 1.0))
        return angle, float(np.clip(0.65 * shape_confidence + 0.35 * extent_confidence, 0.0, 1.0))

    @staticmethod
    def _restore_object(item: RecognizedObject, processing_scale: float) -> RecognizedObject:
        inverse = 1.0 / processing_scale
        return RecognizedObject(
            item.object_id, item.classification_status, item.template_id, item.template_name,
            item.center_x * inverse, item.center_y * inverse, item.angle_deg, item.score,
            item.normalized_score, item.scale * inverse, item.color_bgr,
            tuple((x * inverse, y * inverse) for x, y in item.box_points),
            tuple((x * inverse, y * inverse) for x, y in item.contour_points),
            item.candidate_templates,
            item.pick_point_x * inverse if item.pick_point_x is not None else None,
            item.pick_point_y * inverse if item.pick_point_y is not None else None,
            item.safe_radius_px * inverse if item.safe_radius_px is not None else None,
            item.orientation_confidence,
            item.auto_pick_allowed,
        )
