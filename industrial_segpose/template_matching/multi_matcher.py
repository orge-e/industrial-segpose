"""Joint matching and conflict resolution across a persistent template library."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import cv2
import numpy as np

from .library import LoadedTemplateEntry, TemplateLibrary
from .matcher import TemplateMatch, TemplateMatcher, _rotated_iou


@dataclass(frozen=True)
class TemplateCandidateScore:
    template_id: str
    template_name: str
    score: float
    normalized_score: float


@dataclass(frozen=True)
class RecognizedObject:
    object_id: int
    classification_status: str
    template_id: str | None
    template_name: str
    center_x: float
    center_y: float
    angle_deg: float
    score: float
    normalized_score: float
    scale: float
    color_bgr: tuple[int, int, int]
    box_points: tuple[tuple[float, float], ...]
    contour_points: tuple[tuple[float, float], ...]
    candidate_templates: tuple[TemplateCandidateScore, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["color_bgr"] = list(self.color_bgr)
        payload["box_points"] = [list(point) for point in self.box_points]
        payload["contour_points"] = [list(point) for point in self.contour_points]
        payload["candidate_templates"] = [asdict(item) for item in self.candidate_templates]
        return payload


@dataclass(frozen=True)
class MultiTemplateResult:
    objects: tuple[RecognizedObject, ...]
    counts_by_template: dict[str, int]
    ambiguous_count: int
    template_errors: dict[str, str]
    used_templates: tuple[dict, ...]
    source_image_size: tuple[int, int] | None = None
    processing_image_size: tuple[int, int] | None = None
    processing_scale: float = 1.0

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def to_dict(self) -> dict:
        return {
            "object_count": self.object_count,
            "counts_by_template": dict(self.counts_by_template),
            "ambiguous_count": self.ambiguous_count,
            "template_errors": dict(self.template_errors),
            "templates": list(self.used_templates),
            "preprocessing": {
                "source_image_size": list(self.source_image_size) if self.source_image_size else None,
                "processing_image_size": list(self.processing_image_size) if self.processing_image_size else None,
                "processing_scale": self.processing_scale,
            },
            "objects": [item.to_dict() for item in self.objects],
        }


@dataclass(frozen=True)
class _ScoredMatch:
    loaded: LoadedTemplateEntry
    match: TemplateMatch
    normalized_score: float


class MultiTemplateMatcher:
    DEFAULT_MAX_PROCESSING_PIXELS = 3_000_000
    DEFAULT_MAX_PROCESSING_EDGE = 2400

    def __init__(
        self,
        loaded_entries: Iterable[LoadedTemplateEntry],
        ambiguity_margin: float = 0.08,
        cross_template_iou: float = 0.25,
    ):
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError("Ambiguity margin must be in [0, 1]")
        if not 0.0 <= cross_template_iou <= 1.0:
            raise ValueError("Cross-template IoU must be in [0, 1]")
        self.loaded_entries = list(loaded_entries)
        self.ambiguity_margin = float(ambiguity_margin)
        self.cross_template_iou = float(cross_template_iou)
        self.max_processing_pixels = self.DEFAULT_MAX_PROCESSING_PIXELS
        self.max_processing_edge = self.DEFAULT_MAX_PROCESSING_EDGE
        self._matchers = {
            item.entry.template_id: TemplateMatcher(item.model, item.entry.parameters)
            for item in self.loaded_entries
            if item.entry.enabled and item.valid
        }

    @classmethod
    def from_library(cls, library: TemplateLibrary) -> "MultiTemplateMatcher":
        return cls(library.load_entries(), library.ambiguity_margin, library.cross_template_iou)

    @property
    def valid_enabled_count(self) -> int:
        return len(self._matchers)

    def match(self, image: np.ndarray) -> MultiTemplateResult:
        if image is None or image.size == 0:
            raise ValueError("Detection image is empty")
        if not self._matchers:
            raise ValueError("No valid enabled templates are available")
        processing_image, processing_scale = self._prepare_detection_image(image)
        scored: list[_ScoredMatch] = []
        errors: dict[str, str] = {
            item.entry.name: item.error
            for item in self.loaded_entries
            if item.entry.enabled and not item.valid and item.error
        }
        used_templates: list[dict] = []
        for loaded in self.loaded_entries:
            entry = loaded.entry
            if not entry.enabled or not loaded.valid:
                continue
            used_templates.append({
                "template_id": entry.template_id,
                "template_name": entry.name,
                "template_file": entry.template_file,
                "parameters": asdict(entry.parameters),
            })
            try:
                matcher = self._matchers[entry.template_id]
                if processing_scale < 1.0:
                    parameters = replace(
                        entry.parameters,
                        scale_min=entry.parameters.scale_min * processing_scale,
                        scale_max=entry.parameters.scale_max * processing_scale,
                        scale_step=entry.parameters.scale_step * processing_scale,
                    )
                    matcher = TemplateMatcher(loaded.model, parameters)
                matches = matcher.match(processing_image)
                if processing_scale < 1.0:
                    matches = [self._restore_source_coordinates(item, processing_scale) for item in matches]
            except Exception as exc:
                errors[entry.name] = str(exc)
                continue
            threshold = float(entry.parameters.score_threshold)
            denominator = max(1.0 - threshold, 1e-9)
            for match in matches:
                normalized = float(np.clip((match.score - threshold) / denominator, 0.0, 1.0))
                scored.append(_ScoredMatch(loaded, match, normalized))
        objects = self._resolve(scored)
        counts = {entry["template_name"]: 0 for entry in used_templates}
        for item in objects:
            if item.classification_status == "confirmed":
                counts[item.template_name] = counts.get(item.template_name, 0) + 1
        ambiguous = sum(item.classification_status == "ambiguous" for item in objects)
        return MultiTemplateResult(
            tuple(objects),
            counts,
            ambiguous,
            errors,
            tuple(used_templates),
            (int(image.shape[1]), int(image.shape[0])),
            (int(processing_image.shape[1]), int(processing_image.shape[0])),
            float(processing_scale),
        )

    def _prepare_detection_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        pixel_scale = np.sqrt(self.max_processing_pixels / max(float(width * height), 1.0))
        edge_scale = self.max_processing_edge / max(float(width), float(height), 1.0)
        scale = float(min(1.0, pixel_scale, edge_scale))
        if scale >= 0.999:
            return image, 1.0
        target_width = max(5, int(round(width * scale)))
        target_height = max(5, int(round(height * scale)))
        actual_scale = min(target_width / float(width), target_height / float(height))
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, float(actual_scale)

    @staticmethod
    def _restore_source_coordinates(match: TemplateMatch, processing_scale: float) -> TemplateMatch:
        inverse = 1.0 / processing_scale
        return TemplateMatch(
            match.object_id,
            match.template_name,
            match.center_x * inverse,
            match.center_y * inverse,
            match.angle_deg,
            match.score,
            match.scale * inverse,
            tuple((x * inverse, y * inverse) for x, y in match.box_points),
            tuple((x * inverse, y * inverse) for x, y in match.contour_points),
        )

    def _resolve(self, scored: list[_ScoredMatch]) -> list[RecognizedObject]:
        remaining = sorted(scored, key=lambda item: item.normalized_score, reverse=True)
        resolved: list[RecognizedObject] = []
        while remaining:
            best = remaining.pop(0)
            conflicts = [item for item in remaining if self._conflicts(best.match, item.match)]
            conflict_ids = {id(item) for item in conflicts}
            remaining = [item for item in remaining if id(item) not in conflict_ids]
            group = [best, *conflicts]
            by_template: dict[str, _ScoredMatch] = {}
            for item in group:
                key = item.loaded.entry.template_id
                if key not in by_template or item.normalized_score > by_template[key].normalized_score:
                    by_template[key] = item
            alternatives = sorted(by_template.values(), key=lambda item: item.normalized_score, reverse=True)
            winner = alternatives[0]
            runner = alternatives[1] if len(alternatives) > 1 else None
            ambiguous = runner is not None and winner.normalized_score - runner.normalized_score < self.ambiguity_margin
            candidates = tuple(
                TemplateCandidateScore(item.loaded.entry.template_id, item.loaded.entry.name, item.match.score, item.normalized_score)
                for item in alternatives
            )
            entry = winner.loaded.entry
            match = winner.match
            resolved.append(RecognizedObject(
                object_id=0,
                classification_status="ambiguous" if ambiguous else "confirmed",
                template_id=None if ambiguous else entry.template_id,
                template_name="待确认" if ambiguous else entry.name,
                center_x=match.center_x,
                center_y=match.center_y,
                angle_deg=match.angle_deg,
                score=match.score,
                normalized_score=winner.normalized_score,
                scale=match.scale,
                color_bgr=(0, 215, 255) if ambiguous else entry.color_bgr,
                box_points=match.box_points,
                contour_points=match.contour_points,
                candidate_templates=candidates,
            ))
        resolved.sort(key=lambda item: (item.center_y, item.center_x))
        return [
            RecognizedObject(index, item.classification_status, item.template_id, item.template_name, item.center_x, item.center_y, item.angle_deg, item.score, item.normalized_score, item.scale, item.color_bgr, item.box_points, item.contour_points, item.candidate_templates)
            for index, item in enumerate(resolved, 1)
        ]

    def _conflicts(self, first: TemplateMatch, second: TemplateMatch) -> bool:
        first_contour = cv2.convexHull(np.asarray(first.contour_points, np.float32)).reshape(-1, 2)
        second_contour = cv2.convexHull(np.asarray(second.contour_points, np.float32)).reshape(-1, 2)
        if _rotated_iou(first_contour, second_contour) > self.cross_template_iou:
            return True
        first_short = self._short_side(first.box_points)
        second_short = self._short_side(second.box_points)
        distance = float(np.hypot(first.center_x - second.center_x, first.center_y - second.center_y))
        return distance < 0.6 * min(first_short, second_short)

    @staticmethod
    def _short_side(points: tuple[tuple[float, float], ...]) -> float:
        box = np.asarray(points, np.float32)
        if len(box) < 4:
            return 0.0
        lengths = [float(np.linalg.norm(box[(index + 1) % len(box)] - box[index])) for index in range(len(box))]
        return min(lengths) if lengths else 0.0


def draw_multi_template_matches(image: np.ndarray, result: MultiTemplateResult) -> np.ndarray:
    canvas = image.copy()
    longest_edge = float(max(canvas.shape[:2]))
    visual_scale = max(1.0, longest_edge / 1000.0)
    contour_thickness = max(3, int(round(2.2 * visual_scale)))
    box_thickness = max(2, int(round(1.3 * visual_scale)))
    marker_size = max(24, int(round(20 * visual_scale)))
    marker_thickness = max(3, int(round(1.8 * visual_scale)))
    font_scale = min(2.4, max(0.78, longest_edge / 1500.0))
    text_thickness = max(2, int(round(1.5 * visual_scale)))
    text_margin = max(5, int(round(4 * visual_scale)))
    summary = f"Total: {result.object_count}  Ambiguous: {result.ambiguous_count}"
    summary_scale = min(2.4, max(0.82, longest_edge / 1600.0))
    summary_thickness = max(2, int(round(1.4 * visual_scale)))
    summary_size, summary_baseline = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, summary_scale, summary_thickness)
    header_height = summary_size[1] + summary_baseline + 2 * text_margin
    occupied = [(0, 0, min(canvas.shape[1] - 1, summary_size[0] + 3 * text_margin), header_height)]

    def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for item in result.objects:
        color = tuple(map(int, item.color_bgr))
        contour = np.round(np.asarray(item.contour_points)).astype(np.int32)
        box = np.round(np.asarray(item.box_points)).astype(np.int32)
        if len(contour) >= 3:
            cv2.polylines(canvas, [contour], True, color, contour_thickness, cv2.LINE_AA)
        cv2.polylines(canvas, [box], True, color, box_thickness, cv2.LINE_AA)
        center = (int(round(item.center_x)), int(round(item.center_y)))
        cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, marker_size, marker_thickness)
        label = f"#{item.object_id} {item.template_name} {item.angle_deg:.1f}deg {item.score:.3f}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        min_x, max_x = int(box[:, 0].min()), int(box[:, 0].max())
        min_y, max_y = int(box[:, 1].min()), int(box[:, 1].max())
        candidates = [
            (min_x, min_y - text_margin),
            (min_x, max_y + text_size[1] + 2 * text_margin),
            (max_x + 2 * text_margin, (min_y + max_y + text_size[1]) // 2),
            (min_x - text_size[0] - 2 * text_margin, (min_y + max_y + text_size[1]) // 2),
        ]
        placements: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for proposed_x, proposed_y in candidates:
            anchor_x = max(text_margin, min(proposed_x, canvas.shape[1] - text_size[0] - text_margin))
            anchor_y = max(text_size[1] + text_margin, min(proposed_y, canvas.shape[0] - baseline - text_margin))
            rect = (
                max(0, anchor_x - text_margin),
                max(0, anchor_y - text_size[1] - text_margin),
                min(canvas.shape[1] - 1, anchor_x + text_size[0] + text_margin),
                min(canvas.shape[0] - 1, anchor_y + baseline + text_margin),
            )
            placements.append((anchor_x, anchor_y, rect))
        anchor_x, anchor_y, label_rect = min(
            placements,
            key=lambda placement: sum(overlap_area(placement[2], previous) for previous in occupied),
        )
        occupied.append(label_rect)
        cv2.rectangle(
            canvas,
            (label_rect[0], label_rect[1]),
            (label_rect[2], label_rect[3]),
            (18, 24, 34),
            -1,
        )
        cv2.putText(canvas, label, (anchor_x, anchor_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], summary_size[0] + 3 * text_margin), header_height), (18, 24, 34), -1)
    cv2.putText(canvas, summary, (text_margin, text_margin + summary_size[1]), cv2.FONT_HERSHEY_SIMPLEX, summary_scale, (255, 255, 255), summary_thickness, cv2.LINE_AA)
    return canvas
