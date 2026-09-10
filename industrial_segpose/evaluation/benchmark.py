"""算法评估：为多模板匹配器生成确定性的合成压力测试。

The benchmark is intentionally based on the real template library.  It does
not pretend to replace factory acceptance images; it provides a repeatable
regression gate while those images are unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

from ..io.image_reader import write_image
from ..template_matching import (
    MultiTemplateMatcher,
    MultiTemplateResult,
    TemplateLibrary,
    draw_multi_template_matches,
)
from ..template_matching.library import LoadedTemplateEntry
from ..template_matching.matcher import _rotate_expanded


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: int
    template_id: str
    template_name: str
    center_x: float
    center_y: float
    angle_deg: float
    scale: float
    box_points: tuple[tuple[float, float], ...]
    contour_points: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["box_points"] = [list(point) for point in self.box_points]
        value["contour_points"] = [list(point) for point in self.contour_points]
        return value


@dataclass(frozen=True)
class SyntheticStressConfig:
    # Large enough for the current high-resolution textile templates at their
    # configured minimum scale, including arbitrary 360-degree rotation.
    width: int = 2560
    height: int = 1600
    min_objects: int = 1
    max_objects: int = 4
    profile: str = "balanced"
    allow_overlap: bool = False
    placement_attempts: int = 80

    def validate(self) -> None:
        if self.width < 160 or self.height < 120:
            raise ValueError("Synthetic scene must be at least 160 x 120 pixels")
        if self.min_objects < 1 or self.max_objects < self.min_objects:
            raise ValueError("Invalid synthetic object count range")
        if self.profile not in {"clean", "balanced", "harsh"}:
            raise ValueError("Stress profile must be clean, balanced or harsh")
        if self.placement_attempts < 1:
            raise ValueError("placement_attempts must be positive")


@dataclass(frozen=True)
class EvaluationThresholds:
    center_tolerance_px: float = 60.0
    angle_tolerance_deg: float = 8.0

    def validate(self) -> None:
        if self.center_tolerance_px <= 0 or not 0 < self.angle_tolerance_deg <= 180:
            raise ValueError("Evaluation tolerances must be positive")


@dataclass(frozen=True)
class ObjectEvaluation:
    ground_truth_id: int | None
    predicted_id: int | None
    expected_template: str | None
    predicted_template: str | None
    center_error_px: float | None
    angle_error_deg: float | None
    classification_correct: bool
    center_pass: bool
    angle_pass: bool


@dataclass
class SceneEvaluation:
    scene_id: str
    ground_truth_count: int
    prediction_count: int
    matched_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    classification_correct: int
    ambiguous_count: int
    elapsed_ms: float
    center_errors_px: list[float] = field(default_factory=list)
    angle_errors_deg: list[float] = field(default_factory=list)
    objects: list[ObjectEvaluation] = field(default_factory=list)

    @property
    def count_exact(self) -> bool:
        return self.ground_truth_count == self.prediction_count

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["count_exact"] = self.count_exact
        return payload


@dataclass(frozen=True)
class BenchmarkSummary:
    scene_count: int
    ground_truth_count: int
    prediction_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    exact_count_scenes: int
    ambiguous_count: int
    precision: float
    recall: float
    f1: float
    classification_accuracy: float
    mean_center_error_px: float | None
    p95_center_error_px: float | None
    mean_angle_error_deg: float | None
    p95_angle_error_deg: float | None
    mean_elapsed_ms: float


def circular_angle_error(first: float, second: float) -> float:
    """Return the smallest unsigned difference between two 360-degree angles."""

    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    items = list(values)
    return float(np.percentile(items, percentile)) if items else None


def _boxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class SyntheticSceneGenerator:
    """Create deterministic scenes from enabled templates and exact masks."""

    def __init__(
        self,
        library: TemplateLibrary,
        config: SyntheticStressConfig,
        seed: int = 42,
        loaded_entries: list[LoadedTemplateEntry] | None = None,
    ):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(seed)
        source = loaded_entries if loaded_entries is not None else library.load_entries()
        self.templates = [
            item for item in source
            if item.valid and (loaded_entries is not None or item.entry.enabled)
        ]
        if not self.templates:
            raise ValueError("No valid enabled templates are available for the benchmark")

    def _background(self) -> np.ndarray:
        cfg = self.config
        foreground_lightness = []
        for item in self.templates:
            assert item.model is not None
            lab = cv2.cvtColor(item.model.image, cv2.COLOR_BGR2LAB)
            foreground_lightness.append(float(np.median(lab[:, :, 0][item.model.mask > 0])))
        # Prefer a neutral support surface on the opposite side of the target
        # lightness.  Stress profiles then perturb this reference illumination.
        reference_base = 205.0 if float(np.median(foreground_lightness)) < 128.0 else 42.0
        if cfg.profile == "clean":
            return np.full((cfg.height, cfg.width, 3), round(reference_base), dtype=np.uint8)
        y, x = np.mgrid[0 : cfg.height, 0 : cfg.width]
        base = float(np.clip(reference_base + self.rng.uniform(-35, 35), 25, 225))
        gx = self.rng.uniform(-35, 35) * x / max(cfg.width - 1, 1)
        gy = self.rng.uniform(-28, 28) * y / max(cfg.height - 1, 1)
        illumination = base + gx + gy
        noise_sigma = 4.0 if cfg.profile == "balanced" else 10.0
        texture = self.rng.normal(0.0, noise_sigma, (cfg.height, cfg.width))
        gray = np.clip(illumination + texture, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if cfg.profile == "harsh":
            for _ in range(3):
                center = (int(self.rng.integers(0, cfg.width)), int(self.rng.integers(0, cfg.height)))
                radius = int(self.rng.integers(max(20, cfg.width // 12), max(30, cfg.width // 4)))
                shade = int(self.rng.integers(-30, 31))
                overlay = np.zeros((cfg.height, cfg.width), np.int16)
                cv2.circle(overlay, center, radius, shade, -1)
                image = np.clip(image.astype(np.int16) + overlay[:, :, None], 0, 255).astype(np.uint8)
        return image

    def _scale_for(self, item: LoadedTemplateEntry) -> float:
        params = item.entry.parameters
        low, high = params.scale_min, params.scale_max
        assert item.model is not None
        # A rotated rectangle can expand to its diagonal in both dimensions.
        # Using the diagonal keeps every randomly selected angle placeable.
        diagonal = float(np.hypot(item.model.image.shape[1], item.model.image.shape[0]))
        fit = min(self.config.width - 20, self.config.height - 20) / diagonal * 0.82
        high = min(high, fit)
        if high < low:
            raise ValueError(
                f"Template '{item.entry.name}' cannot fit the synthetic scene within its scale range; "
                "increase --width/--height or lower the template scale minimum"
            )
        return float(self.rng.uniform(low, high)) if high > low else float(low)

    def _variant(self, item: LoadedTemplateEntry) -> tuple[np.ndarray, np.ndarray, float, float]:
        assert item.model is not None
        params = item.entry.parameters
        scale = self._scale_for(item)
        angle = float(self.rng.uniform(params.angle_min, params.angle_max))
        size = (
            max(5, int(round(item.model.image.shape[1] * scale))),
            max(5, int(round(item.model.image.shape[0] * scale))),
        )
        image = cv2.resize(item.model.image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        mask = cv2.resize(item.model.mask, size, interpolation=cv2.INTER_NEAREST)
        image = _rotate_expanded(image, angle, 0)
        mask = _rotate_expanded(mask, angle, 0, cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        return image, mask, angle, scale

    def generate(self, scene_index: int) -> tuple[np.ndarray, list[GroundTruthObject]]:
        image = self._background()
        count = int(self.rng.integers(self.config.min_objects, self.config.max_objects + 1))
        occupied: list[tuple[int, int, int, int]] = []
        truth: list[GroundTruthObject] = []
        order = self.rng.integers(0, len(self.templates), size=count)
        for raw_index in order:
            item = self.templates[int(raw_index)]
            patch, mask, angle, scale = self._variant(item)
            height, width = mask.shape
            placement = None
            for _ in range(self.config.placement_attempts):
                left = int(self.rng.integers(5, self.config.width - width - 4))
                top = int(self.rng.integers(5, self.config.height - height - 4))
                candidate = (left, top, width, height)
                if self.config.allow_overlap or not any(_boxes_overlap(candidate, old) for old in occupied):
                    placement = candidate
                    break
            if placement is None:
                continue
            left, top, _, _ = placement
            roi = image[top : top + height, left : left + width]
            alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
            roi[:] = np.clip(patch.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha), 0, 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            contour[:, 0] += left
            contour[:, 1] += top
            contour = cv2.approxPolyDP(contour.reshape(-1, 1, 2), 1.0, True).reshape(-1, 2)
            moments = cv2.moments(mask, binaryImage=True)
            center_x = left + moments["m10"] / moments["m00"]
            center_y = top + moments["m01"] / moments["m00"]
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            truth.append(
                GroundTruthObject(
                    object_id=len(truth) + 1,
                    template_id=item.entry.template_id,
                    template_name=item.entry.name,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    angle_deg=angle,
                    scale=scale,
                    box_points=tuple(map(tuple, box.astype(float))),
                    contour_points=tuple(map(tuple, contour.astype(float))),
                )
            )
            occupied.append(placement)
        if not truth:
            raise RuntimeError(f"Synthetic scene {scene_index} contains no placeable templates")
        image = self._apply_camera_stress(image)
        return image, truth

    def _apply_camera_stress(self, image: np.ndarray) -> np.ndarray:
        if self.config.profile == "clean":
            return image
        contrast = self.rng.uniform(0.82, 1.18) if self.config.profile == "balanced" else self.rng.uniform(0.62, 1.38)
        brightness = self.rng.uniform(-12, 12) if self.config.profile == "balanced" else self.rng.uniform(-28, 28)
        result = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        if self.config.profile == "harsh" or self.rng.random() < 0.45:
            sigma = float(self.rng.uniform(0.35, 1.1 if self.config.profile == "balanced" else 1.8))
            result = cv2.GaussianBlur(result, (0, 0), sigma)
        if self.config.profile == "harsh":
            noise = self.rng.normal(0, 5.0, result.shape).astype(np.float32)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return result


def evaluate_scene(
    scene_id: str,
    truth: list[GroundTruthObject],
    result: MultiTemplateResult,
    elapsed_ms: float,
    thresholds: EvaluationThresholds,
) -> SceneEvaluation:
    thresholds.validate()
    candidates: list[tuple[float, int, int]] = []
    for truth_index, expected in enumerate(truth):
        for prediction_index, predicted in enumerate(result.objects):
            distance = float(np.hypot(expected.center_x - predicted.center_x, expected.center_y - predicted.center_y))
            if distance <= thresholds.center_tolerance_px:
                candidates.append((distance, truth_index, prediction_index))
    candidates.sort(key=lambda item: item[0])
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    object_rows: list[ObjectEvaluation] = []
    center_errors: list[float] = []
    angle_errors: list[float] = []
    classification_correct = 0
    true_positive = 0
    for distance, truth_index, prediction_index in candidates:
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        expected, predicted = truth[truth_index], result.objects[prediction_index]
        angle_error = circular_angle_error(expected.angle_deg, predicted.angle_deg)
        class_ok = predicted.classification_status == "confirmed" and predicted.template_id == expected.template_id
        angle_ok = angle_error <= thresholds.angle_tolerance_deg
        if class_ok and angle_ok:
            true_positive += 1
        classification_correct += int(class_ok)
        center_errors.append(distance)
        angle_errors.append(angle_error)
        object_rows.append(ObjectEvaluation(
            expected.object_id, predicted.object_id, expected.template_name, predicted.template_name,
            distance, angle_error, class_ok, True, angle_ok,
        ))
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
    for index, expected in enumerate(truth):
        if index not in used_truth:
            object_rows.append(ObjectEvaluation(expected.object_id, None, expected.template_name, None, None, None, False, False, False))
    for index, predicted in enumerate(result.objects):
        if index not in used_predictions:
            object_rows.append(ObjectEvaluation(None, predicted.object_id, None, predicted.template_name, None, None, False, False, False))
    return SceneEvaluation(
        scene_id=scene_id,
        ground_truth_count=len(truth),
        prediction_count=result.object_count,
        matched_count=len(used_truth),
        true_positive=true_positive,
        false_positive=result.object_count - true_positive,
        false_negative=len(truth) - true_positive,
        classification_correct=classification_correct,
        ambiguous_count=result.ambiguous_count,
        elapsed_ms=elapsed_ms,
        center_errors_px=center_errors,
        angle_errors_deg=angle_errors,
        objects=object_rows,
    )


def summarize_evaluations(evaluations: list[SceneEvaluation]) -> BenchmarkSummary:
    tp = sum(item.true_positive for item in evaluations)
    fp = sum(item.false_positive for item in evaluations)
    fn = sum(item.false_negative for item in evaluations)
    matched = sum(item.matched_count for item in evaluations)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    center_errors = [value for item in evaluations for value in item.center_errors_px]
    angle_errors = [value for item in evaluations for value in item.angle_errors_deg]
    return BenchmarkSummary(
        scene_count=len(evaluations),
        ground_truth_count=sum(item.ground_truth_count for item in evaluations),
        prediction_count=sum(item.prediction_count for item in evaluations),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        exact_count_scenes=sum(item.count_exact for item in evaluations),
        ambiguous_count=sum(item.ambiguous_count for item in evaluations),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        classification_accuracy=sum(item.classification_correct for item in evaluations) / matched if matched else 0.0,
        mean_center_error_px=float(np.mean(center_errors)) if center_errors else None,
        p95_center_error_px=_percentile(center_errors, 95),
        mean_angle_error_deg=float(np.mean(angle_errors)) if angle_errors else None,
        p95_angle_error_deg=_percentile(angle_errors, 95),
        mean_elapsed_ms=float(np.mean([item.elapsed_ms for item in evaluations])) if evaluations else 0.0,
    )


def run_synthetic_benchmark(
    library_root: str | Path,
    output_root: str | Path,
    scene_count: int = 20,
    seed: int = 42,
    config: SyntheticStressConfig | None = None,
    thresholds: EvaluationThresholds | None = None,
    save_images: bool = True,
) -> tuple[Path, BenchmarkSummary]:
    if scene_count < 1:
        raise ValueError("scene_count must be positive")
    cfg = config or SyntheticStressConfig()
    limits = thresholds or EvaluationThresholds()
    library = TemplateLibrary(library_root).load()
    generator = SyntheticSceneGenerator(library, cfg, seed)
    matcher = MultiTemplateMatcher.from_library(library)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    scenes_dir, annotated_dir = output / "scenes", output / "annotated"
    if save_images:
        scenes_dir.mkdir(exist_ok=True)
        annotated_dir.mkdir(exist_ok=True)
    evaluations: list[SceneEvaluation] = []
    truth_lines: list[str] = []
    prediction_lines: list[str] = []
    evaluation_lines: list[str] = []
    for index in range(scene_count):
        scene_id = f"scene_{index + 1:04d}"
        image, truth = generator.generate(index)
        started = perf_counter()
        result = matcher.match(image)
        elapsed_ms = (perf_counter() - started) * 1000.0
        evaluation = evaluate_scene(scene_id, truth, result, elapsed_ms, limits)
        evaluations.append(evaluation)
        truth_lines.append(json.dumps({"scene_id": scene_id, "objects": [item.to_dict() for item in truth]}, ensure_ascii=False))
        prediction_lines.append(json.dumps({"scene_id": scene_id, **result.to_dict()}, ensure_ascii=False))
        evaluation_lines.append(json.dumps(evaluation.to_dict(), ensure_ascii=False))
        if save_images:
            write_image(scenes_dir / f"{scene_id}.png", image)
            write_image(annotated_dir / f"{scene_id}.png", draw_multi_template_matches(image, result))
    summary = summarize_evaluations(evaluations)
    (output / "ground_truth.jsonl").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")
    (output / "evaluations.jsonl").write_text("\n".join(evaluation_lines) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "benchmark_config.json").write_text(json.dumps({
        "seed": seed,
        "scene_count": scene_count,
        "synthetic": asdict(cfg),
        "evaluation": asdict(limits),
        "template_library": str(Path(library_root)),
        "templates": [item.entry.to_dict() for item in generator.templates],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_scene.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "scene_id", "ground_truth_count", "prediction_count", "matched_count", "true_positive",
            "false_positive", "false_negative", "classification_correct", "ambiguous_count",
            "count_exact", "elapsed_ms", "mean_center_error_px", "mean_angle_error_deg",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in evaluations:
            writer.writerow({
                **{key: getattr(item, key) for key in fields if hasattr(item, key)},
                "count_exact": item.count_exact,
                "mean_center_error_px": float(np.mean(item.center_errors_px)) if item.center_errors_px else "",
                "mean_angle_error_deg": float(np.mean(item.angle_errors_deg)) if item.angle_errors_deg else "",
            })
    return output, summary
