"""算法评估：运行数据库驱动的分割算法基准测试并汇总指标。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

from ..config import DEFAULT_CONFIG
from ..detectors.fluorescent import FluorescentTextileDetector, FluorescentTextileParameters
from ..experiment_db import ExperimentDatabase
from ..experiment_db.models import GroundTruthObject, ImageRecord
from ..io.image_reader import read_image, write_image
from ..measurement.coordinates import output_directed_angle_from_cv
from ..pipeline import SegPosePipeline
from ..types import ImageResult, ObjectResult


ALGORITHM_NAMES = ("threshold", "adaptive_threshold", "color_range", "watershed")
REAL_ALGORITHM_NAMES = (*ALGORITHM_NAMES, "fluorescent_textile")


@dataclass(frozen=True)
class ObjectMatch:
    gt_index: int
    prediction_index: int
    iou: float
    dice: float
    center_error_px: float
    axis_angle_error_deg: float | None
    directed_angle_error_deg: float | None


@dataclass
class ImageBenchmarkResult:
    image_id: int
    image_path: str
    profile: str
    gt_count: int
    prediction_count: int
    matches: list[ObjectMatch] = field(default_factory=list)
    false_positive: int = 0
    false_negative: int = 0
    count_error: int = 0
    count_accuracy: float = 0.0
    exact_count_accuracy: float = 0.0
    processing_time_ms: float = 0.0
    failure: bool = False
    failure_types: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class AlgorithmBenchmarkSummary:
    algorithm: str
    image_count: int
    ground_truth_count: int
    prediction_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    mean_iou: float
    mean_dice: float
    precision: float
    recall: float
    f1: float
    mean_count_error: float
    count_accuracy: float
    exact_count_accuracy: float
    mean_center_error_px: float | None
    mean_axis_angle_error_deg: float | None
    mean_directed_angle_error_deg: float | None
    mean_processing_time_ms: float
    p95_processing_time_ms: float
    fps: float
    failure_rate: float
    worst_profile: str | None


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first > 0, second > 0
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first > 0, second > 0
    denominator = int(np.count_nonzero(a) + np.count_nonzero(b))
    return float(2 * np.count_nonzero(a & b) / denominator) if denominator else 0.0


def axis_angle_error(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return float(min(difference, 180.0 - difference))


def directed_angle_error(first: float, second: float) -> float:
    difference = abs((first - second) % 360.0)
    return float(min(difference, 360.0 - difference))


def match_instances(
    ground_truth: list[GroundTruthObject],
    gt_masks: list[np.ndarray],
    predictions: list[ObjectResult],
    *,
    minimum_iou: float = 0.30,
) -> list[ObjectMatch]:
    """Greedy one-to-one matching ordered by descending instance Mask IoU."""
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_mask in enumerate(gt_masks):
        for prediction_index, prediction in enumerate(predictions):
            if prediction._mask is None:
                continue
            iou = mask_iou(gt_mask, prediction._mask)
            if iou >= minimum_iou:
                candidates.append((iou, gt_index, prediction_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_prediction: set[int] = set()
    matches: list[ObjectMatch] = []
    for iou, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_prediction:
            continue
        gt = ground_truth[gt_index]
        prediction = predictions[prediction_index]
        axis_error = (
            axis_angle_error(gt.axis_angle_deg, prediction.axis_angle_deg)
            if gt.axis_angle_deg is not None and prediction.axis_angle_deg is not None
            else None
        )
        directed_error = (
            directed_angle_error(gt.directed_angle_deg, prediction.directed_angle_deg)
            if gt.directed_angle_deg is not None and prediction.directed_angle_deg is not None
            else None
        )
        matches.append(
            ObjectMatch(
                gt_index,
                prediction_index,
                iou,
                mask_dice(gt_masks[gt_index], prediction._mask),
                float(np.hypot(gt.center_x - prediction.center_x, gt.center_y - prediction.center_y)),
                axis_error,
                directed_error,
            )
        )
        used_gt.add(gt_index)
        used_prediction.add(prediction_index)
    return matches


def default_algorithm_config(name: str) -> dict[str, Any]:
    if name not in ALGORITHM_NAMES:
        raise ValueError(f"Unsupported benchmark algorithm: {name}")
    config = deepcopy(DEFAULT_CONFIG)
    config["segmentation"]["backend"] = name
    config["input"]["resize_max_dimension"] = 0
    config["preprocessing"]["gaussian_blur_kernel"] = 5
    config["morphology"].update({"kernel_size": 3, "open_iterations": 1, "close_iterations": 1})
    config["filter"].update(
        {"min_area_px": 300, "max_area_px": None, "min_width_px": 12, "min_height_px": 12}
    )
    config["threshold"].update({"method": "otsu", "value": 127})
    config["adaptive_threshold"].update({"block_size": 51, "c": -4})
    config["color_range"].update({"space": "lab", "lower": [75, 0, 0], "upper": [255, 255, 255]})
    config["watershed"].update(
        {"distance_threshold_ratio": 0.32, "background_dilate_iterations": 2, "min_marker_area": 25}
    )
    return config


def create_algorithm_pipeline(name: str, config: dict[str, Any] | None = None):
    """Create a common process(image) adapter for a benchmark algorithm."""
    if name == "fluorescent_textile":
        resolved = config or {}
        values = resolved.get("fluorescent_textile", {})
        return _FluorescentBenchmarkPipeline(
            FluorescentTextileParameters(**values), resolved.get("input", {}).get("roi")
        )
    resolved = default_algorithm_config(name)
    if config:
        resolved = _deep_merge(resolved, config)
    return SegPosePipeline(resolved)


class _FluorescentBenchmarkPipeline:
    """Expose the production colour detector through the benchmark contract."""

    def __init__(self, parameters: FluorescentTextileParameters, roi=None):
        self.detector = FluorescentTextileDetector(parameters)
        self.roi = tuple(map(int, roi)) if roi else None

    def process(self, image: np.ndarray, image_name: str = "image") -> ImageResult:
        started = perf_counter()
        height, width = image.shape[:2]
        offset_x = offset_y = 0
        processing = image
        if self.roi:
            offset_x, offset_y, roi_width, roi_height = self.roi
            if (
                min(offset_x, offset_y, roi_width, roi_height) < 0
                or roi_width < 1
                or roi_height < 1
                or offset_x + roi_width > width
                or offset_y + roi_height > height
            ):
                raise ValueError("Benchmark ROI is outside image bounds")
            processing = image[offset_y : offset_y + roi_height, offset_x : offset_x + roi_width]
        result = self.detector.match(processing)
        objects: list[ObjectResult] = []
        for item in result.objects:
            points = np.asarray(item.contour_points, np.float32)
            if len(points) < 3:
                points = np.asarray(item.box_points, np.float32)
            if len(points) < 3:
                continue
            points = np.round(points + np.asarray([offset_x, offset_y], np.float32)).astype(np.int32)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [points], 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = map(float, rect[1])
            long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
            x, y, box_width, box_height = cv2.boundingRect(contour)
            directed_angle = output_directed_angle_from_cv(item.angle_deg)
            output_box = [
                [float(point[0] + offset_x), float((height - 1) - (point[1] + offset_y))]
                for point in item.box_points
            ]
            output_contour = [
                [float(point[0] + offset_x), float((height - 1) - (point[1] + offset_y))]
                for point in item.contour_points
            ]
            confidence = float(item.orientation_confidence or 0.0)
            objects.append(
                ObjectResult(
                    object_id=len(objects) + 1,
                    center_x=float(item.center_x + offset_x),
                    center_y=float((height - 1) - (item.center_y + offset_y)),
                    angle_deg=float(directed_angle % 180.0),
                    angle_method="fluorescent_directed",
                    angle_reliable=confidence >= 0.15,
                    orientation_confidence=confidence,
                    area_px=float(cv2.countNonZero(mask)),
                    perimeter_px=float(cv2.arcLength(contour, True)),
                    bbox_xywh=(x, height - (y + box_height), box_width, box_height),
                    rotated_box_points=output_box,
                    width_px=long_side,
                    height_px=short_side,
                    aspect_ratio=long_side / max(short_side, 1e-9),
                    touches_border=(
                        x <= 1 or y <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1
                    ),
                    contour_points=output_contour,
                    axis_angle_deg=float(directed_angle % 180.0),
                    directed_angle_deg=float(directed_angle),
                    direction_confidence=confidence,
                    _mask=mask,
                )
            )
        elapsed = (perf_counter() - started) * 1000.0
        return ImageResult(
            str(image_name), width, height, len(objects), objects, elapsed, "fluorescent_textile"
        )


class BenchmarkRunner:
    def __init__(
        self,
        database: ExperimentDatabase,
        output_dir: str | Path,
        *,
        minimum_iou: float = 0.30,
    ):
        if not 0.0 < minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be in (0, 1]")
        self.database = database
        self.output_dir = Path(output_dir)
        self.minimum_iou = minimum_iou

    def run(
        self,
        dataset_id: int,
        algorithms: Iterable[str] = ALGORITHM_NAMES,
        config_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, AlgorithmBenchmarkSummary]:
        self.database.initialize()
        images = self.database.images_for_dataset(dataset_id)
        if not images:
            raise ValueError(f"Dataset {dataset_id} contains no images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, AlgorithmBenchmarkSummary] = {}
        all_details: dict[str, list[dict[str, Any]]] = {}
        for name in algorithms:
            if name == "fluorescent_textile":
                config = {
                    "input": {"roi": None},
                    "fluorescent_textile": asdict(FluorescentTextileParameters()),
                }
            else:
                config = default_algorithm_config(name)
            if config_overrides and name in config_overrides:
                config = _deep_merge(config, config_overrides[name])
            summary, details = self._run_algorithm(dataset_id, images, name, config)
            summaries[name] = summary
            all_details[name] = [item.to_dict() for item in details]
        self._write_reports(summaries, all_details)
        return summaries

    def _run_algorithm(self, dataset_id, images, name, config):
        algorithm = self.database.get_or_create_algorithm(name)
        run = self.database.create_run(
            algorithm.id, dataset_id, config, git_state=_git_state()
        )
        run_dir = self.output_dir / name
        mask_dir = run_dir / "prediction_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        details: list[ImageBenchmarkResult] = []
        try:
            pipeline = create_algorithm_pipeline(name, config)
            for image_record in images:
                details.append(
                    self._evaluate_image(run.id, image_record, pipeline, mask_dir)
                )
            summary = summarize_algorithm(name, details)
            for metric_name, value in _summary_metrics(summary).items():
                self.database.add_metric(run.id, metric_name, value)
            self.database.finish_run(run.id, "completed")
            return summary, details
        except Exception:
            self.database.finish_run(run.id, "failed")
            raise

    def _evaluate_image(self, run_id, image_record, pipeline, mask_dir):
        profile = str((image_record.metadata_json or {}).get("profile", "unknown"))
        ground_truth = list(image_record.ground_truth_objects)
        gt_masks = [_read_mask(item.mask_path) for item in ground_truth]
        detail = ImageBenchmarkResult(
            image_record.id, image_record.file_path, profile, len(ground_truth), 0
        )
        try:
            image = read_image(image_record.file_path)
            result = pipeline.process(image, Path(image_record.file_path).name)
            detail.prediction_count = result.object_count
            detail.processing_time_ms = result.processing_time_ms
            for prediction in result.objects:
                mask_path = mask_dir / f"image_{image_record.id:06d}.object_{prediction.object_id:03d}.png"
                if prediction._mask is not None:
                    write_image(mask_path, prediction._mask)
                self.database.add_prediction(
                    run_id,
                    image_record.id,
                    prediction.object_id,
                    mask_path=str(mask_path) if prediction._mask is not None else None,
                    center_x=prediction.center_x,
                    center_y=prediction.center_y,
                    axis_angle_deg=prediction.axis_angle_deg,
                    directed_angle_deg=prediction.directed_angle_deg,
                    confidence=prediction.orientation_confidence,
                    processing_time_ms=result.processing_time_ms,
                    bbox=list(prediction.bbox_xywh),
                    metadata_json={"angle_reliable": prediction.angle_reliable},
                )
            detail.matches = match_instances(
                ground_truth, gt_masks, result.objects, minimum_iou=self.minimum_iou
            )
            detail.false_positive = result.object_count - len(detail.matches)
            detail.false_negative = len(ground_truth) - len(detail.matches)
            detail.count_error = abs(result.object_count - len(ground_truth))
            detail.count_accuracy = max(0.0, 1.0 - detail.count_error / max(len(ground_truth), 1))
            detail.exact_count_accuracy = float(detail.count_error == 0)
            detail.failure_types = _failure_types(ground_truth, gt_masks, result.objects, detail.matches)
            detail.failure = bool(detail.failure_types)
        except Exception as exc:
            detail.failure = True
            detail.failure_types = ["processing_error"]
            detail.error_message = f"{type(exc).__name__}: {exc}"
            detail.false_negative = len(ground_truth)
            detail.count_error = len(ground_truth)
        self._write_image_metrics(run_id, detail)
        return detail

    def _write_image_metrics(self, run_id: int, detail: ImageBenchmarkResult) -> None:
        values = {
            "iou": _mean([item.iou for item in detail.matches], denominator=detail.gt_count),
            "dice": _mean([item.dice for item in detail.matches], denominator=detail.gt_count),
            "precision": len(detail.matches) / max(detail.prediction_count, 1),
            "recall": len(detail.matches) / max(detail.gt_count, 1),
            "count_error": float(detail.count_error),
            "count_accuracy": detail.count_accuracy,
            "exact_count_accuracy": detail.exact_count_accuracy,
            "processing_time_ms": detail.processing_time_ms,
            "failure": float(detail.failure),
        }
        for name, value in values.items():
            self.database.add_metric(
                run_id,
                name,
                value,
                image_id=detail.image_id,
                metadata={"profile": detail.profile, "failure_types": detail.failure_types},
            )
        for item in detail.matches:
            for name, value in (
                ("center_error_px", item.center_error_px),
                ("axis_angle_error_deg", item.axis_angle_error_deg),
                ("directed_angle_error_deg", item.directed_angle_error_deg),
            ):
                if value is not None:
                    self.database.add_metric(
                        run_id,
                        name,
                        value,
                        image_id=detail.image_id,
                        object_index=item.gt_index + 1,
                    )

    def _write_reports(self, summaries, details) -> None:
        summary_payload = {name: asdict(summary) for name, summary in summaries.items()}
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "details.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fields = list(AlgorithmBenchmarkSummary.__dataclass_fields__)
        with (self.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for summary in summaries.values():
                writer.writerow(asdict(summary))
        rows = [
            "# SegPose Algorithm Benchmark",
            "",
            "| Algorithm | IoU | Dice | Precision | Recall | F1 | Center Error | Axis Angle Error | Exact Count | Time ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in summaries.values():
            rows.append(
                f"| {item.algorithm} | {item.mean_iou:.3f} | {item.mean_dice:.3f} | "
                f"{item.precision:.3f} | {item.recall:.3f} | {item.f1:.3f} | "
                f"{_fmt(item.mean_center_error_px)} | {_fmt(item.mean_axis_angle_error_deg)} | "
                f"{item.exact_count_accuracy:.3f} | {item.mean_processing_time_ms:.2f} |"
            )
        rows.extend(["", "## Failure cases", ""])
        failures = 0
        for algorithm, algorithm_details in details.items():
            for detail in algorithm_details:
                if detail["failure"]:
                    failures += 1
                    rows.append(
                        f"- `{algorithm}` image `{detail['image_id']}` ({detail['profile']}): "
                        + ", ".join(detail["failure_types"])
                    )
        if not failures:
            rows.append("- None")
        (self.output_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def summarize_algorithm(name: str, details: list[ImageBenchmarkResult]) -> AlgorithmBenchmarkSummary:
    matches = [match for detail in details for match in detail.matches]
    gt_count = sum(item.gt_count for item in details)
    prediction_count = sum(item.prediction_count for item in details)
    true_positive = len(matches)
    false_positive = sum(item.false_positive for item in details)
    false_negative = sum(item.false_negative for item in details)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    profile_scores: dict[str, list[float]] = {}
    for item in details:
        profile_scores.setdefault(item.profile, []).append(item.count_accuracy)
    worst_profile = min(profile_scores, key=lambda key: float(np.mean(profile_scores[key]))) if profile_scores else None
    timings = [item.processing_time_ms for item in details]
    mean_time = float(np.mean(timings)) if timings else 0.0
    return AlgorithmBenchmarkSummary(
        name,
        len(details),
        gt_count,
        prediction_count,
        true_positive,
        false_positive,
        false_negative,
        _mean([item.iou for item in matches], denominator=gt_count),
        _mean([item.dice for item in matches], denominator=gt_count),
        precision,
        recall,
        2 * precision * recall / max(precision + recall, 1e-12),
        float(np.mean([item.count_error for item in details])) if details else 0.0,
        float(np.mean([item.count_accuracy for item in details])) if details else 0.0,
        float(np.mean([item.exact_count_accuracy for item in details])) if details else 0.0,
        _nullable_mean([item.center_error_px for item in matches]),
        _nullable_mean([item.axis_angle_error_deg for item in matches]),
        _nullable_mean([item.directed_angle_error_deg for item in matches]),
        mean_time,
        float(np.percentile(timings, 95)) if timings else 0.0,
        1000.0 / mean_time if mean_time > 0 else 0.0,
        float(np.mean([item.failure for item in details])) if details else 0.0,
        worst_profile,
    )


def _failure_types(gt, gt_masks, predictions, matches) -> list[str]:
    failures: list[str] = []
    matched_gt = {item.gt_index for item in matches}
    matched_predictions = {item.prediction_index for item in matches}
    if len(matched_gt) < len(gt):
        failures.append("missed_object")
    if len(matched_predictions) < len(predictions):
        failures.append("false_positive")
    for gt_index, gt_mask in enumerate(gt_masks):
        overlaps = sum(
            prediction._mask is not None and mask_iou(gt_mask, prediction._mask) >= 0.10
            for prediction in predictions
        )
        if overlaps > 1:
            failures.append("split")
            break
    for prediction in predictions:
        if prediction._mask is None:
            continue
        overlaps = sum(mask_iou(gt_mask, prediction._mask) >= 0.10 for gt_mask in gt_masks)
        if overlaps > 1:
            failures.append("merge")
            break
    return list(dict.fromkeys(failures))


def _summary_metrics(summary: AlgorithmBenchmarkSummary) -> dict[str, float | None]:
    return {
        key: value
        for key, value in asdict(summary).items()
        if key not in {"algorithm", "worst_profile"} and isinstance(value, (int, float, type(None)))
    }


def _read_mask(path: str | None) -> np.ndarray:
    if path is None:
        raise ValueError("Ground-truth object has no mask_path")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode ground-truth mask: {path}")
    return image


def _mean(values: list[float], *, denominator: int) -> float:
    return float(sum(values) / denominator) if denominator else 0.0


def _nullable_mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _git_state() -> str | None:
    try:
        return subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
