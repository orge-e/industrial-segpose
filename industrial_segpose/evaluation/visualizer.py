"""算法评估：生成不同分割算法的可复用视觉对比图。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .runner import (
    ALGORITHM_NAMES,
    REAL_ALGORITHM_NAMES,
    create_algorithm_pipeline,
    mask_iou,
)
from ..io.image_reader import read_image, write_image
from ..measurement.orientation import min_area_rect_orientation
from ..pipeline import SegPosePipeline
from ..types import ImageResult, ObjectResult


ALGORITHM_LABELS = {
    "threshold": "THRESHOLD",
    "adaptive_threshold": "ADAPTIVE",
    "color_range": "COLOR RANGE",
    "watershed": "WATERSHED",
    "fluorescent_textile": "FLUORESCENT TEXTILE",
}


@dataclass(frozen=True)
class ComparisonEntry:
    profile: str
    image_path: str
    ground_truth_count: int
    predictions: dict[str, int]
    processing_time_ms: dict[str, float]


@dataclass(frozen=True)
class ComparisonBoardResult:
    output_path: Path
    manifest_path: Path
    image_count: int
    algorithms: tuple[str, ...]
    entries: tuple[ComparisonEntry, ...]


def generate_mask_directory_comparison_board(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    max_images: int = 12,
    algorithms: Iterable[str] = ALGORITHM_NAMES,
    tile_width: int = 360,
    minimum_iou: float = 0.30,
    config_overrides: dict[str, dict] | None = None,
) -> ComparisonBoardResult:
    """Render a reusable board for a reviewed real-image dataset.

    The dataset must contain ``images/`` and ``label_maps/``.  Every image is
    paired with ``label_maps/<stem>.labels.png``.  Instance IDs are positive
    integer pixels; zero is the background.  This format is also produced by
    the project's Labelme conversion workflow.
    """
    root = Path(dataset_root)
    destination = Path(output_path)
    algorithm_names = tuple(algorithms)
    _validate_options(root, 1, algorithm_names, tile_width, minimum_iou)
    if max_images < 1:
        raise ValueError("max_images must be positive")
    image_dir = root / "images"
    label_dir = root / "label_maps"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Real dataset must contain images/ and label_maps/ directories")
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    pairs = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in supported):
        label_path = label_dir / f"{image_path.stem}.labels.png"
        if label_path.is_file():
            pairs.append((image_path, label_path))
    if not pairs:
        raise ValueError(f"No reviewed image/label-map pairs found below: {root}")
    selected = pairs[:max_images]
    pipelines = {
        name: create_algorithm_pipeline(name, (config_overrides or {}).get(name))
        for name in algorithm_names
    }
    rows: list[np.ndarray] = []
    entries: list[ComparisonEntry] = []
    for image_path, label_path in selected:
        image = read_image(image_path)
        objects, masks = _ground_truth_from_label_map(label_path, image.shape[:2])
        panels = [_render_ground_truth_panel(image, "real", objects, masks, tile_width)]
        prediction_counts: dict[str, int] = {}
        timings: dict[str, float] = {}
        for name, pipeline in pipelines.items():
            try:
                result = pipeline.process(image.copy(), image_path.name)
                panels.append(_render_prediction_panel(image, name, result, masks, tile_width, minimum_iou))
                prediction_counts[name] = result.object_count
                timings[name] = result.processing_time_ms
            except Exception as exc:
                panels.append(_render_error_panel(image, name, exc, tile_width))
                prediction_counts[name] = 0
                timings[name] = 0.0
        rows.append(cv2.hconcat(panels))
        entries.append(
            ComparisonEntry(
                profile="real",
                image_path=str(image_path),
                ground_truth_count=len(objects),
                predictions=prediction_counts,
                processing_time_ms=timings,
            )
        )
    board_width = rows[0].shape[1]
    board = cv2.vconcat([_render_title(board_width, len(rows), algorithm_names), *rows])
    write_image(destination, board)
    manifest_path = destination.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "output_path": str(destination),
                "available_pairs": len(pairs),
                "rendered_pairs": len(selected),
                "minimum_iou": minimum_iou,
                "tile_width": tile_width,
                "algorithms": list(algorithm_names),
                "entries": [asdict(item) for item in entries],
                "legend": {
                    "cyan": "ground truth",
                    "green": "matched prediction",
                    "magenta": "false positive",
                    "red": "missed ground truth",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ComparisonBoardResult(destination, manifest_path, len(selected), algorithm_names, tuple(entries))


def generate_comparison_board(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    profiles: Iterable[str] | None = None,
    images_per_profile: int = 1,
    algorithms: Iterable[str] = ALGORITHM_NAMES,
    tile_width: int = 360,
    minimum_iou: float = 0.30,
) -> ComparisonBoardResult:
    """Run algorithms and place their visual results side by side.

    Rows are source images and columns are ``INPUT + GT`` followed by one
    column for every requested algorithm. Selection is deterministic: the
    lexicographically first images are used for each profile.
    """
    root = Path(dataset_root)
    destination = Path(output_path)
    algorithm_names = tuple(algorithms)
    _validate_options(root, images_per_profile, algorithm_names, tile_width, minimum_iou)
    selected = _select_images(root, profiles, images_per_profile)
    if not selected:
        raise ValueError(f"No benchmark images found below: {root}")

    pipelines = {name: create_algorithm_pipeline(name) for name in algorithm_names}
    rows: list[np.ndarray] = []
    entries: list[ComparisonEntry] = []
    for profile, image_path in selected:
        image = read_image(image_path)
        gt_objects, gt_masks = _load_ground_truth(root / profile, image_path.stem)
        panels = [_render_ground_truth_panel(image, profile, gt_objects, gt_masks, tile_width)]
        prediction_counts: dict[str, int] = {}
        timings: dict[str, float] = {}
        for name, pipeline in pipelines.items():
            try:
                result = pipeline.process(image.copy(), image_path.name)
                panels.append(
                    _render_prediction_panel(
                        image, name, result, gt_masks, tile_width, minimum_iou
                    )
                )
                prediction_counts[name] = result.object_count
                timings[name] = result.processing_time_ms
            except Exception as exc:  # comparison boards should expose failures, not abort
                panels.append(_render_error_panel(image, name, exc, tile_width))
                prediction_counts[name] = 0
                timings[name] = 0.0
        rows.append(cv2.hconcat(panels))
        entries.append(
            ComparisonEntry(
                profile=profile,
                image_path=str(image_path),
                ground_truth_count=len(gt_objects),
                predictions=prediction_counts,
                processing_time_ms=timings,
            )
        )

    board_width = rows[0].shape[1]
    title = _render_title(board_width, len(selected), algorithm_names)
    board = cv2.vconcat([title, *rows])
    write_image(destination, board)
    manifest_path = destination.with_suffix(".json")
    manifest = {
        "dataset_root": str(root),
        "output_path": str(destination),
        "minimum_iou": minimum_iou,
        "tile_width": tile_width,
        "algorithms": list(algorithm_names),
        "entries": [asdict(item) for item in entries],
        "legend": {
            "cyan": "ground truth",
            "green": "matched prediction",
            "magenta": "false positive",
            "red": "missed ground truth",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ComparisonBoardResult(
        destination, manifest_path, len(selected), algorithm_names, tuple(entries)
    )


def _validate_options(root, images_per_profile, algorithms, tile_width, minimum_iou) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if images_per_profile < 1:
        raise ValueError("images_per_profile must be positive")
    if tile_width < 240:
        raise ValueError("tile_width must be at least 240 pixels")
    if not 0.0 < minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be in (0, 1]")
    unsupported = [name for name in algorithms if name not in REAL_ALGORITHM_NAMES]
    if unsupported:
        raise ValueError(f"Unsupported algorithms: {', '.join(unsupported)}")
    if not algorithms:
        raise ValueError("At least one algorithm is required")


def _select_images(root: Path, profiles, images_per_profile: int) -> list[tuple[str, Path]]:
    requested = list(profiles) if profiles else sorted(
        item.name for item in root.iterdir() if (item / "images").is_dir()
    )
    selected: list[tuple[str, Path]] = []
    for profile in requested:
        image_dir = root / profile / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Profile image directory does not exist: {image_dir}")
        for image_path in sorted(image_dir.glob("scene_*.png"))[:images_per_profile]:
            selected.append((profile, image_path))
    return selected


def _load_ground_truth(profile_root: Path, stem: str) -> tuple[list[dict], list[np.ndarray]]:
    metadata_path = profile_root / "metadata" / f"{stem}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    objects = list(metadata.get("objects", []))
    masks = []
    for item in objects:
        mask_path = profile_root / "instance_masks" / (
            f"{stem}.object_{int(item['object_id']):03d}.png"
        )
        masks.append(cv2.cvtColor(read_image(mask_path), cv2.COLOR_BGR2GRAY))
    return objects, masks


def _ground_truth_from_label_map(
    label_path: Path, expected_shape: tuple[int, int]
) -> tuple[list[dict], list[np.ndarray]]:
    data = np.fromfile(label_path, np.uint8)
    labels = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"Could not decode label map: {label_path}")
    if labels.ndim == 3:
        labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
    if labels.shape != expected_shape:
        raise ValueError(
            f"Label map shape {labels.shape} does not match image shape {expected_shape}: {label_path}"
        )
    objects: list[dict] = []
    masks: list[np.ndarray] = []
    for instance_id in sorted(int(value) for value in np.unique(labels) if value > 0):
        mask = np.where(labels == instance_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        center_x = float(moments["m10"] / moments["m00"])
        center_y_cv = float(moments["m01"] / moments["m00"])
        angle, _ = min_area_rect_orientation(contour)
        objects.append(
            {
                "object_id": instance_id,
                "center": [center_x, float(mask.shape[0] - 1 - center_y_cv)],
                "axis_angle_deg": angle,
            }
        )
        masks.append(mask)
    return objects, masks


def _render_title(width: int, image_count: int, algorithms: tuple[str, ...]) -> np.ndarray:
    canvas = np.full((76, width, 3), (22, 25, 31), np.uint8)
    cv2.putText(
        canvas, "SEGMENTATION VISUAL COMPARISON", (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 243, 248), 2, cv2.LINE_AA,
    )
    subtitle = (
        f"{image_count} input images | {len(algorithms)} algorithms | "
        "cyan=GT  green=match  magenta=FP  red=miss"
    )
    cv2.putText(
        canvas, subtitle, (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
        0.47, (153, 165, 180), 1, cv2.LINE_AA,
    )
    return canvas


def _base_panel(image: np.ndarray, title: str, subtitle: str, tile_width: int) -> tuple[np.ndarray, float, int]:
    header_height = 56
    scale = tile_width / image.shape[1]
    image_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (tile_width, image_height), interpolation=cv2.INTER_AREA)
    panel = np.full((header_height + image_height, tile_width, 3), (15, 18, 23), np.uint8)
    panel[header_height:] = resized
    cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (238, 242, 248), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (163, 174, 188), 1, cv2.LINE_AA)
    return panel, scale, header_height


def _render_ground_truth_panel(image, profile, objects, masks, tile_width) -> np.ndarray:
    panel, scale, top = _base_panel(
        image, "INPUT + GROUND TRUTH", f"profile={profile}  gt={len(objects)}", tile_width
    )
    overlay = panel[top:].copy()
    for item, mask in zip(objects, masks):
        scaled_mask = _resize_mask(mask, overlay.shape[1], overlay.shape[0])
        _blend_mask(overlay, scaled_mask, (220, 190, 35), 0.20)
        _draw_mask_contour(overlay, scaled_mask, (255, 220, 55), 2)
        _draw_pose(
            overlay,
            float(item["center"][0]) * scale,
            (image.shape[0] - 1 - float(item["center"][1])) * scale,
            float(item.get("axis_angle_deg", 0.0)),
            f"GT{int(item['object_id'])}",
            (255, 220, 55),
        )
    panel[top:] = overlay
    return panel


def _render_prediction_panel(image, name, result, gt_masks, tile_width, minimum_iou) -> np.ndarray:
    matches = _greedy_mask_matches(gt_masks, result.objects, minimum_iou)
    matched_gt = {gt for gt, _ in matches}
    matched_predictions = {pred for _, pred in matches}
    subtitle = f"pred={result.object_count}/{len(gt_masks)}  {result.processing_time_ms:.1f} ms"
    panel, scale, top = _base_panel(image, ALGORITHM_LABELS[name], subtitle, tile_width)
    overlay = panel[top:].copy()

    for gt_index, gt_mask in enumerate(gt_masks):
        if gt_index not in matched_gt:
            _draw_mask_contour(
                overlay, _resize_mask(gt_mask, overlay.shape[1], overlay.shape[0]), (45, 45, 230), 3
            )
    for prediction_index, prediction in enumerate(result.objects):
        if prediction._mask is None:
            continue
        matched = prediction_index in matched_predictions
        color = (75, 210, 95) if matched else (210, 55, 210)
        scaled_mask = _resize_mask(prediction._mask, overlay.shape[1], overlay.shape[0])
        _blend_mask(overlay, scaled_mask, color, 0.22)
        _draw_mask_contour(overlay, scaled_mask, color, 2)
        _draw_pose(
            overlay,
            prediction.center_x * scale,
            (image.shape[0] - 1 - prediction.center_y) * scale,
            prediction.axis_angle_deg or 0.0,
            f"#{prediction.object_id} {prediction.axis_angle_deg or 0.0:.1f}deg",
            color,
        )
    panel[top:] = overlay
    return panel


def _render_error_panel(image, name, error: Exception, tile_width: int) -> np.ndarray:
    panel, _, top = _base_panel(image, ALGORITHM_LABELS[name], "PROCESSING ERROR", tile_width)
    shade = panel[top:].copy()
    shade[:] = (20, 20, 45)
    panel[top:] = cv2.addWeighted(panel[top:], 0.30, shade, 0.70, 0)
    cv2.putText(
        panel, type(error).__name__, (16, top + 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (70, 90, 245), 2, cv2.LINE_AA,
    )
    return panel


def _greedy_mask_matches(gt_masks, predictions, minimum_iou) -> list[tuple[int, int]]:
    candidates = []
    for gt_index, gt_mask in enumerate(gt_masks):
        for prediction_index, prediction in enumerate(predictions):
            if prediction._mask is None:
                continue
            score = mask_iou(gt_mask, prediction._mask)
            if score >= minimum_iou:
                candidates.append((score, gt_index, prediction_index))
    candidates.sort(reverse=True)
    used_gt, used_predictions, matches = set(), set(), []
    for _, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        matches.append((gt_index, prediction_index))
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
    return matches


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    active = mask > 0
    if not np.any(active):
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[active] = np.clip(
        image[active].astype(np.float32) * (1.0 - alpha) + color_array * alpha, 0, 255
    ).astype(np.uint8)


def _draw_mask_contour(image: np.ndarray, mask: np.ndarray, color, thickness: int) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _draw_pose(image, center_x, center_y, axis_angle_deg, label, color) -> None:
    center = (int(round(center_x)), int(round(center_y)))
    length = max(16, int(min(image.shape[:2]) * 0.075))
    theta = np.deg2rad(-axis_angle_deg)
    endpoint = (
        int(round(center[0] + length * np.cos(theta))),
        int(round(center[1] + length * np.sin(theta))),
    )
    cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.arrowedLine(image, center, endpoint, color, 2, cv2.LINE_AA, tipLength=0.22)
    x = max(4, min(center[0] + 7, image.shape[1] - 118))
    y = max(16, min(center[1] - 8, image.shape[0] - 6))
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
