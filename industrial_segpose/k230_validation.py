"""Offline validation of a K230 template bundle against captured images.

This intentionally mirrors the low-cost segmentation and geometry scoring used
on the module.  It lets template profiles be tuned on a desktop before a bundle
is copied to the SD card.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from k230_runtime.detector import normalize_gripper_angle, robust_lab_threshold
from .io.image_reader import read_image, write_image


def _load_bundle(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    templates: list[dict] = []
    for item in manifest.get("templates", []):
        metadata_path = manifest_path.parent / item["metadata_file"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("enabled", True):
            templates.append(metadata)
    return templates


def _lab_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 0] * (100.0 / 255.0), lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0


def _template_mask(planes: tuple[np.ndarray, np.ndarray, np.ndarray], template: dict) -> np.ndarray:
    luminance, channel_a, channel_b = planes
    segmentation = template.get("segmentation", {})
    output = np.zeros(luminance.shape, np.uint8)
    for raw in segmentation.get("lab_thresholds", []):
        threshold = robust_lab_threshold(segmentation, raw)
        if threshold is None:
            continue
        selected = (
            (luminance >= threshold[0]) & (luminance <= threshold[1])
            & (channel_a >= threshold[2]) & (channel_a <= threshold[3])
            & (channel_b >= threshold[4]) & (channel_b <= threshold[5])
        )
        output[selected] = 255
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return output


def _principal_angle(contour: np.ndarray) -> float:
    samples = contour.reshape(-1, 2).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(samples, mean=None)
    vector = eigenvectors[0]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _score_contour(contour: np.ndarray, template: dict) -> dict | None:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    (center_x, center_y), (first, second), _ = cv2.minAreaRect(contour)
    observed_long = max(float(first), float(second))
    observed_short = max(1.0, min(float(first), float(second)))
    shape = template.get("shape_features", {})
    template_long = max(1.0, float(shape.get("major_axis_length_px", template.get("width", 1))))
    template_short = max(1.0, float(shape.get("minor_axis_length_px", template.get("height", 1))))
    bbox_scale = math.sqrt((observed_long / template_long) * (observed_short / template_short))
    area_scale = math.sqrt(area / max(float(template.get("mask_area_px", 1)), 1.0))
    scale = 0.55 * bbox_scale + 0.45 * area_scale
    parameters = template.get("parameters", {})
    if not float(parameters.get("scale_min", 0.45)) <= scale <= float(parameters.get("scale_max", 1.8)):
        return None
    ratio = observed_long / observed_short
    template_ratio = template_long / template_short
    ratio_score = max(0.0, min(1.0, 1.0 - abs(ratio - template_ratio) / max(template_ratio, 0.01)))
    density = area / max(observed_long * observed_short, 1.0)
    expected_density = float(shape.get("axis_fill_ratio", template.get("fill_ratio", 0.5)))
    density_score = max(0.0, min(1.0, 1.0 - abs(density - expected_density) / max(expected_density, 0.15)))
    scale_score = max(0.0, min(1.0, 1.0 - abs(bbox_scale - area_scale) / max(bbox_scale, area_scale, 0.1)))
    confidence = (0.34 * ratio_score + 0.24 * density_score + 0.18 * scale_score) / 0.76
    if confidence < float(parameters.get("score_threshold", 0.60)):
        return None
    angle = normalize_gripper_angle(_principal_angle(contour) - float(template.get("reference_angle_deg", 0.0)))
    x, y, width, height = cv2.boundingRect(contour)
    return {
        "template_id": template["template_id"],
        "template_name": template.get("name", template["template_id"]),
        "confidence": round(float(confidence), 4),
        "center_xy": [round(float(center_x), 2), round(float(center_y), 2)],
        "angle_deg": round(float(angle), 2),
        "scale": round(float(scale), 4),
        "bbox": [int(x), int(y), int(width), int(height)],
        "contour": contour,
    }


def validate_image(image: np.ndarray, templates: list[dict]) -> tuple[list[dict], dict[str, np.ndarray]]:
    planes = _lab_planes(image)
    candidates: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    frame_area = image.shape[0] * image.shape[1]
    for template in templates:
        mask = _template_mask(planes, template)
        masks[template["template_id"]] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = float(template.get("segmentation", {}).get("area_threshold", 120))
        for contour in contours:
            if cv2.contourArea(contour) < minimum or cv2.contourArea(contour) >= frame_area * 0.85:
                continue
            candidate = _score_contour(contour, template)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    selected: list[dict] = []
    for candidate in candidates:
        if any(math.dist(candidate["center_xy"], item["center_xy"]) < 0.6 * min(candidate["bbox"][2:]) for item in selected):
            continue
        selected.append(candidate)
    return selected, masks


def _serializable(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "contour"}


def validate_paths(manifest_path: Path, image_paths: Iterable[str | Path], output_root: Path) -> Path:
    templates = _load_bundle(manifest_path)
    if not templates:
        raise ValueError("K230 template bundle has no enabled templates")
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"bundle": str(manifest_path), "images": []}
    for index, raw_path in enumerate(image_paths, start=1):
        image_path = Path(raw_path)
        try:
            image = read_image(image_path)
        except (OSError, ValueError, FileNotFoundError):
            continue
        results, _ = validate_image(image, templates)
        preview = image.copy()
        for result in results:
            cv2.drawContours(preview, [result["contour"]], -1, (0, 210, 255), 3)
            x, y = map(int, result["center_xy"])
            cv2.drawMarker(preview, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
            cv2.putText(preview, "%s %.3f %.1fdeg" % (result["template_name"], result["confidence"], result["angle_deg"]), (max(4, x - 100), max(24, y - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2, cv2.LINE_AA)
        preview_path = output_root / (f"{index:04d}_{image_path.stem}_validated.jpg")
        write_image(preview_path, preview)
        report["images"].append({"source": str(image_path), "preview": str(preview_path), "detections": [_serializable(item) for item in results]})
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def validate_folder(manifest_path: Path, image_root: Path, output_root: Path) -> Path:
    image_paths = sorted(
        path for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return validate_paths(manifest_path, image_paths, output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate K230 templates against captured images on the desktop")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", default=Path("reports/k230_validation"), type=Path)
    args = parser.parse_args(argv)
    print(validate_folder(args.bundle, args.images, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
