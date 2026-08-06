"""Export the desktop template library into a compact K230 deployment bundle."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .io.image_reader import write_image
from .template_matching import TemplateLibrary


K230_BUNDLE_VERSION = 1
POSE_CANVAS_SIZE = 192


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tight_crop(image: np.ndarray, mask: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Template mask is empty")
    x, y, width, height = cv2.boundingRect(points)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.shape[1], x + width + padding)
    bottom = min(image.shape[0], y + height + padding)
    return image[top:bottom, left:right].copy(), mask[top:bottom, left:right].copy()


def _resize(image: np.ndarray, mask: np.ndarray, max_edge: int) -> tuple[np.ndarray, np.ndarray, float]:
    scale = min(1.0, float(max_edge) / max(image.shape[:2]))
    if scale >= 0.999:
        return image, mask, 1.0
    size = (
        max(8, int(round(image.shape[1] * scale))),
        max(8, int(round(image.shape[0] * scale))),
    )
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return resized_image, np.where(resized_mask > 0, 255, 0).astype(np.uint8), scale


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments["m00"]) < 1e-9:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _safe_pick_point(mask: np.ndarray) -> tuple[tuple[float, float], float]:
    distance = cv2.distanceTransform(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    return (float(location[0]), float(location[1])), float(radius)


def _contour(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return simplified.astype(float).tolist()


def _reference_angle(mask: np.ndarray) -> float:
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 2:
        return 0.0
    samples = points.reshape(-1, 2).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(samples, mean=None)
    vector = eigenvectors[0]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _shape_features(mask: np.ndarray) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 3:
        height, width = mask.shape[:2]
        major, minor = float(max(width, height)), float(max(1, min(width, height)))
    else:
        (_, _), (first, second), _ = cv2.minAreaRect(points)
        major, minor = float(max(first, second)), float(max(1.0, min(first, second)))
    area = float(cv2.contourArea(contour)) if contour is not None else float(np.count_nonzero(mask))
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None else area
    return {
        "major_axis_length_px": major,
        "minor_axis_length_px": minor,
        "axis_fill_ratio": float(np.count_nonzero(mask) / max(major * minor, 1.0)),
        "elongation": float(1.0 - minor / max(major, 1.0)),
        "solidity": float(area / max(hull_area, 1.0)),
        "roundness": float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0)),
    }


def _pose_canvas(gray: np.ndarray, mask: np.ndarray, size: int = POSE_CANVAS_SIZE) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    scale = min((size - 16) / gray.shape[1], (size - 16) / gray.shape[0])
    target = (
        max(8, int(round(gray.shape[1] * scale))),
        max(8, int(round(gray.shape[0] * scale))),
    )
    resized_gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST)
    left = (size - target[0]) // 2
    top = (size - target[1]) // 2
    region = canvas[top : top + target[1], left : left + target[0]]
    region[resized_mask > 0] = resized_gray[resized_mask > 0]
    return canvas


def _segmentation_profile(image: np.ndarray, mask: np.ndarray) -> dict:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0].astype(np.float32)
    if pixels.size == 0:
        raise ValueError("Template mask is empty")
    low = np.percentile(pixels, 3, axis=0)
    high = np.percentile(pixels, 97, axis=0)
    median = np.median(pixels, axis=0)
    # OpenMV/CanMV LAB ranges are L=[0,100], A/B=[-128,127].
    threshold = [
        int(max(0, round(low[0] * 100.0 / 255.0) - 7)),
        int(min(100, round(high[0] * 100.0 / 255.0) + 7)),
        int(max(-128, round(low[1] - 128) - 8)),
        int(min(127, round(high[1] - 128) + 8)),
        int(max(-128, round(low[2] - 128) - 8)),
        int(min(127, round(high[2] - 128) + 8)),
    ]
    median_l = float(median[0] * 100.0 / 255.0)
    foreground_lab_median = [
        median_l,
        float(median[1] - 128.0),
        float(median[2] - 128.0),
    ]
    chroma = float(np.hypot(median[1] - 128.0, median[2] - 128.0))
    background = lab[mask == 0].astype(np.float32)
    background_l = float(np.median(background[:, 0]) * 100.0 / 255.0) if background.size else median_l
    if background.size:
        background_median = np.median(background, axis=0)
        background_lab_median = [
            background_l,
            float(background_median[1] - 128.0),
            float(background_median[2] - 128.0),
        ]
    else:
        background_lab_median = list(foreground_lab_median)

    # A broad rectangular LAB range is deliberately tolerant to exposure, but
    # can consequently include most of a neutral conveyor.  Store the channel
    # that best separates foreground from the local background so the K230 can
    # restore that boundary after applying exposure tolerance.
    foreground_canmv = np.column_stack(
        (pixels[:, 0] * 100.0 / 255.0, pixels[:, 1] - 128.0, pixels[:, 2] - 128.0)
    )
    if background.size:
        background_canmv = np.column_stack(
            (background[:, 0] * 100.0 / 255.0, background[:, 1] - 128.0, background[:, 2] - 128.0)
        )
        fg_spread = np.maximum(np.percentile(foreground_canmv, 75, axis=0) - np.percentile(foreground_canmv, 25, axis=0), 1.0)
        bg_spread = np.maximum(np.percentile(background_canmv, 75, axis=0) - np.percentile(background_canmv, 25, axis=0), 1.0)
        delta = np.asarray(foreground_lab_median) - np.asarray(background_lab_median)
        separation = np.abs(delta) / (fg_spread + bg_spread)
        channel_index = int(np.argmax(separation))
        channel_names = ("l", "a", "b")
        discriminative_channel = {
            "channel": channel_names[channel_index],
            "foreground_side": "above" if delta[channel_index] >= 0 else "below",
            "cutoff": float(
                (foreground_lab_median[channel_index] + background_lab_median[channel_index]) * 0.5
            ),
            "foreground_median": float(foreground_lab_median[channel_index]),
            "background_median": float(background_lab_median[channel_index]),
            "separation_score": float(separation[channel_index]),
            "enabled": bool(abs(delta[channel_index]) >= 3.0),
        }
    else:
        discriminative_channel = {"enabled": False}
    dark_texture = median_l < 55.0 and chroma < 22.0
    polarity = "similar"
    if dark_texture and median_l - background_l > 4.0:
        threshold[0] = max(threshold[0], int(round((median_l + background_l) * 0.5)))
        polarity = "brighter"
    elif dark_texture and background_l - median_l > 4.0:
        threshold[1] = min(threshold[1], int(round((median_l + background_l) * 0.5)))
        polarity = "darker"
    return {
        "mode": "dark_texture" if dark_texture else "color",
        "polarity": polarity,
        "foreground_l_median": median_l,
        "background_l_median": background_l,
        "foreground_lab_median": foreground_lab_median,
        "background_lab_median": background_lab_median,
        "discriminative_channel": discriminative_channel,
        "lab_thresholds": [threshold],
        "exposure_tolerance_l": 10,
        "chroma_tolerance": 3,
        "pixels_threshold": max(80, int(np.count_nonzero(mask) * 0.015)),
        "area_threshold": max(120, int(np.count_nonzero(mask) * 0.025)),
        "merge": True,
        "margin": 4,
    }


def export_k230_bundle(
    library_root: str | Path,
    output_root: str | Path,
    max_edge: int = 384,
    padding: int = 6,
    overwrite: bool = False,
) -> Path:
    if max_edge < 64:
        raise ValueError("max_edge must be at least 64")
    library = TemplateLibrary(library_root).load()
    destination = Path(output_root)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"K230 bundle directory is not empty: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    templates: list[dict] = []
    errors: dict[str, str] = {}
    for loaded in library.load_entries():
        entry = loaded.entry
        if not entry.enabled:
            continue
        if not loaded.valid:
            errors[entry.template_id] = loaded.error or "invalid template"
            continue
        try:
            image, mask = _tight_crop(loaded.model.image, loaded.model.mask, padding)
            image, mask, resize_scale = _resize(image, mask, max_edge)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            edge = cv2.Canny(gray, 45, 135)
            edge = cv2.bitwise_and(edge, edge, mask=mask)
            pose = _pose_canvas(gray, mask)
            template_dir = staging / entry.template_id
            template_dir.mkdir()
            files = {
                "gray": template_dir / "template_gray.png",
                "mask": template_dir / "template_mask.png",
                "edge": template_dir / "template_edge.png",
                "pose": template_dir / "template_pose.pgm",
            }
            write_image(files["gray"], gray)
            write_image(files["mask"], mask)
            write_image(files["edge"], edge)
            write_image(files["pose"], pose)
            center = _mask_center(mask)
            pick_point, pick_radius = _safe_pick_point(mask)
            mask_area = int(np.count_nonzero(mask))
            metadata = {
                "format_version": K230_BUNDLE_VERSION,
                "template_id": entry.template_id,
                "name": entry.name,
                "enabled": True,
                "color_bgr": list(entry.color_bgr),
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "reference_center_xy": list(center),
                "pick_point_xy": list(pick_point),
                "pick_radius_px": pick_radius,
                "mask_area_px": mask_area,
                "fill_ratio": float(mask_area / mask.size),
                "shape_features": _shape_features(mask),
                "reference_angle_deg": _reference_angle(mask),
                "pose_canvas_size": POSE_CANVAS_SIZE,
                "segmentation": _segmentation_profile(image, mask),
                "contour_points": _contour(mask),
                "desktop_to_k230_scale": resize_scale,
                "parameters": asdict(entry.parameters),
                "assets": {key: path.name for key, path in files.items()},
                "sha256": {key: _sha256(path) for key, path in files.items()},
            }
            metadata_path = template_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            templates.append(
                {
                    "template_id": entry.template_id,
                    "name": entry.name,
                    "metadata_file": f"{entry.template_id}/metadata.json",
                    "metadata_sha256": _sha256(metadata_path),
                }
            )
        except Exception as exc:
            errors[entry.template_id] = str(exc)

    manifest = {
        "format_version": K230_BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_library_format": library.FORMAT_VERSION,
        "ambiguity_margin": library.ambiguity_margin,
        "cross_template_iou": library.cross_template_iou,
        "max_template_edge": int(max_edge),
        "templates": templates,
        "skipped_errors": errors,
    }
    manifest_path = staging / "template_library.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)
    return destination / "template_library.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a compact K230 template bundle")
    parser.add_argument("--library", default="templates")
    parser.add_argument("--output", default="k230_templates")
    parser.add_argument("--max-edge", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    path = export_k230_bundle(args.library, args.output, args.max_edge, overwrite=args.overwrite)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
