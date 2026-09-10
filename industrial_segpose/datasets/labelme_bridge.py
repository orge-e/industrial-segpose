"""Bridge project detectors, Labelme polygon editing, and instance label maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..detectors.fluorescent import FluorescentTextileDetector, FluorescentTextileParameters
from ..io.image_reader import read_image, write_image
from ..measurement.orientation import min_area_rect_orientation


@dataclass(frozen=True)
class LabelmePreannotationResult:
    json_path: Path
    overlay_path: Path
    object_count: int


@dataclass(frozen=True)
class LabelMapConversionResult:
    label_map_path: Path
    overlay_path: Path
    metadata_path: Path
    object_count: int


def create_fluorescent_labelme_preannotation(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    label: str = "workpiece",
    parameters: FluorescentTextileParameters | None = None,
) -> LabelmePreannotationResult:
    """Run the factory detector and save editable Labelme polygons."""
    source = Path(image_path)
    image = read_image(source)
    result = FluorescentTextileDetector(parameters).match(image)
    shapes = []
    for object_result in result.objects:
        points = [list(map(float, point)) for point in object_result.contour_points]
        if len(points) < 3:
            points = [list(map(float, point)) for point in object_result.box_points]
        if len(points) < 3:
            continue
        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": int(object_result.object_id),
                "description": (
                    f"auto score={object_result.score:.3f}; "
                    f"status={object_result.classification_status}"
                ),
                "shape_type": "polygon",
                "flags": {"auto_generated": True},
                "mask": None,
            }
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{source.stem}.json"
    payload = {
        "version": "5.0.0",
        "flags": {"auto_preannotation": True},
        "shapes": shapes,
        "imagePath": source.name,
        "imageData": None,
        "imageHeight": int(image.shape[0]),
        "imageWidth": int(image.shape[1]),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Labelme resolves imagePath relative to the JSON, so keep an editable copy beside it.
    write_image(root / source.name, image)
    overlay_path = root / f"{source.stem}.prelabel.overlay.jpg"
    write_image(overlay_path, _draw_shapes_overlay(image, shapes))
    return LabelmePreannotationResult(json_path, overlay_path, len(shapes))


def labelme_json_to_instance_map(
    json_path: str | Path,
    output_dir: str | Path,
) -> LabelMapConversionResult:
    """Convert reviewed Labelme polygons to uint16 instance labels and metadata."""
    annotation_path = Path(json_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_path = annotation_path.parent / payload["imagePath"]
    image = read_image(image_path)
    expected = (int(payload.get("imageHeight", image.shape[0])), int(payload.get("imageWidth", image.shape[1])))
    if image.shape[:2] != expected:
        raise ValueError(f"Labelme dimensions {expected} do not match image {image.shape[:2]}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    label_map = np.zeros(image.shape[:2], np.uint16)
    accepted_shapes: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for instance_id, shape in enumerate(payload.get("shapes", []), 1):
        mask = _shape_mask(shape, image.shape[:2])
        if cv2.countNonZero(mask) < 4:
            continue
        # Later polygons take precedence, matching Labelme's visible z-order.
        label_map[mask > 0] = instance_id
        accepted_shapes.append(shape)
        objects.append(_measure_instance(mask, instance_id, shape.get("label", "workpiece")))

    label_map_path = root / f"{image_path.stem}.labels.png"
    ok, encoded = cv2.imencode(".png", label_map)
    if not ok:
        raise OSError(f"Could not encode label map: {label_map_path}")
    encoded.tofile(label_map_path)
    overlay_path = root / f"{image_path.stem}.labels.overlay.jpg"
    write_image(overlay_path, _draw_instance_overlay(image, label_map, objects))
    metadata_path = root / f"{image_path.stem}.labels.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_image": str(image_path),
                "source_annotation": str(annotation_path),
                "object_count": len(objects),
                "objects": objects,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return LabelMapConversionResult(label_map_path, overlay_path, metadata_path, len(objects))


def _shape_mask(shape: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, np.uint8)
    points = np.asarray(shape.get("points", []), np.float32)
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "polygon" and len(points) >= 3:
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    elif shape_type == "rectangle" and len(points) >= 2:
        first, second = np.round(points[:2]).astype(np.int32)
        cv2.rectangle(mask, tuple(first), tuple(second), 255, -1)
    elif shape_type == "circle" and len(points) >= 2:
        center = points[0]
        radius = int(round(float(np.linalg.norm(points[1] - center))))
        cv2.circle(mask, tuple(np.round(center).astype(np.int32)), radius, 255, -1)
    return mask


def _measure_instance(mask: np.ndarray, instance_id: int, label: str) -> dict[str, Any]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    center_x = float(moments["m10"] / moments["m00"])
    center_y_cv = float(moments["m01"] / moments["m00"])
    x, y, width, height = cv2.boundingRect(contour)
    angle, confidence = min_area_rect_orientation(contour)
    return {
        "object_id": instance_id,
        "label": label,
        "center": [center_x, float(mask.shape[0] - 1 - center_y_cv)],
        "axis_angle_deg": angle,
        "orientation_confidence": confidence,
        "bbox": [float(x), float(mask.shape[0] - (y + height)), float(width), float(height)],
        "area_px": int(cv2.countNonZero(mask)),
    }


def _draw_shapes_overlay(image: np.ndarray, shapes: list[dict[str, Any]]) -> np.ndarray:
    output = image.copy()
    layer = image.copy()
    for index, shape in enumerate(shapes, 1):
        points = np.round(np.asarray(shape["points"], np.float32)).astype(np.int32)
        color = _instance_color(index)
        cv2.fillPoly(layer, [points], color)
        cv2.polylines(output, [points], True, color, 3, cv2.LINE_AA)
        center = tuple(np.round(points.mean(axis=0)).astype(int))
        cv2.putText(output, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(output, 0.72, layer, 0.28, 0)


def _draw_instance_overlay(image: np.ndarray, label_map: np.ndarray, objects: list[dict[str, Any]]) -> np.ndarray:
    output = image.copy()
    layer = image.copy()
    for item in objects:
        instance_id = int(item["object_id"])
        mask = np.where(label_map == instance_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = _instance_color(instance_id)
        layer[mask > 0] = color
        cv2.drawContours(output, contours, -1, color, 3, cv2.LINE_AA)
        x = int(round(item["center"][0]))
        y = int(round(image.shape[0] - 1 - item["center"][1]))
        cv2.drawMarker(output, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(output, f"#{instance_id}", (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(output, 0.72, layer, 0.28, 0)


def _instance_color(instance_id: int) -> tuple[int, int, int]:
    hue = int((instance_id * 37) % 180)
    pixel = np.uint8([[[hue, 210, 245]]])
    return tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])
