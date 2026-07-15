import cv2
import numpy as np
from ..types import ImageResult

COLORS = [(46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15), (230, 126, 34), (231, 76, 60)]


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*label_map.shape, 3), np.uint8)
    for label in (v for v in np.unique(label_map) if v > 0):
        output[label_map == label] = COLORS[(int(label) - 1) % len(COLORS)]
    return output


def draw_overlay(image: np.ndarray, result: ImageResult) -> np.ndarray:
    canvas = image.copy()
    tint = canvas.copy()
    for obj in result.objects:
        color = COLORS[(obj.object_id - 1) % len(COLORS)]
        if obj._mask is not None: tint[obj._mask > 0] = color
    canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0)
    for obj in result.objects:
        color = COLORS[(obj.object_id - 1) % len(COLORS)]
        mask = obj._mask
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)
        box = np.round(np.asarray(obj.rotated_box_points)).astype(np.int32)
        cv2.polylines(canvas, [box], True, color, 1, cv2.LINE_AA)
        cx, cy = int(round(obj.center_x)), int(round(obj.center_y))
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        radians = np.radians(obj.angle_deg)
        half = max(15.0, obj.width_px / 2.0)
        dx, dy = int(np.cos(radians) * half), int(np.sin(radians) * half)
        cv2.line(canvas, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 255, 255), 2, cv2.LINE_AA)
        label = f"#{obj.object_id} {obj.angle_deg:.1f} deg" + ("" if obj.angle_reliable else " ?")
        cv2.putText(canvas, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    header = f"objects={result.object_count}  backend={result.backend}  time={result.processing_time_ms:.1f} ms"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 570), 28), (0, 0, 0), -1)
    cv2.putText(canvas, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
