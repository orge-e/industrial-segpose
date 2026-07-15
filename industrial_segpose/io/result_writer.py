"""Run directory creation and JSON/CSV/image exports."""

import csv
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from .image_reader import write_image
from ..types import ImageResult
from ..visualization import colorize_labels, draw_overlay

CSV_FIELDS = ["image_name", "object_id", "center_x", "center_y", "angle_deg", "angle_method", "angle_reliable", "orientation_confidence", "area_px", "perimeter_px", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "rotated_width_px", "rotated_height_px", "aspect_ratio", "touches_border", "processing_time_ms"]


class ResultWriter:
    def __init__(self, output_root: str | Path, config: dict):
        self.config = config
        self.run_dir = Path(output_root) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        for name in ("annotated", "masks", "labels", "debug", "json", "csv"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def write_image_result(self, image: np.ndarray, result: ImageResult, label_map: np.ndarray, debug_images: dict[str, np.ndarray]) -> None:
        # Preserve source identity without creating nested output trees. This
        # prevents recursive batches with repeated basenames from overwriting.
        source = Path(result.image_name)
        stem = "__".join(source.with_suffix("").parts)
        stem = stem.replace(":", "_")
        cfg = self.config["output"]
        if cfg["save_masks"]:
            for obj in result.objects:
                filename = f"{stem}_object_{obj.object_id:03d}.png"
                write_image(self.run_dir / "masks" / filename, obj._mask)
                obj.mask_filename = filename
        if cfg["save_annotated"]: write_image(self.run_dir / "annotated" / f"{stem}_annotated.png", draw_overlay(image, result))
        if cfg["save_label_map"]: write_image(self.run_dir / "labels" / f"{stem}_labels.png", colorize_labels(label_map))
        if cfg["save_debug"]:
            for name, debug in debug_images.items():
                if debug.dtype not in (np.uint8, np.uint16): debug = cv2.normalize(debug, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                write_image(self.run_dir / "debug" / f"{stem}_{name}.png", debug)
        if cfg["save_json"]:
            (self.run_dir / "json" / f"{stem}.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        for obj in result.objects:
            x, y, width, height = obj.bbox_xywh
            self.rows.append({"image_name": result.image_name, "object_id": obj.object_id, "center_x": obj.center_x, "center_y": obj.center_y, "angle_deg": obj.angle_deg, "angle_method": obj.angle_method, "angle_reliable": obj.angle_reliable, "orientation_confidence": obj.orientation_confidence, "area_px": obj.area_px, "perimeter_px": obj.perimeter_px, "bbox_x": x, "bbox_y": y, "bbox_width": width, "bbox_height": height, "rotated_width_px": obj.width_px, "rotated_height_px": obj.height_px, "aspect_ratio": obj.aspect_ratio, "touches_border": obj.touches_border, "processing_time_ms": result.processing_time_ms})

    def finalize(self, results: list[ImageResult], wall_time_ms: float) -> None:
        if self.config["output"]["save_csv"]:
            with (self.run_dir / "csv" / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(self.rows)
        summary = {"images_total": len(results), "images_succeeded": sum(r.success for r in results), "images_failed": sum(not r.success for r in results), "objects_total": sum(r.object_count for r in results), "wall_time_ms": wall_time_ms, "backend": self.config["segmentation"]["backend"], "results": [{"image_name": r.image_name, "success": r.success, "object_count": r.object_count, "error_message": r.error_message} for r in results]}
        (self.run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
