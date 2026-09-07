"""Generate assisted-segmentation diagnostics for real field captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.template_matching.mask_assist import METHOD_LABELS, build_assisted_mask


def _read(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def _panel(image: np.ndarray, mask: np.ndarray, label: str, quality: float, coverage: float) -> np.ndarray:
    height, width = image.shape[:2]
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255
    selected = mask > 0
    if np.any(selected):
        overlay[selected] = cv2.addWeighted(image[selected], 0.45, green[selected], 0.55, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), max(2, round(min(height, width) / 240)))
    banner = max(34, round(height * 0.08))
    canvas = np.zeros((height + banner, width, 3), np.uint8)
    canvas[banner:] = overlay
    cv2.putText(
        canvas,
        f"{label}  quality={quality:.3f} coverage={coverage:.1%}",
        (10, round(banner * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(width, height) / 900),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*", help="Only process files whose name contains this text")
    parser.add_argument("--skip-auto", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        path for path in args.input.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and args.pattern in str(path)
    )
    if not paths:
        raise SystemExit("No images found")

    methods = [name for name in METHOD_LABELS if name not in {"grabcut"}]
    if args.skip_auto:
        methods.remove("auto")
    summary: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        image = _read(path)
        scale = min(1.0, 640.0 / max(image.shape[:2]))
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panels = [_panel(preview, np.zeros(preview.shape[:2], np.uint8), "original", 0.0, 0.0)]
        item = {"image": str(path), "shape": list(image.shape), "methods": {}}
        for method in methods:
            result = build_assisted_mask(image, method)
            mask = cv2.resize(result.mask, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)
            panels.append(_panel(preview, mask, method, result.quality_score, result.coverage))
            item["methods"][method] = {
                "selected_method": result.method,
                "quality": result.quality_score,
                "coverage": result.coverage,
            }
        target_width = 480
        panels = [cv2.resize(panel, (target_width, round(panel.shape[0] * target_width / panel.shape[1]))) for panel in panels]
        rows = []
        for start in range(0, len(panels), 3):
            row = panels[start:start + 3]
            while len(row) < 3:
                row.append(np.zeros_like(panels[0]))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        _write(args.output / f"{index:02d}_{path.stem}_methods.jpg", sheet)
        summary.append(item)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(paths)} diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
