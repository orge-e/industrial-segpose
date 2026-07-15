"""Build reproducible before/after visual comparisons for noise robustness.

The "before" side is an ablation baseline representing the earlier generic
pipeline: fixed pink-colour thresholds and absolute (not relative) dark-textile
texture energy.  The "after" side runs the production multi-template matcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.io.image_reader import read_image, write_image
from industrial_segpose.template_matching import (
    MultiTemplateMatcher,
    TemplateLibrary,
    TemplateMatcher,
    draw_multi_template_matches,
)
import industrial_segpose.template_matching.multi_matcher as multi_matcher_module


DEFAULT_IMAGES = {
    "mixed": "bc735d748e3613e1b10ee7ff4b250608.jpg",
    "pink": "40f9a5a94a294c22bd41cfbebfd8e27f.jpg",
    "black": "36b489842d2e394c7f1d2c62a091f15c.jpg",
}

EXPECTED_COUNTS = {
    "mixed": {"t1": 1, "Black_Textile_01": 1},
    "pink": {"t1": 1, "Black_Textile_01": 0},
    "black": {"t1": 0, "Black_Textile_01": 1},
}


def _legacy_dark_textile_candidates(image: np.ndarray) -> np.ndarray:
    """Absolute high-frequency baseline used before exposure normalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 2.0)
    texture = cv2.GaussianBlur(np.abs(gray - local), (0, 0), 4.0)
    enhanced = np.clip(texture * 7.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = np.where(enhanced >= max(24.0, float(threshold)), 255, 0).astype(np.uint8)
    size = max(5, int(round(min(image.shape[:2]) * 0.0075)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


class LegacyTemplateMatcher(TemplateMatcher):
    """A deliberately unnormalized matcher used only for the comparison."""

    def _pose_candidate_mask(self, image: np.ndarray) -> np.ndarray:
        if self.parameters.feature_mode == "dark_textile":
            return _legacy_dark_textile_candidates(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Fixed thresholds work in nominal lighting, but are intentionally not
        # adapted to the current scene exposure or the stored template colour.
        raw = (
            (lab[:, :, 1] >= 130)
            & (hsv[:, :, 1] >= 18)
            & (hsv[:, :, 2] >= 55)
        )
        binary = np.where(raw, 255, 0).astype(np.uint8)
        size = max(3, int(round(min(image.shape[:2]) * 0.0035)))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _run_matcher(library: TemplateLibrary, image: np.ndarray, legacy: bool):
    original = multi_matcher_module.TemplateMatcher
    try:
        if legacy:
            multi_matcher_module.TemplateMatcher = LegacyTemplateMatcher
        matcher = MultiTemplateMatcher.from_library(library)
        started = time.perf_counter()
        result = matcher.match(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result, elapsed_ms
    finally:
        multi_matcher_module.TemplateMatcher = original


def _exposure(image: np.ndarray, gain: float, bias: float = 0.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _uneven_light(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    horizontal = np.linspace(0.52, 1.48, width, dtype=np.float32)
    vertical = np.linspace(1.12, 0.88, height, dtype=np.float32)[:, None]
    gain = np.clip(vertical * horizontal[None, :], 0.48, 1.55)
    return np.clip(image.astype(np.float32) * gain[:, :, None] + 4.0, 0, 255).astype(np.uint8)


def _impulse_noise(image: np.ndarray, seed: int) -> np.ndarray:
    noisy = image.copy()
    rng = np.random.default_rng(seed)
    count = max(1200, int(image.shape[0] * image.shape[1] * 0.015))
    ys = rng.integers(0, image.shape[0], count)
    xs = rng.integers(0, image.shape[1], count)
    values = rng.choice(np.asarray([0, 255], dtype=np.uint8), count)
    noisy[ys, xs] = values[:, None]
    return noisy


def build_conditions(image: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    return {
        "01_original": image,
        "02_underexposed": _exposure(image, 0.55),
        "03_overexposed": _exposure(image, 1.45, 12.0),
        "04_uneven_light": _uneven_light(image),
        "05_impulse_noise": _impulse_noise(image, seed),
    }


def _fit_panel(image: np.ndarray, width: int = 720, height: int = 520) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 26, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _titled_panel(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    panel = _fit_panel(image)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 66), (15, 22, 34), -1)
    cv2.putText(panel, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, detail, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (198, 213, 232), 1, cv2.LINE_AA)
    return panel


def _result_detail(result, elapsed_ms: float) -> str:
    counts = ", ".join(f"{name}:{count}" for name, count in result.counts_by_template.items())
    return f"total={result.object_count}  ambiguous={result.ambiguous_count}  {counts}  {elapsed_ms:.0f} ms"


def _save_case(
    output: Path,
    source_name: str,
    condition: str,
    image: np.ndarray,
    before_result,
    before_ms: float,
    after_result,
    after_ms: float,
) -> Path:
    input_panel = _titled_panel(image, "SAME INPUT", condition)
    before_panel = _titled_panel(
        draw_multi_template_matches(image, before_result),
        "BEFORE - fixed threshold baseline",
        _result_detail(before_result, before_ms),
    )
    after_panel = _titled_panel(
        draw_multi_template_matches(image, after_result),
        "AFTER - robust production algorithm",
        _result_detail(after_result, after_ms),
    )
    comparison = np.hstack((input_panel, before_panel, after_panel))
    path = output / "cases" / f"{source_name}_{condition}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, comparison)
    return path


def _contact_sheet(paths: list[Path], output_path: Path) -> None:
    rows = [read_image(path) for path in paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory containing the three source images listed in DEFAULT_IMAGES.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output = (args.output or project_root / "outputs" / "noise_robustness_comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = TemplateLibrary(project_root / "templates").load()

    rows: list[dict] = []
    contact_sheets: dict[str, list[Path]] = {}
    for source_index, (source_name, filename) in enumerate(DEFAULT_IMAGES.items(), 1):
        source_path = args.image_root / filename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        original = read_image(source_path)
        contact_sheets[source_name] = []
        for condition, image in build_conditions(original, seed=20260715 + source_index).items():
            before_result, before_ms = _run_matcher(library, image, legacy=True)
            after_result, after_ms = _run_matcher(library, image, legacy=False)
            path = _save_case(
                output,
                source_name,
                condition,
                image,
                before_result,
                before_ms,
                after_result,
                after_ms,
            )
            contact_sheets[source_name].append(path)
            rows.append({
                "source": source_name,
                "condition": condition,
                "before_correct": before_result.counts_by_template == EXPECTED_COUNTS[source_name],
                "after_correct": after_result.counts_by_template == EXPECTED_COUNTS[source_name],
                "before_count": before_result.object_count,
                "after_count": after_result.object_count,
                "before_ambiguous": before_result.ambiguous_count,
                "after_ambiguous": after_result.ambiguous_count,
                "before_counts": json.dumps(before_result.counts_by_template, ensure_ascii=False),
                "after_counts": json.dumps(after_result.counts_by_template, ensure_ascii=False),
                "before_mean_score": round(float(np.mean([item.score for item in before_result.objects])), 4) if before_result.objects else "",
                "after_mean_score": round(float(np.mean([item.score for item in after_result.objects])), 4) if after_result.objects else "",
                "before_ms": round(before_ms, 2),
                "after_ms": round(after_ms, 2),
                "comparison_image": str(path),
            })

    for source_name, paths in contact_sheets.items():
        _contact_sheet(paths, output / f"{source_name}_all_conditions.jpg")

    csv_path = output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({
            "baseline": "fixed-threshold ablation",
            "case_count": len(rows),
            "before_correct_cases": sum(bool(row["before_correct"]) for row in rows),
            "after_correct_cases": sum(bool(row["after_correct"]) for row in rows),
            "cases": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    print(f"Generated {len(rows)} before/after cases")


if __name__ == "__main__":
    main()
