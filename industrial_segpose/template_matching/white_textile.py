"""White-on-white cut-seam enhancement and enclosed-part segmentation.

The workpiece may still lie inside the source sheet, so foreground colour is
not a useful cue.  This module treats the dark/shadowed cut seam as the signal,
normalises slow illumination changes, reconnects short seam gaps and extracts
regions enclosed by the resulting barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WhiteTextileParameters:
    """Scale-independent parameters for desktop white textile processing."""

    max_processing_edge: int = 1600
    illumination_sigma_ratio: float = 0.055
    seam_percentile: float = 88.0
    close_gap_ratio: float = 0.010
    barrier_width_ratio: float = 0.004
    minimum_area_ratio: float = 0.008
    maximum_area_ratio: float = 0.90

    def validate(self) -> None:
        if self.max_processing_edge < 128:
            raise ValueError("max_processing_edge must be at least 128")
        if not 0.005 <= self.illumination_sigma_ratio <= 0.25:
            raise ValueError("illumination_sigma_ratio must be in [0.005, 0.25]")
        if not 50.0 <= self.seam_percentile <= 99.9:
            raise ValueError("seam_percentile must be in [50, 99.9]")
        if not 0.0 < self.close_gap_ratio <= 0.10:
            raise ValueError("close_gap_ratio must be in (0, 0.10]")
        if not 0.0 < self.barrier_width_ratio <= 0.05:
            raise ValueError("barrier_width_ratio must be in (0, 0.05]")
        if not 0.0 < self.minimum_area_ratio < self.maximum_area_ratio <= 1.0:
            raise ValueError("Invalid candidate area ratio range")


@dataclass(frozen=True)
class WhiteTextileResult:
    """All intermediate images needed for tuning and visual diagnostics."""

    mask: np.ndarray
    normalized: np.ndarray
    seam_response: np.ndarray
    seam_binary: np.ndarray
    closed_boundary: np.ndarray
    candidate_count: int
    coverage: float
    quality_score: float
    processing_scale: float
    candidate_masks: tuple[np.ndarray, ...]


def _odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size + 1 if size % 2 == 0 else size


def _resize_for_processing(image: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale >= 0.999:
        return image.copy(), 1.0
    size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def _remove_small_components(binary: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == label] = 255
    return output


def white_cut_seam_feature(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return illumination-normalised gray and a soft dark-seam response."""

    if image is None or image.size == 0:
        raise ValueError("White textile image is empty")
    if image.ndim == 2:
        gray = image.astype(np.uint8, copy=True)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("White textile image must be gray or BGR")

    gray = cv2.medianBlur(gray, 3)
    minimum_edge = min(gray.shape[:2])
    sigma = max(3.0, minimum_edge * 0.055)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    # Ratio normalisation preserves narrow seams while cancelling gradients.
    normalised = cv2.divide(gray, np.maximum(background, 1), scale=128.0)
    normalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalised)

    responses: list[np.ndarray] = []
    for ratio in (0.008, 0.016, 0.030):
        size = _odd_size(minimum_edge * ratio, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(cv2.morphologyEx(normalised, cv2.MORPH_BLACKHAT, kernel))
    blackhat = np.maximum.reduce(responses)

    gradient_x = cv2.Sobel(normalised, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalised, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Blackhat dominates; gradient restores weak seam portions with little
    # shadow but a measurable intensity transition.
    response = cv2.addWeighted(blackhat, 0.78, gradient, 0.35, 0.0)
    response = cv2.GaussianBlur(response, (3, 3), 0)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalised, response


def contour_template_feature(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a soft contour template and its narrow matching support band."""

    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if binary.ndim != 2 or int(np.count_nonzero(binary)) < 25:
        raise ValueError("White textile template mask is empty")
    base = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    radius = _odd_size(min(binary.shape) * 0.025, 5)
    band = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    distance = cv2.distanceTransform(255 - base, cv2.DIST_L2, 3)
    sigma = max(1.0, radius / 4.0)
    feature = np.exp(-0.5 * (distance / sigma) ** 2) * 255.0
    feature[band == 0] = 0.0
    return feature.astype(np.uint8), band


def _enclosed_candidates(
    boundary: np.ndarray,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> list[np.ndarray]:
    """Find free-space components isolated from the image border by a seam."""

    free = np.where(boundary > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    image_area = float(free.size)
    candidates: list[np.ndarray] = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if x == 0 or y == 0 or x + width >= free.shape[1] or y + height >= free.shape[0]:
            continue
        ratio = area / image_area
        if minimum_area_ratio <= ratio <= maximum_area_ratio:
            candidate = np.where(labels == label, 255, 0).astype(np.uint8)
            candidates.append(candidate)
    return candidates


def _candidate_score(mask: np.ndarray, seam_binary: np.ndarray) -> float:
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    seam_near = cv2.dilate(seam_binary, np.ones((5, 5), np.uint8))
    support = float(np.count_nonzero((boundary > 0) & (seam_near > 0))) / max(
        float(np.count_nonzero(boundary)), 1.0
    )
    coverage = float(np.count_nonzero(mask)) / mask.size
    coverage_score = min(1.0, coverage / 0.08) * min(1.0, 0.75 / max(coverage, 1e-6))
    points = cv2.findNonZero(mask)
    if points is None:
        return 0.0
    x, y, width, height = cv2.boundingRect(points)
    margin = min(x, y, mask.shape[1] - x - width, mask.shape[0] - y - height)
    border_score = min(1.0, margin / max(0.03 * min(mask.shape), 1.0))
    return float(np.clip(0.68 * support + 0.20 * coverage_score + 0.12 * border_score, 0.0, 1.0))


def _segment_with_hypothesis(
    response: np.ndarray,
    seam_percentile: float,
    close_gap_ratio: float,
    barrier_width_ratio: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]:
    """Build one seam-closure hypothesis and return its restored regions.

    A clean cut is handled by the conservative hypothesis.  Real textile cuts
    often contain short invisible sections caused by glare, loose fibres or a
    shallow knife path; a second hypothesis may therefore use a stronger
    closing kernel without changing the feature extractor itself.
    """

    nonzero = response[response > 0]
    percentile = float(np.percentile(nonzero, seam_percentile)) if nonzero.size else 255.0
    otsu_threshold, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = max(12.0, min(percentile, float(otsu_threshold) * 1.15))
    seam = np.where(response >= threshold, 255, 0).astype(np.uint8)
    minimum_edge = min(seam.shape)
    seam = _remove_small_components(seam, max(4, int(round(minimum_edge * 0.008))))

    close_size = _odd_size(minimum_edge * close_gap_ratio, 3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(seam, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    barrier_size = _odd_size(minimum_edge * barrier_width_ratio, 3)
    barrier = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (barrier_size, barrier_size)),
        iterations=1,
    )

    scored: list[tuple[float, np.ndarray]] = []
    for candidate in _enclosed_candidates(barrier, minimum_area_ratio, maximum_area_ratio):
        score = _candidate_score(candidate, seam)
        restored = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Restore roughly half of the thick virtual barrier so that the final
        # mask follows the physical cut centreline instead of shrinking inward.
        recovery_size = _odd_size(close_size + barrier_size, 3)
        restored = cv2.dilate(
            restored,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (recovery_size, recovery_size)),
            iterations=1,
        )
        scored.append((score, restored))
    return seam, barrier, scored


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_selected = first > 0
    second_selected = second > 0
    union = int(np.count_nonzero(first_selected | second_selected))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first_selected & second_selected) / union)


def segment_white_textile(
    image: np.ndarray,
    parameters: WhiteTextileParameters | None = None,
) -> WhiteTextileResult:
    """Segment the dominant white cut part enclosed by a visible seam."""

    params = parameters or WhiteTextileParameters()
    params.validate()
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("White textile segmentation input must be a non-empty image")
    processed, scale = _resize_for_processing(image, params.max_processing_edge)
    normalized, response = white_cut_seam_feature(processed)

    seam, barrier, primary = _segment_with_hypothesis(
        response,
        params.seam_percentile,
        params.close_gap_ratio,
        params.barrier_width_ratio,
        params.minimum_area_ratio,
        params.maximum_area_ratio,
    )
    hypotheses: list[tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]] = [
        (seam, barrier, primary)
    ]
    primary_quality = max((item[0] for item in primary), default=0.0)
    if not primary or primary_quality < 0.58:
        # Recovery for interrupted real-world seams.  The higher percentile
        # suppresses wrinkles while the larger closing radius bridges short
        # missing cut sections.  It is deliberately conditional so clean
        # images retain the more precise conservative mask.
        recovery = _segment_with_hypothesis(
            response,
            max(params.seam_percentile, 92.0),
            max(params.close_gap_ratio, 0.030),
            max(params.barrier_width_ratio, 0.005),
            min(params.minimum_area_ratio, 0.002),
            params.maximum_area_ratio,
        )
        hypotheses.append(recovery)

    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for hypothesis_seam, hypothesis_barrier, hypothesis_candidates in hypotheses:
        for score, candidate in hypothesis_candidates:
            duplicate_index = next(
                (index for index, item in enumerate(scored) if _mask_iou(candidate, item[1]) >= 0.65),
                None,
            )
            record = (score, candidate, hypothesis_seam, hypothesis_barrier)
            if duplicate_index is None:
                scored.append(record)
            elif score > scored[duplicate_index][0]:
                scored[duplicate_index] = record

    if scored:
        quality, mask, seam, barrier = max(scored, key=lambda item: item[0])
    else:
        quality = 0.0
        mask = np.zeros(seam.shape, np.uint8)
    restored_candidates = [item[1] for item in sorted(scored, key=lambda item: item[0], reverse=True)]

    original_size = (image.shape[1], image.shape[0])
    if scale < 0.999:
        normalized = cv2.resize(normalized, original_size, interpolation=cv2.INTER_LINEAR)
        response = cv2.resize(response, original_size, interpolation=cv2.INTER_LINEAR)
        seam = cv2.resize(seam, original_size, interpolation=cv2.INTER_NEAREST)
        barrier = cv2.resize(barrier, original_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        restored_candidates = [
            cv2.resize(candidate, original_size, interpolation=cv2.INTER_NEAREST)
            for candidate in restored_candidates
        ]
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    restored_candidates = [
        np.where(candidate > 0, 255, 0).astype(np.uint8)
        for candidate in restored_candidates
    ]
    coverage = float(np.count_nonzero(mask) / mask.size)
    return WhiteTextileResult(
        mask=mask,
        normalized=normalized,
        seam_response=response,
        seam_binary=seam,
        closed_boundary=barrier,
        candidate_count=len(restored_candidates),
        coverage=coverage,
        quality_score=float(quality),
        processing_scale=scale,
        candidate_masks=tuple(restored_candidates),
    )


def save_white_textile_diagnostics(
    result: WhiteTextileResult,
    output_directory: str | Path,
    original_image: np.ndarray | None = None,
) -> Path:
    """Save deterministic intermediate PNG files and a compact JSON summary."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    images = {
        "02_illumination_normalized.png": result.normalized,
        "03_seam_response.png": result.seam_response,
        "04_seam_binary.png": result.seam_binary,
        "05_closed_boundary.png": result.closed_boundary,
        "06_segmented_mask.png": result.mask,
    }
    if original_image is not None:
        images = {"01_original.png": original_image, **images}
    for name, image in images.items():
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise OSError(f"Failed to encode diagnostic image: {name}")
        (destination / name).write_bytes(encoded.tobytes())
    summary = {
        "candidate_count": result.candidate_count,
        "coverage": result.coverage,
        "quality_score": result.quality_score,
        "processing_scale": result.processing_scale,
    }
    (destination / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
