"""Classical image-processing helpers for assisted irregular template masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .white_textile import segment_white_textile


METHOD_LABELS = {
    "auto": "智能组合",
    "textile_chroma": "纺织色度分割",
    "dark_textile": "暗色纹理分割",
    "white_cut_seam": "白色裁剪缝分割",
    "grabcut": "GrabCut",
    "border_color": "边界颜色差",
    "detached_textile": "独立工件背景建模",
    "otsu_light": "亮目标 Otsu",
    "otsu_dark": "暗目标 Otsu",
}


@dataclass(frozen=True)
class AssistedMaskResult:
    mask: np.ndarray
    method: str
    method_label: str
    coverage: float
    quality_score: float


@dataclass(frozen=True)
class AutomaticTemplateSelection:
    roi_xywh: tuple[int, int, int, int]
    mask: np.ndarray
    method: str
    method_label: str
    full_image_coverage: float
    quality_score: float


@dataclass(frozen=True)
class TemplateMaskQuality:
    valid: bool
    coverage: float
    component_count: int
    main_component_ratio: float
    touches_border: bool
    message: str


def analyze_template_mask(mask: np.ndarray) -> TemplateMaskQuality:
    """Evaluate whether a manually or automatically prepared mask is usable."""

    candidate = np.asarray(mask)
    if candidate.ndim == 3:
        candidate = candidate[:, :, 0]
    if candidate.ndim != 2 or candidate.size == 0:
        return TemplateMaskQuality(False, 0.0, 0, 0.0, False, "Mask为空或尺寸无效")
    binary = np.where(candidate > 0, 255, 0).astype(np.uint8)
    foreground = int(np.count_nonzero(binary))
    coverage = foreground / float(binary.size)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 25),
        reverse=True,
    )
    component_count = len(areas)
    main_ratio = (areas[0] / foreground) if areas and foreground else 0.0
    touches_border = bool(
        np.any(binary[0]) or np.any(binary[-1]) or np.any(binary[:, 0]) or np.any(binary[:, -1])
    )
    issues: list[str] = []
    if foreground < 25:
        issues.append("有效像素过少")
    if coverage < 0.02:
        issues.append("目标覆盖率过低")
    elif coverage > 0.95:
        issues.append("Mask几乎覆盖整个ROI")
    if component_count > 1 and main_ratio < 0.92:
        issues.append(f"存在{component_count}个明显分离区域")
    if touches_border:
        issues.append("目标接触ROI边界")
    valid = (
        foreground >= 25
        and 0.02 <= coverage <= 0.95
        and (component_count <= 1 or main_ratio >= 0.92)
        and not touches_border
    )
    if issues:
        message = "；".join(issues)
    else:
        message = f"Mask质量正常，覆盖率{coverage:.1%}，主体占比{main_ratio:.1%}"
    return TemplateMaskQuality(valid, coverage, component_count, main_ratio, touches_border, message)


def _kernel(image: np.ndarray) -> np.ndarray:
    size = max(3, int(round(min(image.shape[:2]) * 0.025)))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _main_component(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = _kernel(image)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    height, width = binary.shape
    image_center = np.asarray([width / 2.0, height / 2.0])
    diagonal = max(float(np.hypot(width, height)), 1.0)
    best_label, best_score = 0, -1.0
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        distance = float(np.linalg.norm(centroids[label] - image_center)) / diagonal
        centrality = max(0.15, 1.0 - 1.8 * distance)
        component = labels == label
        border_pixels = int(np.count_nonzero(component[0]) + np.count_nonzero(component[-1]) + np.count_nonzero(component[:, 0]) + np.count_nonzero(component[:, -1]))
        border_penalty = 0.55 if border_pixels > 0 else 1.0
        score = area * centrality * border_penalty
        if score > best_score:
            best_label, best_score = label, score
    output = np.where(labels == best_label, 255, 0).astype(np.uint8) if best_label else np.zeros_like(binary)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        output.fill(0)
        cv2.drawContours(output, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return output


def _grabcut(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_x = max(2, int(round(width * 0.04)))
    margin_y = max(2, int(round(height * 0.04)))
    if width - 2 * margin_x < 2 or height - 2 * margin_y < 2:
        return np.zeros((height, width), np.uint8)
    labels = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, labels, (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y), background, foreground, 4, cv2.GC_INIT_WITH_RECT)
    return np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _border_color(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    strip = max(1, int(round(min(height, width) * 0.06)))
    border = np.concatenate(
        [lab[:strip].reshape(-1, 3), lab[-strip:].reshape(-1, 3), lab[:, :strip].reshape(-1, 3), lab[:, -strip:].reshape(-1, 3)],
        axis=0,
    )
    background_color = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background_color, axis=2)
    normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized = cv2.GaussianBlur(normalized, (5, 5), 0)
    _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def extract_detached_textile_candidates(image: np.ndarray) -> np.ndarray:
    """Segment a detached textile part from a mostly smooth support surface.

    A linear Lab background model is fitted from robust border samples.  This
    cancels the broad exposure gradient visible in real field captures. A weak
    response is then grown from reliable seeds so wrinkles do not split a pale
    target or leave only its brightest half.
    """

    lab = cv2.cvtColor(cv2.medianBlur(image, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    strip = max(4, int(round(min(height, width) * 0.075)))
    border_mask = np.zeros((height, width), np.uint8)
    border_mask[:strip] = 1
    border_mask[-strip:] = 1
    border_mask[:, :strip] = 1
    border_mask[:, -strip:] = 1

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx - width * 0.5) / max(float(width), 1.0)
    ny = (yy - height * 0.5) / max(float(height), 1.0)
    border = border_mask > 0
    design = np.column_stack((np.ones(int(np.count_nonzero(border))), nx[border], ny[border]))
    samples = lab[border]

    median = np.median(samples, axis=0)
    mad = np.maximum(np.median(np.abs(samples - median), axis=0) * 1.4826, (5.0, 2.5, 2.5))
    inlier = np.sqrt(np.sum(((samples - median) / mad) ** 2, axis=1)) <= 3.5
    fit_design, fit_samples = (design[inlier], samples[inlier]) if int(np.count_nonzero(inlier)) >= 30 else (design, samples)
    coefficients, *_ = np.linalg.lstsq(fit_design, fit_samples, rcond=None)
    full_design = np.stack((np.ones_like(nx), nx, ny), axis=2)
    background = full_design @ coefficients
    residual = lab - background

    border_residual = residual[border]
    scale = np.maximum(np.median(np.abs(border_residual), axis=0) * 1.4826, (5.5, 2.8, 2.8))
    distance = np.sqrt(np.sum((residual / scale) ** 2, axis=2))
    distance = cv2.GaussianBlur(distance.astype(np.float32), (0, 0), 1.5)
    upper = max(float(np.percentile(distance, 99.5)), 1.0)
    response = np.clip(distance / upper * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    strong_threshold = max(28.0, float(otsu))
    weak_threshold = max(15.0, strong_threshold * 0.48)
    strong = response >= strong_threshold
    weak = np.where(response >= weak_threshold, 255, 0).astype(np.uint8)

    size = max(3, int(round(min(height, width) * 0.006)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    output = np.zeros((height, width), np.uint8)
    center = np.asarray([width * 0.5, height * 0.5])
    diagonal = max(float(np.hypot(width, height)), 1.0)
    ranked: list[tuple[float, int]] = []
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(25, int(height * width * 0.002)) or not np.any(strong & component):
            continue
        distance_to_center = float(np.linalg.norm(centroids[label] - center)) / diagonal
        x, y, component_width, component_height = map(int, stats[label, :4])
        touches_border = x == 0 or y == 0 or x + component_width == width or y + component_height == height
        score = area * max(0.2, 1.0 - 1.8 * distance_to_center) * (0.35 if touches_border else 1.0)
        ranked.append((score, label))
    if ranked:
        output[labels == max(ranked)[1]] = 255
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=2)
    return output


def extract_textile_chroma_candidates(image: np.ndarray) -> np.ndarray:
    """Extract lightly colored textile from a neutral/gray background."""
    lab = cv2.cvtColor(cv2.medianBlur(image, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    # A single a-channel threshold has an arbitrary direction: for pale yellow
    # parts it selects nearly the entire neutral table.  Euclidean chroma is
    # direction-independent and therefore works for pink, yellow, blue, etc.
    chroma = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)
    chroma = cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 2.0)
    upper = max(float(np.percentile(chroma, 99.5)), 1.0)
    response = np.clip(chroma / upper * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # A small absolute floor rejects low-level colour casts from auto white balance
    # sensor while retaining low-saturation textile under weak exposure.
    absolute_floor = min(255.0, 3.0 / upper * 255.0)
    combined = np.where(response >= max(float(threshold), absolute_floor), 255, 0).astype(np.uint8)
    kernel_size = max(5, int(round(min(image.shape[:2]) * 0.02)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


def extract_dark_textile_candidates(image: np.ndarray) -> np.ndarray:
    """Extract textured dark fabric from a smoother dark conveyor surface.

    The signal is local high-frequency energy rather than absolute brightness,
    so uneven illumination and black-on-black scenes remain usable.  All large
    candidate regions are retained here; callers may select one component or
    compare every component with a template.
    """
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray_u8, 3)
    # Replace only strong impulses.  Ordinary weave contrast is retained while
    # hot pixels, dust highlights and isolated dark points are suppressed.
    cleaned = np.where(cv2.absdiff(gray_u8, median) > 28, median, gray_u8).astype(np.float32)
    local_illumination = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
    # Relative local contrast is substantially more stable than raw intensity
    # under exposure drift, shadows and illumination gradients.
    high_frequency = np.abs(cleaned - local_illumination) / (local_illumination + 18.0) * 110.0
    texture_energy = cv2.GaussianBlur(high_frequency, (0, 0), 4.0)
    enhanced = np.clip(texture_energy * 7.0, 0, 255).astype(np.uint8)
    threshold, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Very dark low-noise scenes can produce an unrealistically small Otsu
    # threshold.  A small floor avoids turning the conveyor into foreground.
    binary = np.where(enhanced >= max(24.0, float(threshold)), 255, 0).astype(np.uint8)
    size = max(5, int(round(min(image.shape[:2]) * 0.0075)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Weave energy is sparse inside very dark fabric.  Filling each external
    # response contour restores the physical silhouette while keeping separate
    # workpieces as separate connected components.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    minimum_contour_area = max(20.0, image.shape[0] * image.shape[1] * 0.001)
    for contour in contours:
        if cv2.contourArea(contour) >= minimum_contour_area:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    return filled


def _otsu(image: np.ndarray, inverse: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return mask


def _quality(image: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0
    coverage = float(np.count_nonzero(selected) / selected.size)
    if coverage < 0.015 or coverage > 0.97:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = gray[selected]
    outside = gray[~selected]
    contrast = abs(float(inside.mean()) - float(outside.mean())) / max(float(gray.std()) * 2.0, 1.0) if inside.size and outside.size else 0.0
    contrast = min(1.0, contrast)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edges = cv2.dilate(cv2.Canny(gray.astype(np.uint8), 50, 150), np.ones((3, 3), np.uint8)) > 0
    alignment = float(np.count_nonzero(boundary & edges) / max(np.count_nonzero(boundary), 1))
    perimeter_selected = int(np.count_nonzero(selected[0]) + np.count_nonzero(selected[-1]) + np.count_nonzero(selected[:, 0]) + np.count_nonzero(selected[:, -1]))
    perimeter_total = max(2 * image.shape[0] + 2 * image.shape[1], 1)
    border_penalty = min(0.45, perimeter_selected / perimeter_total)
    coverage_score = 1.0 if 0.05 <= coverage <= 0.85 else 0.55
    return max(0.0, 0.42 * alignment + 0.38 * contrast + 0.20 * coverage_score - border_penalty)


def _candidate(image: np.ndarray, method: str) -> AssistedMaskResult:
    method_quality: float | None = None
    if method == "grabcut":
        raw = _grabcut(image)
    elif method == "textile_chroma":
        raw = extract_textile_chroma_candidates(image)
    elif method == "dark_textile":
        raw = extract_dark_textile_candidates(image)
    elif method == "white_cut_seam":
        white_result = segment_white_textile(image)
        raw = white_result.mask
        method_quality = white_result.quality_score
    elif method == "border_color":
        raw = _border_color(image)
    elif method == "detached_textile":
        raw = extract_detached_textile_candidates(image)
    elif method == "otsu_light":
        raw = _otsu(image, False)
    elif method == "otsu_dark":
        raw = _otsu(image, True)
    else:
        raise ValueError(f"Unknown assisted mask method: {method}")
    mask = _main_component(raw, image)
    coverage = float(np.count_nonzero(mask) / mask.size)
    quality = _quality(image, mask)
    if method_quality is not None:
        quality = max(quality, method_quality)
    return AssistedMaskResult(mask, method, METHOD_LABELS[method], coverage, quality)


def build_assisted_mask(image: np.ndarray, method: str = "auto") -> AssistedMaskResult:
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("Assisted mask input must be a non-empty BGR image")
    if min(image.shape[:2]) < 10:
        raise ValueError("Assisted mask image must be at least 10 x 10 pixels")
    if method != "auto":
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown assisted mask method: {method}")
        return _candidate(image, method)
    methods = [
        "textile_chroma", "dark_textile", "white_cut_seam",
        "detached_textile", "border_color", "otsu_light", "otsu_dark",
    ]
    # GrabCut remains available as an explicit manual option, but is not part
    # of automatic competition. Four iterations were taking seconds per compact-camera
    # frame and could freeze the template-building UI.
    results = [_candidate(image, name) for name in methods]
    best = max(results, key=lambda item: item.quality_score)
    textile = results[0]
    white_cut = results[2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)
    border_band = np.concatenate([chroma[0], chroma[-1], chroma[:, 0], chroma[:, -1]])
    neutral_background = float(np.median(border_band)) < 8.0
    textile_points = cv2.findNonZero(textile.mask)
    textile_is_internal = False
    if textile_points is not None:
        tx, ty, tw, th = cv2.boundingRect(textile_points)
        textile_is_internal = tx > 0 and ty > 0 and tx + tw < image.shape[1] and ty + th < image.shape[0]
    textile_selected = textile.mask > 0
    textile_chroma_gain = (
        float(np.median(chroma[textile_selected])) - float(np.median(border_band))
        if np.any(textile_selected) else 0.0
    )
    if (
        neutral_background
        and textile_is_internal
        and textile_chroma_gain >= 3.0
        and 0.03 <= textile.coverage <= 0.80
        and textile.quality_score >= 0.15
    ):
        best = textile
    # A cut piece still embedded in the source sheet is an internal region,
    # while chroma segmentation usually returns the much larger mother sheet.
    # Prefer a credible seam-enclosed region when this size relationship makes
    # the two interpretations unambiguous.
    if (
        white_cut.quality_score >= 0.55
        and 0.002 <= white_cut.coverage <= 0.35
        and textile.coverage >= white_cut.coverage * 2.0
    ):
        best = white_cut
    return AssistedMaskResult(best.mask, best.method, f"智能组合 → {best.method_label}", best.coverage, best.quality_score)


def locate_assisted_template(
    image: np.ndarray,
    method: str = "auto",
    padding_ratio: float = 0.035,
) -> AutomaticTemplateSelection:
    """Locate one dominant target in a full image and return a tight ROI-local mask."""
    if not 0.0 <= padding_ratio <= 0.5:
        raise ValueError("Template padding ratio must be in [0, 0.5]")
    image_height, image_width = image.shape[:2]
    max_pixels = 2_000_000
    processing_scale = min(1.0, float(np.sqrt(max_pixels / max(float(image_height * image_width), 1.0))))
    if processing_scale < 0.999:
        processed = cv2.resize(
            image,
            (max(10, int(round(image_width * processing_scale))), max(10, int(round(image_height * processing_scale)))),
            interpolation=cv2.INTER_AREA,
        )
        processed_result = build_assisted_mask(processed, method)
        full_mask = cv2.resize(processed_result.mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        result = AssistedMaskResult(
            full_mask,
            processed_result.method,
            processed_result.method_label,
            processed_result.coverage,
            processed_result.quality_score,
        )
    else:
        result = build_assisted_mask(image, method)
    points = cv2.findNonZero(np.where(result.mask > 0, 255, 0).astype(np.uint8))
    if points is None or len(points) < 25:
        raise ValueError("No valid target was found in the imported image")
    x, y, width, height = cv2.boundingRect(points)
    padding = max(8, int(round(min(width, height) * padding_ratio)))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(image_width, x + width + padding), min(image_height, y + height + padding)
    local_mask = result.mask[y0:y1, x0:x1].copy()
    if int(np.count_nonzero(local_mask)) < 25:
        raise ValueError("Automatically located template mask is empty")
    return AutomaticTemplateSelection(
        (x0, y0, x1 - x0, y1 - y0),
        local_mask,
        result.method,
        result.method_label,
        result.coverage,
        result.quality_score,
    )
