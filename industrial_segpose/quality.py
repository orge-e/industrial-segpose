"""Lightweight image-quality checks for production camera frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityConfig:
    min_mean_brightness: float = 18.0
    max_mean_brightness: float = 242.0
    max_dark_fraction: float = 0.88
    max_bright_fraction: float = 0.68
    min_sharpness: float = 12.0
    max_illumination_spread: float = 1.05
    analysis_max_edge: int = 720

    def validate(self) -> None:
        if not 0 <= self.min_mean_brightness < self.max_mean_brightness <= 255:
            raise ValueError("Invalid brightness quality range")
        if not 0 <= self.max_dark_fraction <= 1 or not 0 <= self.max_bright_fraction <= 1:
            raise ValueError("Quality fractions must be in [0, 1]")
        if self.min_sharpness < 0 or self.max_illumination_spread <= 0:
            raise ValueError("Invalid sharpness or illumination threshold")
        if self.analysis_max_edge < 64:
            raise ValueError("analysis_max_edge must be at least 64")


@dataclass(frozen=True)
class ImageQualityReport:
    passed: bool
    score: float
    mean_brightness: float
    dark_fraction: float
    bright_fraction: float
    sharpness: float
    illumination_spread: float
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload


def evaluate_image_quality(
    image: np.ndarray,
    config: ImageQualityConfig | None = None,
) -> ImageQualityReport:
    """Assess exposure, focus and large-scale illumination without modifying a frame."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    settings = config or ImageQualityConfig()
    settings.validate()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    height, width = gray.shape[:2]
    scale = min(1.0, settings.analysis_max_edge / max(float(width), float(height)))
    if scale < 0.999:
        gray = cv2.resize(
            gray,
            (max(32, int(round(width * scale))), max(32, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.medianBlur(gray, 3)
    mean = float(np.mean(gray))
    dark_fraction = float(np.mean(gray <= 10))
    bright_fraction = float(np.mean(gray >= 248))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sigma = max(8.0, min(gray.shape[:2]) / 18.0)
    illumination = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    low, high = np.percentile(illumination, (5.0, 95.0))
    illumination_spread = float((high - low) / max(mean, 12.0))

    issues: list[str] = []
    if mean < settings.min_mean_brightness or dark_fraction > settings.max_dark_fraction:
        issues.append("underexposed")
    if mean > settings.max_mean_brightness or bright_fraction > settings.max_bright_fraction:
        issues.append("overexposed")
    if sharpness < settings.min_sharpness:
        issues.append("blurred")
    if illumination_spread > settings.max_illumination_spread:
        issues.append("uneven_illumination")

    penalties = [
        max(0.0, (settings.min_mean_brightness - mean) / max(settings.min_mean_brightness, 1.0)),
        max(0.0, (mean - settings.max_mean_brightness) / max(255.0 - settings.max_mean_brightness, 1.0)),
        max(0.0, (dark_fraction - settings.max_dark_fraction) / max(1.0 - settings.max_dark_fraction, 1e-6)),
        max(0.0, (bright_fraction - settings.max_bright_fraction) / max(1.0 - settings.max_bright_fraction, 1e-6)),
        max(0.0, (settings.min_sharpness - sharpness) / max(settings.min_sharpness, 1.0)),
        max(0.0, (illumination_spread - settings.max_illumination_spread) / settings.max_illumination_spread),
    ]
    score = float(np.clip(1.0 - max(penalties, default=0.0), 0.0, 1.0))
    return ImageQualityReport(
        not issues,
        score,
        mean,
        dark_fraction,
        bright_fraction,
        sharpness,
        illumination_spread,
        tuple(issues),
    )
