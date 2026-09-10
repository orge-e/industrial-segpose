"""Choose a candidate-extraction strategy from template appearance statistics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .model import TemplateModel


@dataclass(frozen=True)
class FeatureModeRecommendation:
    mode: str
    confidence: float
    reason: str
    foreground_lightness: float
    foreground_chroma: float
    foreground_texture: float
    background_lightness: float | None
    background_chroma: float | None
    background_texture: float | None


def _texture_energy(gray: np.ndarray) -> np.ndarray:
    local = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    return cv2.GaussianBlur(np.abs(gray.astype(np.float32) - local), (0, 0), 2.0)


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def recommend_feature_mode(model: TemplateModel) -> FeatureModeRecommendation:
    """Recommend a robust mode without using a class name or hard-coded colour.

    Colour is used only to propose a candidate extractor.  Final identity is
    always established by the stored irregular mask/contour.
    """

    image = model.image
    mask = model.mask > 0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = _texture_energy(gray)
    chroma_map = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)

    foreground_l = float(np.median(lab[:, :, 0][mask]))
    foreground_chroma = float(np.median(chroma_map[mask]))
    foreground_texture = float(np.median(texture[mask]))
    outside = ~mask
    # Ignore a one-pixel mask transition when estimating local background.
    outside = cv2.erode(outside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    background_l = _median_or_none(lab[:, :, 0][outside])
    background_chroma = _median_or_none(chroma_map[outside])
    background_texture = _median_or_none(texture[outside])

    if background_l is None:
        return FeatureModeRecommendation(
            "edges", 0.45, "模板没有可用背景像素，采用通用轮廓边缘",
            foreground_l, foreground_chroma, foreground_texture,
            None, None, None,
        )

    lightness_delta = abs(foreground_l - background_l)
    signed_lightness_delta = foreground_l - background_l
    chroma_delta = abs(foreground_chroma - float(background_chroma))
    appearance_delta = float(np.hypot(lightness_delta, chroma_delta))

    if foreground_chroma >= 9.0 and (chroma_delta >= 3.0 or appearance_delta >= 8.0):
        confidence = float(np.clip(0.55 + foreground_chroma / 60.0 + chroma_delta / 80.0, 0.0, 0.98))
        return FeatureModeRecommendation(
            "pose_tolerant", confidence, "前景具有稳定色度，使用模板自适应色度候选",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    # Compact cameras' auto white balance can render white cloth pale yellow/gray with Lab
    # lightness around 140-150.  Relative similarity to the mother material is
    # the important signal; requiring photographic white incorrectly routes
    # these templates to generic edges and loses the board seam detector.
    if foreground_l >= 130.0 and background_l >= 120.0 and appearance_delta < 20.0:
        confidence = float(np.clip(0.82 - appearance_delta / 60.0, 0.55, 0.95))
        return FeatureModeRecommendation(
            "white_cut_seam", confidence, "前景与浅色母料外观接近，使用裁剪缝闭合",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    if (
        foreground_chroma < 6.0
        and foreground_l >= 135.0
        and background_l >= 80.0
        and signed_lightness_delta >= 10.0
    ):
        confidence = float(np.clip(0.58 + signed_lightness_delta / 100.0, 0.58, 0.92))
        return FeatureModeRecommendation(
            "white_cut_seam", confidence, "浅色独立工件缺少稳定色度，使用阴影与闭合轮廓定位",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    if foreground_l <= 100.0 and signed_lightness_delta <= -18.0:
        confidence = float(np.clip(0.62 + (-signed_lightness_delta) / 220.0, 0.62, 0.96))
        return FeatureModeRecommendation(
            "dark_textile", confidence, "工件显著暗于支撑面，使用暗色轮廓与纹理候选",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    texture_ratio = foreground_texture / max(float(background_texture), 0.5)
    if foreground_l < 165.0 and foreground_texture >= 2.0 and texture_ratio >= 1.22:
        confidence = float(np.clip(0.52 + (texture_ratio - 1.0) * 0.22, 0.52, 0.94))
        return FeatureModeRecommendation(
            "dark_textile", confidence, "前景主要依靠局部纹理与背景区分",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    if appearance_delta >= 14.0:
        return FeatureModeRecommendation(
            "pose_tolerant", 0.62, "前景与背景存在可用外观差异，使用模板自适应候选",
            foreground_l, foreground_chroma, foreground_texture,
            background_l, background_chroma, background_texture,
        )

    return FeatureModeRecommendation(
        "edges", 0.50, "未发现可靠颜色或纹理通道，回退到通用轮廓边缘",
        foreground_l, foreground_chroma, foreground_texture,
        background_l, background_chroma, background_texture,
    )
