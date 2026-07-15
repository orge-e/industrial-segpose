"""YAML configuration loading and validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "industrial-segpose"},
    "input": {"resize_max_dimension": 2048, "roi": None},
    "preprocessing": {"color_space": "gray", "gaussian_blur_kernel": 5, "median_blur_kernel": 0, "clahe_enabled": False, "clahe_clip_limit": 2.0, "contrast_alpha": 1.0},
    "segmentation": {"backend": "threshold", "invert": False},
    "threshold": {"method": "otsu", "value": 127},
    "adaptive_threshold": {"block_size": 31, "c": 5},
    "color_range": {"space": "hsv", "lower": [0, 0, 0], "upper": [180, 255, 200]},
    "morphology": {"kernel_size": 3, "open_iterations": 1, "close_iterations": 1},
    "watershed": {"distance_threshold_ratio": 0.35, "background_dilate_iterations": 3, "min_marker_area": 30},
    "filter": {"min_area_px": 100, "max_area_px": None, "min_width_px": 5, "min_height_px": 5, "min_aspect_ratio": 0.05, "max_aspect_ratio": 20.0, "min_circularity": 0.0, "reject_border_objects": False},
    "measurement": {"center_method": "moments", "angle_method": "pca", "angle_range": "0_180", "min_orientation_confidence": 0.15},
    "output": {"save_annotated": True, "save_masks": True, "save_label_map": True, "save_debug": False, "save_json": True, "save_csv": True},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    config = _merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    backend = config["segmentation"]["backend"]
    if backend not in {"threshold", "adaptive_threshold", "color_range", "watershed"}:
        raise ValueError(f"Unsupported segmentation backend: {backend}")
    for section, key in (("preprocessing", "gaussian_blur_kernel"), ("preprocessing", "median_blur_kernel"), ("morphology", "kernel_size")):
        value = int(config[section][key])
        if value < 0 or (value > 0 and value % 2 == 0):
            raise ValueError(f"{section}.{key} must be zero or a positive odd integer")
    block = int(config["adaptive_threshold"]["block_size"])
    if block < 3 or block % 2 == 0:
        raise ValueError("adaptive_threshold.block_size must be an odd integer >= 3")
    ratio = float(config["watershed"]["distance_threshold_ratio"])
    if not 0 < ratio < 1:
        raise ValueError("watershed.distance_threshold_ratio must be between 0 and 1")
    if config["measurement"]["center_method"] not in {"moments", "min_area_rect"}:
        raise ValueError("Unsupported center method")
    if config["measurement"]["angle_method"] not in {"pca", "min_area_rect"}:
        raise ValueError("Unsupported angle method")
