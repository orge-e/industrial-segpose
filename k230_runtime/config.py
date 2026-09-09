"""JSON configuration loader compatible with CPython and MicroPython."""

try:
    import ujson as json
except ImportError:
    import json


DEFAULT_CONFIG = {
    "device_id": "k230-line-01",
    "template_library": "/sdcard/industrial_vision/templates/template_library.json",
    "camera": {"width": 640, "height": 480, "channel": 1, "algorithm_width": 320, "algorithm_height": 240},
    "display": {"width": 640, "height": 480, "to_ide": True},
    "ui": {
        "product_name": "FlexPose Vision",
        "subtitle": "柔性工件定位系统",
    },
    "capture": {
        "root": "/sdcard/industrial_vision/captures",
        "jpeg_quality": 95,
    },
    "error_reports": {"root": "/sdcard/industrial_vision/logs"},
    "tracking": {
        "max_distance": 90,
        "max_missed": 6,
        "line_axis": "x",
        "line_position": 520,
        "direction": 1,
    },
    "quality_gate": {
        "enabled": True,
        "reject_bad_frames": True,
        "min_l_mean": 12.0,
        "max_l_mean": 92.0,
        "min_l_stdev": 5.0,
        "max_grid_spread": 42.0,
        "maximum_retries": 3,
    },
    "calibration": {
        "enabled": False,
        "path": "/sdcard/industrial_vision/calibration/calibration.json",
        "axis_snapshot_mm": [0.0, 0.0],
    },
    # Board-side authoring remains in the codebase but is disabled by default.
    # Templates are authored and validated on the desktop, then deployed as a
    # compact read-only bundle to keep the production runtime stable.
    "template_authoring": {"enabled": True, "require_preview": True},
    "transport": {"type": "console", "uart_id": 1, "baudrate": 115200},
    "heartbeat_interval_ms": 1000,
    "minimum_confidence": 0.60,
    "emit_ambiguous": False,
}


def _merge(base, updates):
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    if path is None:
        return _merge(DEFAULT_CONFIG, {})
    # MicroPython's open() does not consistently accept the encoding keyword.
    with open(path, "r") as stream:
        payload = json.loads(stream.read())
    if not isinstance(payload, dict):
        raise ValueError("runtime configuration must be an object")
    return _merge(DEFAULT_CONFIG, payload)
