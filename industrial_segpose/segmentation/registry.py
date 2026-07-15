from .threshold import ThresholdBackend
from .adaptive_threshold import AdaptiveThresholdBackend
from .color_range import ColorRangeBackend
from .watershed import WatershedBackend

BACKENDS = {c.name: c for c in (ThresholdBackend, AdaptiveThresholdBackend, ColorRangeBackend, WatershedBackend)}


def create_backend(name: str, config: dict):
    try:
        return BACKENDS[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(BACKENDS)}") from exc
