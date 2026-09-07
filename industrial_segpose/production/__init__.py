"""生产运行子包：集中管理相机、质量检查、跟踪、计数和运行报告。"""

from .camera import CameraSettings, OpenCVCameraSource
from .quality import ImageQualityConfig, ImageQualityReport, evaluate_image_quality
from .runtime import ConveyorFrameResult, ConveyorSession, draw_conveyor_frame

__all__ = [
    "CameraSettings",
    "OpenCVCameraSource",
    "ImageQualityConfig",
    "ImageQualityReport",
    "evaluate_image_quality",
    "ConveyorFrameResult",
    "ConveyorSession",
    "draw_conveyor_frame",
]
