"""生产运行：封装实时工作站使用的 OpenCV 相机和视频源。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSettings:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    exposure: float | None = None
    gain: float | None = None
    auto_exposure: bool | None = None
    backend: str = "auto"
    fourcc: str | None = None
    buffer_size: int = 1

    def validate(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("Camera width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("Camera height must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("Camera FPS must be positive")
        if self.backend.lower() not in {"auto", "dshow", "msmf"}:
            raise ValueError("Camera backend must be auto, dshow or msmf")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ValueError("Camera FOURCC must contain exactly four characters")


class OpenCVCameraSource:
    def __init__(self, source: int | str = 0, settings: CameraSettings | None = None):
        self.source = source
        self.settings = settings or CameraSettings()
        self.capture: cv2.VideoCapture | None = None

    @classmethod
    def from_text(cls, value: str, settings: CameraSettings | None = None) -> "OpenCVCameraSource":
        normalized = value.strip()
        if not normalized:
            raise ValueError("Camera source cannot be empty")
        source: int | str = int(normalized) if normalized.lstrip("-").isdigit() else normalized
        return cls(source, settings)

    @property
    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def open(self) -> None:
        self.close()
        self.settings.validate()
        backend = {
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
        }[self.settings.backend.lower()]
        capture = cv2.VideoCapture(self.source, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")
        self.capture = capture
        if self.settings.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.fourcc))
        if self.settings.auto_exposure is not None:
            # OpenCV backends use different conventions. These values cover the
            # common DirectShow/MSMF camera drivers on Windows.
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if self.settings.auto_exposure else 0.25)
        values = (
            (cv2.CAP_PROP_FRAME_WIDTH, self.settings.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height),
            (cv2.CAP_PROP_FPS, self.settings.fps),
            (cv2.CAP_PROP_EXPOSURE, self.settings.exposure),
            (cv2.CAP_PROP_GAIN, self.settings.gain),
            (cv2.CAP_PROP_BUFFERSIZE, self.settings.buffer_size),
        )
        for prop, value in values:
            if value is not None:
                capture.set(prop, float(value))

    def actual_settings(self) -> dict[str, float]:
        if not self.is_opened or self.capture is None:
            raise RuntimeError("Camera source is not open")
        return {
            "width": float(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.capture.get(cv2.CAP_PROP_FPS)),
            "exposure": float(self.capture.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(self.capture.get(cv2.CAP_PROP_GAIN)),
        }

    def read(self) -> np.ndarray:
        if not self.is_opened:
            raise RuntimeError("Camera source is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise EOFError("Camera/video source returned no frame")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "OpenCVCameraSource":
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
