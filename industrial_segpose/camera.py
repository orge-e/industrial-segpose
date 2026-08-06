"""OpenCV camera/video source used by the production UI."""

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
    buffer_size: int = 1


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
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")
        self.capture = capture
        values = (
            (cv2.CAP_PROP_FRAME_WIDTH, self.settings.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height),
            (cv2.CAP_PROP_FPS, self.settings.fps),
            (cv2.CAP_PROP_EXPOSURE, self.settings.exposure),
            (cv2.CAP_PROP_BUFFERSIZE, self.settings.buffer_size),
        )
        for prop, value in values:
            if value is not None:
                capture.set(prop, float(value))

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
