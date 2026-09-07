"""Qt 服务：提供耗时图像任务使用的后台工作线程。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot


class FunctionWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, function: Callable[[], object]):
        super().__init__()
        self.function = function

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.function())
        except Exception as exc:  # UI boundary: convert domain errors to user-facing text.
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
"""Qt 服务：提供耗时图像任务使用的后台工作线程。"""
