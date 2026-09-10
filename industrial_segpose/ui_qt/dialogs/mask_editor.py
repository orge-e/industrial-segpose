"""Qt 对话框：提供模板 Mask 的可视化检查与人工修正。"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class MaskPaintCanvas(QWidget):
    def __init__(self, image: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.mode = "add"
        self.radius = 12
        self._painting = False
        self.setMinimumSize(680, 420)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy())

    def _target_rect(self) -> QRectF:
        source = self.pixmap.size()
        scale = min(self.width() / max(source.width(), 1), self.height() / max(source.height(), 1))
        width, height = source.width() * scale, source.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _to_image(self, position: QPointF) -> tuple[int, int] | None:
        rect = self._target_rect()
        if not rect.contains(position):
            return None
        x = int((position.x() - rect.x()) / rect.width() * self.image.shape[1])
        y = int((position.y() - rect.y()) / rect.height() * self.image.shape[0])
        return max(0, min(self.image.shape[1] - 1, x)), max(0, min(self.image.shape[0] - 1, y))

    def _paint_at(self, position: QPointF) -> None:
        point = self._to_image(position)
        if point is None:
            return
        cv2.circle(self.mask, point, self.radius, 255 if self.mode == "add" else 0, -1, cv2.LINE_AA)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self._paint_at(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._painting:
            self._paint_at(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = False

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#060A10"))
        target = self._target_rect()
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        rgba = np.zeros((*self.mask.shape, 4), np.uint8)
        rgba[self.mask > 0] = (59, 130, 246, 100)
        h, w = self.mask.shape
        overlay = QPixmap.fromImage(QImage(rgba.data, w, h, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy())
        painter.drawPixmap(target, overlay, QRectF(overlay.rect()))
        painter.setPen(QPen(QColor("#263244"), 1))
        painter.drawRect(target)


class MaskEditorDialog(QDialog):
    def __init__(self, image: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检查并修正 Mask")
        self.resize(960, 700)
        self.result_mask: np.ndarray | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        tools = QHBoxLayout()
        hint = QLabel("左键绘制；蓝色区域为模板有效区域")
        hint.setObjectName("Secondary")
        tools.addWidget(hint)
        tools.addStretch()
        self.mode_group = QButtonGroup(self)
        for index, text in enumerate(("添加", "擦除")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setChecked(index == 0)
            self.mode_group.addButton(button, index)
            tools.addWidget(button)
        tools.addWidget(QLabel("画笔"))
        radius = QSlider(Qt.Orientation.Horizontal)
        radius.setRange(2, 60)
        radius.setValue(12)
        radius.setFixedWidth(110)
        tools.addWidget(radius)
        root.addLayout(tools)
        self.canvas = MaskPaintCanvas(image, mask)
        root.addWidget(self.canvas, 1)
        actions = QHBoxLayout()
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear)
        actions.addWidget(clear)
        actions.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        confirm = QPushButton("确认 Mask")
        confirm.setObjectName("PrimaryButton")
        confirm.clicked.connect(self._accept)
        actions.addWidget(confirm)
        root.addLayout(actions)
        self.mode_group.idClicked.connect(lambda index: setattr(self.canvas, "mode", "add" if index == 0 else "erase"))
        radius.valueChanged.connect(lambda value: setattr(self.canvas, "radius", value))

    def _clear(self) -> None:
        self.canvas.mask.fill(0)
        self.canvas.update()

    def _accept(self) -> None:
        if int(np.count_nonzero(self.canvas.mask)) < 25:
            return
        self.result_mask = self.canvas.mask.copy()
        self.accept()
"""Qt 对话框：提供模板 Mask 的可视化检查与人工修正。"""
