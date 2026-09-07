"""Reusable scene/view based industrial image viewer.

Scene coordinates are always image pixels.  This keeps ROI and future
detection overlays independent from zoom, DPI scaling and viewport size.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..models.template_state import ViewMode
from ..theme.tokens import COLORS


def _bgr_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    qimage = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _mask_pixmap(mask: np.ndarray, mode: ViewMode) -> QPixmap:
    height, width = mask.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    selected = mask > 0
    if mode == ViewMode.MASK:
        rgba[selected] = (235, 240, 246, 255)
    else:
        rgba[selected] = (59, 130, 246, 92)
    qimage = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


class ImageViewer(QGraphicsView):
    roi_changed = Signal(object)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        self.graphics_scene = QGraphicsScene()
        super().__init__(self.graphics_scene, parent)
        self.setBackgroundBrush(QColor(COLORS["canvas"]))
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(4)
        self.roi_item = QGraphicsRectItem()
        self.roi_item.setZValue(6)
        self.roi_item.setPen(QPen(QColor(COLORS["primary"]), 2, Qt.PenStyle.DashLine))
        self.roi_item.setBrush(QColor(59, 130, 246, 18))
        self.roi_item.hide()
        self.graphics_scene.addItem(self.image_item)
        self.graphics_scene.addItem(self.mask_item)
        self.graphics_scene.addItem(self.roi_item)
        self._image: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._view_mode = ViewMode.ORIGINAL
        self._roi_enabled = False
        self._select_origin = None
        self._panning = False
        self._pan_origin = QPoint()
        self._zoom = 1.0

    @property
    def image(self) -> np.ndarray | None:
        return self._image

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def set_image(self, image: np.ndarray | None, *, fit: bool = True) -> None:
        self._image = None if image is None else image.copy()
        self._mask = None
        self.mask_item.setPixmap(QPixmap())
        self.clear_roi(emit=False)
        if image is None:
            self.image_item.setPixmap(QPixmap())
            self.graphics_scene.setSceneRect(QRectF())
            return
        self.image_item.setPixmap(_bgr_pixmap(image))
        height, width = image.shape[:2]
        self.graphics_scene.setSceneRect(QRectF(0, 0, width, height))
        if fit:
            self.fit_to_image()

    def set_roi(self, roi: tuple[int, int, int, int] | None, *, emit: bool = False) -> None:
        if roi is None:
            self.clear_roi(emit=emit)
            return
        x, y, width, height = map(int, roi)
        self._roi = (x, y, width, height)
        self.roi_item.setRect(QRectF(x, y, width, height))
        self.roi_item.show()
        if self._mask is not None:
            self.mask_item.setPos(x, y)
        if emit:
            self.roi_changed.emit(self._roi)

    def clear_roi(self, *, emit: bool = True) -> None:
        self._roi = None
        self.roi_item.hide()
        self.mask_item.setPixmap(QPixmap())
        self._mask = None
        if emit:
            self.roi_changed.emit(None)

    def enable_roi_selection(self, enabled: bool = True) -> None:
        self._roi_enabled = bool(enabled)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_mask(self, mask: np.ndarray | None, roi: tuple[int, int, int, int] | None = None) -> None:
        self._mask = None if mask is None else np.where(mask > 0, 255, 0).astype(np.uint8)
        if roi is not None:
            self.set_roi(roi)
        self._refresh_mask_item()

    def set_view_mode(self, mode: ViewMode | str) -> None:
        self._view_mode = ViewMode(mode)
        self.image_item.setOpacity(0.14 if self._view_mode == ViewMode.MASK else 1.0)
        self._refresh_mask_item()

    def _refresh_mask_item(self) -> None:
        if self._mask is None or self._view_mode == ViewMode.ORIGINAL:
            self.mask_item.setPixmap(QPixmap())
            return
        self.mask_item.setPixmap(_mask_pixmap(self._mask, self._view_mode))
        if self._roi:
            self.mask_item.setPos(self._roi[0], self._roi[1])

    def fit_to_image(self) -> None:
        if not self.image_item.pixmap().isNull():
            self.resetTransform()
            self.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()
            self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self._apply_zoom(1.2)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.2)

    def _apply_zoom(self, factor: float) -> None:
        if self._image is None:
            return
        next_scale = self.transform().m11() * factor
        if 0.03 <= next_scale <= 30.0:
            self.scale(factor, factor)
            self._zoom = self.transform().m11()
            self.zoom_changed.emit(self._zoom)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self._apply_zoom(1.18 if event.angleDelta().y() > 0 else 1 / 1.18)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._roi_enabled and self._image is not None:
            point = self.mapToScene(event.position().toPoint())
            if self.graphics_scene.sceneRect().contains(point):
                self._select_origin = point
                self.roi_item.setRect(QRectF(point, point))
                self.roi_item.show()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position().toPoint() - self._pan_origin
            self._pan_origin = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._select_origin is not None:
            current = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._select_origin, current).normalized().intersected(self.graphics_scene.sceneRect())
            self.roi_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._roi_enabled else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._select_origin is not None:
            rect = self.roi_item.rect().normalized()
            self._select_origin = None
            x, y = max(0, int(round(rect.x()))), max(0, int(round(rect.y())))
            width, height = int(round(rect.width())), int(round(rect.height()))
            if width >= 5 and height >= 5:
                self.set_roi((x, y, width, height), emit=True)
            else:
                self.clear_roi()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None and abs(self._zoom - self.transform().m11()) < 1e-9:
            pass
