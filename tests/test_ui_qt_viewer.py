import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from industrial_segpose.ui_qt.components import ImageViewer
from industrial_segpose.ui_qt.models import ViewMode


def test_image_viewer_keeps_roi_in_image_coordinates() -> None:
    app = QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    image = np.zeros((120, 200, 3), np.uint8)
    mask = np.full((40, 60), 255, np.uint8)
    viewer.set_image(image)
    viewer.set_roi((20, 30, 60, 40))
    viewer.set_mask(mask)
    viewer.set_view_mode(ViewMode.OVERLAY)
    viewer.zoom_in()
    assert viewer.roi == (20, 30, 60, 40)
    assert tuple(map(int, viewer.roi_item.rect().getRect())) == (20, 30, 60, 40)
    assert int(viewer.mask_item.pos().x()) == 20
    assert int(viewer.mask_item.pos().y()) == 30
    app.processEvents()
