"""Qt 主题：生成与设计系统一致的轻量矢量图标。"""

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


PATHS = {
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m3 16 5-5 4 4 3-3 6 6"/>',
    "camera": '<path d="M14.5 4 16 7h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h3l1.5-3z"/><circle cx="12" cy="13" r="4"/>',
    "scan": '<path d="M3 7V4a1 1 0 0 1 1-1h3M17 3h3a1 1 0 0 1 1 1v3M21 17v3a1 1 0 0 1-1 1h-3M7 21H4a1 1 0 0 1-1-1v-3"/>',
    "activity": '<path d="M3 12h4l2-7 4 14 2-7h6"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 1 1-14 0 7 7 0 0 1 14 0z"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
    "zoom-in": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4M11 8v6M8 11h6"/>',
    "zoom-out": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4M8 11h6"/>',
    "fit": '<path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/>',
    "mask": '<path d="M4 4h16v16H4z"/><path d="M4 12c3-5 6 5 9 0s5 2 7-2"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
    "circle": '<circle cx="12" cy="12" r="7"/>',
    "dot": '<circle cx="12" cy="12" r="3"/>',
}


def icon(name: str, color: str = "#94A3B8", size: int = 18) -> QIcon:
    body = PATHS.get(name, PATHS["scan"])
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
"""Qt 主题：生成与设计系统一致的轻量矢量图标。"""
