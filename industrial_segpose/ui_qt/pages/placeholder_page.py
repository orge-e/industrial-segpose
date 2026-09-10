"""Qt 页面：为尚未完成正式迁移的功能提供明确占位说明。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        label = QLabel(title)
        label.setObjectName("PageTitle")
        layout.addWidget(label)
        hint = QLabel(message)
        hint.setObjectName("PageHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
        state = QLabel("此页面仅保留迁移占位，不接入业务。")
        state.setObjectName("Muted")
        state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(state)
        layout.addStretch()
"""Qt 页面：为尚未完成正式迁移的功能提供明确占位说明。"""
