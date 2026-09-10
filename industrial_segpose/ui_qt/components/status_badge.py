"""Qt 通用组件：显示运行状态及其语义颜色。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from ..theme.tokens import COLORS


class StatusBadge(QLabel):
    def __init__(self, text: str = "就绪", tone: str = "success", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.set_status(text, tone)

    def set_status(self, text: str, tone: str = "success") -> None:
        color = COLORS.get(tone, COLORS["secondary"])
        self.setText(f"●  {text}")
        self.setStyleSheet(f"color:{color}; font-size:10px; font-weight:600;")
"""Qt 通用组件：显示运行状态及其语义颜色。"""
