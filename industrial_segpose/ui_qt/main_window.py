"""PySide6 正式界面：组织全局导航、页面容器和设备状态栏。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .components import StatusBadge
from .pages import PlaceholderPage, TemplatePage
from .theme.icons import icon
from .theme.tokens import COLORS


class MainWindow(QMainWindow):
    def __init__(self, project_root: str | Path):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle("FlexPose Vision")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)
        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        shell = QVBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._header())
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        middle.addWidget(self._sidebar())
        self.pages = QStackedWidget()
        self.template_page = TemplatePage(self.project_root)
        self.detection_page = PlaceholderPage("模板检测", "该页面将在建立模板流程通过 Design Freeze 后迁移。现有 Tkinter 检测功能不受影响。")
        self.live_page = PlaceholderPage("实时视觉", "实时相机、生产指标和结果表暂未接入，本阶段只完成正式建立模板页面。")
        self.validation_page = PlaceholderPage("算法验证", "压力测试和参数搜索将在后续迁移，当前继续使用稳定版入口。")
        for page in (self.template_page, self.detection_page, self.live_page, self.validation_page):
            self.pages.addWidget(page)
        middle.addWidget(self.pages, 1)
        shell.addLayout(middle, 1)
        shell.addWidget(self._footer())
        self.template_page.navigate_requested.connect(self.show_page)
        self.template_page.status_changed.connect(self._set_footer_status)

    def _header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(54)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)
        mark = QLabel("FP")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(30, 30)
        mark.setStyleSheet(f"background:{COLORS['primary']}; border-radius:5px; font-weight:700;")
        layout.addWidget(mark)
        name = QLabel("FlexPose Vision")
        name.setObjectName("AppName")
        layout.addWidget(name)
        layout.addStretch()
        layout.addWidget(StatusBadge("SYSTEM READY", "success"))
        device = QLabel("DESKTOP")
        device.setObjectName("Secondary")
        layout.addWidget(device)
        version = QLabel(f"v{__version__}")
        version.setObjectName("Muted")
        layout.addWidget(version)
        return header

    def _sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(194)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 18, 0, 16)
        layout.setSpacing(2)
        title = QLabel("工作区")
        title.setObjectName("Muted")
        title.setContentsMargins(20, 0, 0, 8)
        layout.addWidget(title)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        entries = (("建立模板", "image"), ("模板检测", "scan"), ("实时视觉", "camera"), ("算法验证", "activity"))
        self.nav_buttons: list[QToolButton] = []
        for index, (text, icon_name) in enumerate(entries):
            button = QToolButton()
            button.setObjectName("NavButton")
            button.setText(text)
            button.setIcon(icon(icon_name, COLORS["secondary"], 18))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        separator = QFrame()
        separator.setObjectName("Divider")
        layout.addSpacing(14)
        layout.addWidget(separator)
        tools = QLabel("工具")
        tools.setObjectName("Muted")
        tools.setContentsMargins(20, 12, 0, 8)
        layout.addWidget(tools)
        for text, icon_name in (("标定工具", "scan"), ("设置", "settings")):
            button = QToolButton()
            button.setObjectName("NavButton")
            button.setText(text)
            button.setIcon(icon(icon_name))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setEnabled(False)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(button)
        layout.addStretch()
        engine = QLabel("OpenCV / CPU\nQt 正式迁移阶段")
        engine.setObjectName("Muted")
        engine.setContentsMargins(20, 0, 0, 0)
        layout.addWidget(engine)
        self.nav_group.idClicked.connect(self._navigate)
        return sidebar

    def _footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        footer.setFixedHeight(34)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(24)
        self.footer_status = StatusBadge("就绪", "success")
        layout.addWidget(self.footer_status)
        layout.addWidget(StatusBadge("模板库已连接", "info"))
        layout.addStretch()
        state = QLabel("PySide6 建立模板正式迁移 · Tkinter 稳定版保留")
        state.setObjectName("Muted")
        layout.addWidget(state)
        return footer

    def _set_footer_status(self, text: str, tone: str) -> None:
        self.footer_status.set_status(text, tone)

    def _navigate(self, index: int) -> None:
        self.pages.setCurrentIndex(index)

    def show_page(self, name: str) -> None:
        mapping = {"template": 0, "detection": 1, "live": 2, "validation": 3}
        index = mapping.get(name, 0)
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
"""PySide6 正式界面：组织全局导航、页面容器和设备状态栏。"""
