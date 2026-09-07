"""Qt 通用组件：显示模板建立流程及当前步骤。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QLabel, QPushButton, QVBoxLayout

from ..models.template_state import TemplateStep
from ..theme.icons import icon
from ..theme.tokens import COLORS


class WorkflowStepper(QFrame):
    step_requested = Signal(int)
    steps = (("01", "基准图像"), ("02", "ROI"), ("03", "Mask"), ("04", "模板参数"), ("05", "保存完成"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFixedWidth(158)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 18, 12, 16)
        layout.setSpacing(0)
        title = QLabel("建模流程")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        layout.addSpacing(12)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        self.connectors: list[QFrame] = []
        for index, (number, label) in enumerate(self.steps):
            button = QPushButton(f"{number}  {label}")
            button.setObjectName("StepButton")
            button.setCheckable(True)
            self.group.addButton(button, index)
            self.buttons.append(button)
            layout.addWidget(button)
            if index < len(self.steps) - 1:
                connector = QFrame()
                connector.setFixedSize(1, 12)
                connector.setStyleSheet(f"background:{COLORS['divider']}; margin-left:7px;")
                self.connectors.append(connector)
                layout.addWidget(connector)
        layout.addStretch()
        hint = QLabel("绿色表示已完成\n蓝色表示当前步骤\n红色表示需要修正")
        hint.setObjectName("Muted")
        layout.addWidget(hint)
        self.group.idClicked.connect(self.step_requested.emit)

    def set_state(self, active: TemplateStep, completed: set[TemplateStep], error_step: TemplateStep | None = None) -> None:
        for index, button in enumerate(self.buttons):
            step = TemplateStep(index)
            button.blockSignals(True)
            button.setChecked(step == active)
            button.blockSignals(False)
            if error_step == step:
                button.setIcon(icon("circle", COLORS["error"], 16))
            elif step in completed and step != active:
                button.setIcon(icon("check", COLORS["success"], 16))
            elif step == active:
                button.setIcon(icon("dot", COLORS["primary"], 16))
            else:
                button.setIcon(icon("circle", COLORS["muted"], 16))
"""Qt 通用组件：显示模板建立流程及当前步骤。"""
