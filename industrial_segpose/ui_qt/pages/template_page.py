"""Qt 页面：实现正式模板建立工作流及基准图像交互。"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...template_matching.matcher import MatchParameters
from ..components import ImageViewer, WorkflowStepper
from ..dialogs import MaskEditorDialog
from ..models import TemplatePageState, TemplateStep, ViewMode
from ..services import SegmentationOutput, TemplateService
from ..services.workers import FunctionWorker
from ..theme.icons import icon
from ..theme.tokens import COLORS, SPACE


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _field(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("FieldLabel")
    return label


class ViewerPanel(QFrame):
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Viewer")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(5)
        toolbar.addWidget(_section("基准图像"))
        toolbar.addSpacing(10)
        self.mode_buttons: list[QPushButton] = []
        for index, (text, value) in enumerate((("原图", "original"), ("Mask", "mask"), ("叠加", "overlay"))):
            button = QPushButton(text)
            button.setObjectName("SegmentButton")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.clicked.connect(lambda _checked=False, current=value: self._select_mode(current))
            toolbar.addWidget(button)
            self.mode_buttons.append(button)
        toolbar.addStretch()
        self.roi_mode = QPushButton("框选 ROI")
        self.roi_mode.setCheckable(True)
        self.roi_mode.setIcon(icon("scan"))
        toolbar.addWidget(self.roi_mode)
        zoom_out = QPushButton()
        zoom_out.setIcon(icon("zoom-out"))
        zoom_out.setFixedWidth(34)
        toolbar.addWidget(zoom_out)
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("Secondary")
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self.zoom_label)
        zoom_in = QPushButton()
        zoom_in.setIcon(icon("zoom-in"))
        zoom_in.setFixedWidth(34)
        toolbar.addWidget(zoom_in)
        fit = QPushButton()
        fit.setIcon(icon("fit"))
        fit.setFixedWidth(34)
        toolbar.addWidget(fit)
        root.addLayout(toolbar)
        self.viewer = ImageViewer()
        root.addWidget(self.viewer, 1)
        metadata = QHBoxLayout()
        self.size_label = QLabel("— × —")
        self.size_label.setObjectName("Muted")
        metadata.addWidget(self.size_label)
        rgb = QLabel("RGB")
        rgb.setObjectName("Muted")
        metadata.addWidget(rgb)
        self.roi_label = QLabel("ROI —")
        self.roi_label.setObjectName("Muted")
        metadata.addWidget(self.roi_label)
        metadata.addStretch()
        hint = QLabel("滚轮缩放 · 中键平移 · 左键框选 ROI")
        hint.setObjectName("Muted")
        metadata.addWidget(hint)
        root.addLayout(metadata)
        self.roi_mode.toggled.connect(self.viewer.enable_roi_selection)
        zoom_out.clicked.connect(self.viewer.zoom_out)
        zoom_in.clicked.connect(self.viewer.zoom_in)
        fit.clicked.connect(self.viewer.fit_to_image)
        self.viewer.zoom_changed.connect(lambda value: self.zoom_label.setText(f"{value * 100:.0f}%"))

    def _select_mode(self, mode: str) -> None:
        for button, value in zip(self.mode_buttons, ("original", "mask", "overlay"), strict=True):
            button.setChecked(value == mode)
        self.mode_changed.emit(mode)

    def set_mode(self, mode: ViewMode) -> None:
        for button, value in zip(self.mode_buttons, ViewMode, strict=True):
            button.setChecked(value == mode)
        self.viewer.set_view_mode(mode)


class TemplateInspector(QFrame):
    name_changed = Signal(str)
    segment_requested = Signal()
    clear_roi_requested = Signal()
    edit_mask_requested = Signal()
    parameters_changed = Signal()
    save_requested = Signal(bool)

    FEATURE_MODES = {
        "自动选择": "auto",
        "边缘特征": "edges",
        "灰度特征": "gray",
        "纺织色度": "textile_chroma",
        "柔性姿态": "pose_tolerant",
        "暗色纹理": "dark_textile",
        "白色裁剪缝": "white_cut_seam",
    }
    MASK_METHODS = {
        "智能组合": "auto",
        "纺织色度": "textile_chroma",
        "暗色纹理": "dark_textile",
        "白色裁剪缝": "white_cut_seam",
        "独立工件背景": "detached_textile",
        "边界颜色差": "border_color",
        "亮目标 Otsu": "otsu_light",
        "暗目标 Otsu": "otsu_dark",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Elevated")
        self.setFixedWidth(286)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(12)
        root.addWidget(_section("模板属性"))
        self.context_title = QLabel("基准图像")
        self.context_title.setStyleSheet("font-size:18px; font-weight:600;")
        root.addWidget(self.context_title)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._source_page())
        self.stack.addWidget(self._roi_page())
        self.stack.addWidget(self._mask_page())
        self.stack.addWidget(self._parameters_page())
        self.stack.addWidget(self._save_page())
        root.addWidget(self.stack, 1)

    @staticmethod
    def _page() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)
        return page, layout

    def _source_page(self) -> QWidget:
        page, layout = self._page()
        layout.addWidget(_field("模板名称（工件类型）"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如：鞋面裁片_A")
        self.name_input.textChanged.connect(self.name_changed)
        layout.addWidget(self.name_input)
        layout.addSpacing(6)
        layout.addWidget(_section("图像来源"))
        self.source_value = QLabel("尚未选择图像")
        self.source_value.setObjectName("Secondary")
        self.source_value.setWordWrap(True)
        layout.addWidget(self.source_value)
        self.source_size = QLabel("— × —")
        self.source_size.setObjectName("Muted")
        layout.addWidget(self.source_size)
        layout.addStretch()
        return page

    def _roi_page(self) -> QWidget:
        page, layout = self._page()
        hint = QLabel("在图像中框选一个完整工件，并在边缘保留少量背景。释放鼠标后自动分割。")
        hint.setObjectName("Secondary")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.roi_values: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        for row, key in enumerate(("X", "Y", "宽度", "高度")):
            grid.addWidget(_field(key), row, 0)
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(value, row, 1)
            self.roi_values[key] = value
        layout.addLayout(grid)
        clear = QPushButton("清除并重新选择 ROI")
        clear.setIcon(icon("scan"))
        clear.clicked.connect(self.clear_roi_requested)
        layout.addWidget(clear)
        self.roi_status = QLabel("○  等待 ROI")
        self.roi_status.setObjectName("Muted")
        layout.addWidget(self.roi_status)
        layout.addStretch()
        return page

    def _mask_page(self) -> QWidget:
        page, layout = self._page()
        layout.addWidget(_field("分割方法"))
        self.mask_algorithm_select = QComboBox()
        self.mask_algorithm_select.addItems(self.MASK_METHODS)
        layout.addWidget(self.mask_algorithm_select)
        self.mask_method = QLabel("方法  —")
        self.mask_method.setObjectName("Secondary")
        layout.addWidget(self.mask_method)
        self.mask_coverage = QLabel("覆盖率  —")
        layout.addWidget(self.mask_coverage)
        self.mask_boundary = QLabel("边界质量  —")
        layout.addWidget(self.mask_boundary)
        self.mask_status = QLabel("○  尚未建立 Mask")
        self.mask_status.setObjectName("Muted")
        self.mask_status.setWordWrap(True)
        layout.addWidget(self.mask_status)
        rerun = QPushButton("重新自动分割")
        rerun.clicked.connect(self.segment_requested)
        layout.addWidget(rerun)
        edit = QPushButton("检查并修正 Mask")
        edit.setIcon(icon("mask"))
        edit.clicked.connect(self.edit_mask_requested)
        layout.addWidget(edit)
        layout.addStretch()
        return page

    def _parameters_page(self) -> QWidget:
        page, layout = self._page()
        self.threshold = self._double(0.05, 1.0, 0.72, 0.01, 2)
        self.angle_min = self._double(-180, 180, -180, 1, 1)
        self.angle_max = self._double(-180, 180, 180, 1, 1)
        self.angle_step = self._double(0.1, 90, 2, 0.5, 1)
        self.scale_min = self._double(0.1, 5, 1.0, 0.05, 2)
        self.scale_max = self._double(0.1, 5, 1.0, 0.05, 2)
        self.scale_step = self._double(0.01, 1, 0.1, 0.01, 2)
        self.nms = self._double(0, 1, 0.25, 0.05, 2)
        for label, control in (("匹配阈值", self.threshold), ("最小角度", self.angle_min), ("最大角度", self.angle_max), ("角度步长", self.angle_step), ("最小尺度", self.scale_min), ("最大尺度", self.scale_max), ("尺度步长", self.scale_step), ("NMS IoU", self.nms)):
            layout.addWidget(_field(label))
            layout.addWidget(control)
            control.valueChanged.connect(self.parameters_changed)
        layout.addWidget(_field("特征模式"))
        self.feature = QComboBox()
        self.feature.addItems(self.FEATURE_MODES)
        self.feature.currentIndexChanged.connect(self.parameters_changed)
        layout.addWidget(self.feature)
        layout.addStretch()
        return page

    def _save_page(self) -> QWidget:
        page, layout = self._page()
        self.save_summary = QLabel("请先完成图像、ROI 与 Mask。")
        self.save_summary.setObjectName("Secondary")
        self.save_summary.setWordWrap(True)
        layout.addWidget(self.save_summary)
        self.library_count = QLabel("模板库：—")
        self.library_count.setObjectName("Muted")
        layout.addWidget(self.library_count)
        self.save_button = QPushButton("保存模板")
        self.save_button.setIcon(icon("save"))
        self.save_button.clicked.connect(lambda: self.save_requested.emit(False))
        layout.addWidget(self.save_button)
        self.save_detect_button = QPushButton("保存并进入检测")
        self.save_detect_button.clicked.connect(lambda: self.save_requested.emit(True))
        layout.addWidget(self.save_detect_button)
        layout.addStretch()
        return page

    @staticmethod
    def _double(minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setSingleStep(step)
        control.setDecimals(decimals)
        return control

    def parameters(self) -> MatchParameters:
        feature_mode = self.FEATURE_MODES[self.feature.currentText()]
        return MatchParameters(
            score_threshold=self.threshold.value(),
            angle_min=self.angle_min.value(),
            angle_max=self.angle_max.value(),
            angle_step=self.angle_step.value(),
            scale_min=self.scale_min.value(),
            scale_max=self.scale_max.value(),
            scale_step=self.scale_step.value(),
            nms_iou_threshold=self.nms.value(),
            use_edges=feature_mode != "gray",
            feature_mode=feature_mode,
        )

    def mask_algorithm(self) -> str:
        return self.MASK_METHODS[self.mask_algorithm_select.currentText()]

    def set_step(self, step: TemplateStep) -> None:
        titles = ("基准图像", "ROI", "Mask", "匹配参数", "保存完成")
        self.context_title.setText(titles[int(step)])
        self.stack.setCurrentIndex(int(step))

    def render(self, state: TemplatePageState, library_count: int) -> None:
        self.set_step(state.step)
        self.source_value.setText(state.source_path.name if state.source_path else "尚未选择图像")
        self.source_size.setText(state.image_size_text)
        if state.roi_xywh:
            for key, value in zip(("X", "Y", "宽度", "高度"), state.roi_xywh, strict=True):
                self.roi_values[key].setText(f"{value} px")
            self.roi_status.setText("●  ROI 已建立")
            self.roi_status.setObjectName("Success")
        else:
            for value in self.roi_values.values():
                value.setText("—")
            self.roi_status.setText("○  等待 ROI")
            self.roi_status.setObjectName("Muted")
        quality = state.mask_quality
        self.mask_method.setText(f"方法  {state.mask_method or '—'}")
        self.mask_coverage.setText(f"覆盖率  {quality.coverage:.1%}" if quality else "覆盖率  —")
        self.mask_boundary.setText(f"主体占比  {quality.main_component_ratio:.1%}" if quality else "主体占比  —")
        if quality:
            self.mask_status.setText(("●  " if quality.valid else "▲  ") + quality.message)
            self.mask_status.setObjectName("Success" if quality.valid else "Warning")
        else:
            self.mask_status.setText("○  尚未建立 Mask")
            self.mask_status.setObjectName("Muted")
        for widget in (self.roi_status, self.mask_status):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        self.save_button.setEnabled(state.can_save)
        self.save_detect_button.setEnabled(state.can_save)
        self.save_summary.setText("模板信息完整，可以写入持久模板库。" if state.can_save else "请填写名称并完成图像、ROI 与 Mask。")
        self.library_count.setText(f"模板库：{library_count} 项")


class TemplatePage(QWidget):
    navigate_requested = Signal(str)
    status_changed = Signal(str, str)

    def __init__(self, project_root: str | Path, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.template_service = TemplateService(self.project_root)
        self.state = TemplatePageState()
        self._thread: QThread | None = None
        self._worker: FunctionWorker | None = None
        self._build_ui()
        self._render()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(14)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        title = QLabel("建立模板")
        title.setObjectName("PageTitle")
        hint = QLabel("从单工件参考图像建立可复用的 ROI 与 Mask，并加入持久模板库。")
        hint.setObjectName("PageHint")
        titles.addWidget(title)
        titles.addWidget(hint)
        heading.addLayout(titles)
        heading.addStretch()
        self.import_button = QPushButton("导入图像")
        self.import_button.setObjectName("PrimaryButton")
        self.import_button.setIcon(icon("image", COLORS["text"]))
        self.import_button.clicked.connect(self._choose_image)
        heading.addWidget(self.import_button)
        root.addLayout(heading)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        root.addWidget(self.progress)
        body = QHBoxLayout()
        body.setSpacing(14)
        self.stepper = WorkflowStepper()
        body.addWidget(self.stepper)
        self.viewer_panel = ViewerPanel()
        body.addWidget(self.viewer_panel, 1)
        self.inspector = TemplateInspector()
        body.addWidget(self.inspector)
        root.addLayout(body, 1)
        self.stepper.step_requested.connect(self._request_step)
        self.viewer_panel.viewer.roi_changed.connect(self._roi_changed)
        self.viewer_panel.mode_changed.connect(self._view_mode_changed)
        self.inspector.name_changed.connect(self._name_changed)
        self.inspector.segment_requested.connect(self._segment_current_roi)
        self.inspector.clear_roi_requested.connect(self._clear_roi)
        self.inspector.edit_mask_requested.connect(self._edit_mask)
        self.inspector.parameters_changed.connect(self._parameters_changed)
        self.inspector.save_requested.connect(self._save_template)

    def _set_state(self, **changes) -> None:
        self.state = self.state.evolve(**changes)
        self._render()

    def _render(self) -> None:
        self.stepper.set_state(self.state.step, self.state.completed_steps(), TemplateStep.MASK if self.state.error else None)
        self.inspector.render(self.state, self.template_service.template_count())
        self.viewer_panel.size_label.setText(self.state.image_size_text)
        if self.state.roi_xywh:
            self.viewer_panel.roi_label.setText(f"ROI {self.state.roi_xywh[2]} × {self.state.roi_xywh[3]}")
        else:
            self.viewer_panel.roi_label.setText("ROI —")
        self.viewer_panel.set_mode(self.state.view_mode)
        self.progress.setVisible(self.state.busy)
        self.import_button.setEnabled(not self.state.busy)
        self.status_changed.emit(self.state.busy_message or self.state.error or "就绪", "error" if self.state.error else ("info" if self.state.busy else "success"))

    def _request_step(self, index: int) -> None:
        step = TemplateStep(index)
        if step >= TemplateStep.ROI and not self.state.has_image:
            self.status_changed.emit("请先导入基准图像", "warning")
            return
        if step >= TemplateStep.MASK and not self.state.has_roi:
            self.status_changed.emit("请先框选 ROI", "warning")
            return
        self._set_state(step=step)
        self.viewer_panel.roi_mode.setChecked(step == TemplateStep.ROI)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择基准图像", str(self.project_root), "图像 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if path:
            self._load_image(Path(path))

    def _load_image(self, path: Path) -> None:
        try:
            image = self.template_service.load_image(path)
        except Exception as exc:
            self._show_error("图像加载失败", str(exc))
            return
        self.viewer_panel.viewer.set_image(image)
        self.viewer_panel.roi_mode.setChecked(True)
        suggested_name = self.state.name or path.stem[:48]
        self.inspector.name_input.setText(suggested_name)
        self._set_state(
            step=TemplateStep.ROI,
            image=image,
            source_path=path,
            name=suggested_name,
            roi_xywh=None,
            mask=None,
            mask_quality=None,
            mask_method="",
            view_mode=ViewMode.ORIGINAL,
            error="",
            saved_template_id=None,
        )

    def _roi_changed(self, roi) -> None:
        if roi is None:
            self._set_state(step=TemplateStep.ROI, roi_xywh=None, mask=None, mask_quality=None, mask_method="", view_mode=ViewMode.ORIGINAL)
            return
        self._set_state(step=TemplateStep.MASK, roi_xywh=tuple(roi), mask=None, mask_quality=None, mask_method="", view_mode=ViewMode.OVERLAY, error="")
        self.viewer_panel.roi_mode.setChecked(False)
        self._segment_current_roi()

    def _clear_roi(self) -> None:
        self.viewer_panel.viewer.clear_roi()
        self.viewer_panel.roi_mode.setChecked(True)

    def _segment_current_roi(self) -> None:
        if not self.state.can_segment:
            return
        image, roi = self.state.image.copy(), self.state.roi_xywh
        self._run_async(
            "正在自动分割 ROI…",
            lambda: self.template_service.segment_roi(image, roi, self.inspector.mask_algorithm()),
            self._segmentation_done,
        )

    def _segmentation_done(self, output: SegmentationOutput) -> None:
        self.viewer_panel.viewer.set_mask(output.mask, self.state.roi_xywh)
        self._set_state(
            step=TemplateStep.MASK,
            mask=output.mask,
            mask_method=output.method_label,
            mask_quality=output.quality,
            view_mode=ViewMode.OVERLAY,
            error="" if output.quality.valid else output.quality.message,
        )

    def _edit_mask(self) -> None:
        if not self.state.has_image or not self.state.has_roi:
            return
        x, y, width, height = self.state.roi_xywh
        crop = self.state.image[y : y + height, x : x + width].copy()
        initial = self.state.mask if self.state.has_mask else __import__("numpy").zeros((height, width), dtype="uint8")
        dialog = MaskEditorDialog(crop, initial, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_mask is not None:
            quality = self.template_service.analyze_mask(dialog.result_mask)
            self.viewer_panel.viewer.set_mask(dialog.result_mask, self.state.roi_xywh)
            self._set_state(mask=dialog.result_mask, mask_method="人工修正", mask_quality=quality, view_mode=ViewMode.OVERLAY, error="" if quality.valid else quality.message)

    def _name_changed(self, value: str) -> None:
        if value != self.state.name:
            self._set_state(name=value, saved_template_id=None)

    def _parameters_changed(self) -> None:
        try:
            parameters = self.inspector.parameters()
            parameters.validate()
        except Exception as exc:
            self._set_state(error=str(exc))
            return
        self._set_state(parameters=parameters, error="")

    def _view_mode_changed(self, value: str) -> None:
        mode = ViewMode(value)
        self.viewer_panel.viewer.set_view_mode(mode)
        if mode != self.state.view_mode:
            self._set_state(view_mode=mode)

    def _save_template(self, continue_to_detection: bool) -> None:
        if not self.state.can_save:
            return
        try:
            entry = self.template_service.save_template(
                name=self.state.name,
                image=self.state.image,
                source_path=self.state.source_path,
                roi_xywh=self.state.roi_xywh,
                mask=self.state.mask,
                parameters=self.state.parameters,
            )
        except Exception as exc:
            self._show_error("模板保存失败", str(exc))
            return
        self._set_state(step=TemplateStep.SAVE, saved_template_id=entry.template_id, error="")
        QMessageBox.information(self, "模板已保存", f"工件类型：{entry.name}\n模板ID：{entry.template_id}\n\n已加入持久模板库。")
        if continue_to_detection:
            self.navigate_requested.emit("detection")

    def _run_async(self, message: str, function, callback) -> None:
        if self._thread is not None:
            return
        thread = QThread(self)
        worker = FunctionWorker(function)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(callback)
        worker.failed.connect(lambda error: self._show_error("处理失败", error))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._task_finished)
        self._thread = thread
        self._worker = worker
        self._set_state(busy=True, busy_message=message, error="")
        thread.start()

    def _task_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_state(busy=False, busy_message="")

    def _show_error(self, title: str, message: str) -> None:
        self._set_state(error=message)
        QMessageBox.critical(self, title, message)

    def load_review_sample(self) -> None:
        """Load checked-in real reference assets for visual QA only."""
        image_path = self.project_root / "samples" / "real_reference" / "anta_reference_roi.png"
        mask_path = self.project_root / "samples" / "real_reference" / "anta_reference_mask.png"
        image = self.template_service.load_image(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError("视觉验收样本 Mask 与图像尺寸不一致")
        roi = (0, 0, image.shape[1], image.shape[0])
        quality = self.template_service.analyze_mask(mask)
        self.viewer_panel.viewer.set_image(image)
        self.viewer_panel.viewer.set_roi(roi)
        self.viewer_panel.viewer.set_mask(mask, roi)
        self.inspector.name_input.setText("Pink_Textile_Reference")
        self._set_state(
            step=TemplateStep.MASK,
            name="Pink_Textile_Reference",
            source_path=image_path,
            image=image,
            roi_xywh=roi,
            mask=mask,
            mask_method="已校验参考 Mask",
            mask_quality=quality,
            view_mode=ViewMode.OVERLAY,
        )
"""Qt 页面：实现正式模板建立工作流及基准图像交互。"""
