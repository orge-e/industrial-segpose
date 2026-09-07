"""Tk 稳定版界面：完成模板建立、旋转目标检测与实时工作站操作。"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from time import perf_counter
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from .. import __version__
from ..production.camera import CameraSettings, OpenCVCameraSource
from ..detectors.fluorescent import FluorescentTextileDetector, FluorescentTextileParameters
from ..production.report import RuntimeReport
from .calibration import CalibrationWorkbench
from ..io.image_reader import read_image, write_image
from ..production.runtime import ConveyorSession, ConveyorFrameResult, draw_conveyor_frame
from ..template_matching import (
    METHOD_LABELS,
    MatchParameters,
    MultiTemplateMatcher,
    MultiTemplateResult,
    RecognizedObject,
    TemplateLibrary,
    TemplateModel,
    analyze_template_mask,
    build_assisted_mask,
    locate_assisted_template,
    draw_multi_template_matches,
    write_multi_template_result,
)
from .demo import prepare_template_demo
from ..production.tracking import TrackingConfig
from ..evaluation.benchmark import EvaluationThresholds, SyntheticStressConfig, run_synthetic_benchmark
from ..template_matching.audit import (
    ParameterSearchConfig,
    ParameterSearchResult,
    TemplateAudit,
    audit_template_library,
    search_template_parameters,
    write_parameter_search_report,
    write_template_audit_report,
)
from ..ui import COLORS, SPACE, configure_industrial_theme


IMAGE_TYPES = [("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")]


def _application_data_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.home() / "IndustrialSegPose"


FEATURE_MODE_LABELS = {
    "pose_tolerant": "柔性姿态",
    "dark_textile": "暗色纹理",
    "white_cut_seam": "白色裁剪缝",
    "textile_chroma": "纺织色度",
    "edges": "轮廓边缘",
    "gray": "灰度纹理",
    "auto": "智能多模式",
}


def template_workflow_status(
    *, has_image: bool, has_roi: bool, has_mask: bool, has_name: bool
) -> tuple[str, str]:
    """Return the next action for the desktop template-authoring workflow."""

    if not has_image:
        return "第 1 步 / 3：导入基准图像", "导入一张只包含单个工件的清晰图像，程序会优先自动定位和分割。"
    if not has_roi:
        return "第 1 步 / 3：确认工件范围", "在图像中拖框选择工件，或重新执行自动分割。"
    if not has_mask:
        return "第 2 步 / 3：检查分割轮廓", "当前只有粗 ROI；建议检查并修正 Mask，减少背景对匹配的干扰。"
    if not has_name:
        return "第 3 步 / 3：填写工件类型", "填写明确且唯一的模板名称，然后保存到模板库。"
    return "准备完成：可以保存模板", "保存后可直接进入检测页，用现场图像验证中心、角度和数量。"


def detection_workflow_status(
    *, valid_template_count: int, has_image: bool, running: bool, has_result: bool
) -> tuple[str, str]:
    """Return the next action for the offline-detection workflow."""

    if valid_template_count <= 0:
        return "检测未就绪：没有可用模板", "请先在模板库中启用至少一个状态正常的模板。"
    if not has_image:
        return "第 1 步 / 2：选择检测图像", f"当前有 {valid_template_count} 个有效模板可用；请选择待检测图像。"
    if running:
        return "正在检测", "正在运行全部已启用模板，请等待结果完成。"
    if not has_result:
        return "第 2 步 / 2：开始检测", f"图像与 {valid_template_count} 个模板已就绪，点击“开始检测”。"
    return "检测完成：可检查并保存结果", "结果表给出类型、中心坐标、旋转角度和当前画面数量。"


class ImageCanvas(tk.Canvas):
    """Fit-to-window image canvas with an optional draggable ROI."""

    def __init__(self, master, selectable: bool = False, **kwargs):
        super().__init__(master, background=COLORS["canvas"], highlightthickness=0, **kwargs)
        self.selectable = selectable
        self.on_view_changed = None
        self.image: np.ndarray | None = None
        self.photo: tk.PhotoImage | None = None
        self.scale = 1.0
        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.roi: tuple[int, int, int, int] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._pan_start: tuple[int, int] | None = None
        self._pan_origin: tuple[float, float] | None = None
        self.bind("<Configure>", lambda _event: self._render())
        self.bind("<MouseWheel>", self._mouse_wheel)
        self.bind("<Button-4>", lambda event: self._zoom_at(1.2, event.x, event.y))
        self.bind("<Button-5>", lambda event: self._zoom_at(1 / 1.2, event.x, event.y))
        self.bind("<ButtonPress-2>", self._start_pan)
        self.bind("<B2-Motion>", self._move_pan)
        self.bind("<ButtonRelease-2>", self._end_pan)
        if selectable:
            self.bind("<ButtonPress-1>", self._press)
            self.bind("<B1-Motion>", self._drag)
            self.bind("<ButtonRelease-1>", self._release)

    def set_image(self, image: np.ndarray | None, clear_roi: bool = True, reset_view: bool = True) -> None:
        self.image = image.copy() if image is not None else None
        if clear_roi:
            self.roi = None
        if reset_view:
            self.zoom_factor = 1.0
            self.pan_x = self.pan_y = 0.0
        self._render()
        self._notify_view_changed()

    def zoom_in(self) -> None:
        self._zoom_at(1.25, self.winfo_width() / 2, self.winfo_height() / 2)

    def zoom_out(self) -> None:
        self._zoom_at(0.8, self.winfo_width() / 2, self.winfo_height() / 2)

    def fit_to_window(self) -> None:
        self.zoom_factor = 1.0
        self.pan_x = self.pan_y = 0.0
        self._render()
        self._notify_view_changed()

    def _notify_view_changed(self) -> None:
        callback = self.on_view_changed
        if callback is not None:
            callback()

    def _mouse_wheel(self, event) -> None:
        self._zoom_at(1.2 if event.delta > 0 else 1 / 1.2, event.x, event.y)

    def _zoom_at(self, multiplier: float, canvas_x: float, canvas_y: float) -> None:
        if self.image is None:
            return
        old_scale = max(self.scale, 1e-9)
        image_x = (canvas_x - self.offset_x) / old_scale
        image_y = (canvas_y - self.offset_y) / old_scale
        old_zoom = self.zoom_factor
        self.zoom_factor = min(12.0, max(0.5, old_zoom * multiplier))
        if abs(self.zoom_factor - old_zoom) < 1e-9:
            return
        source = self._render_source()
        height, width = source.shape[:2]
        fit = min(self.winfo_width() / width, self.winfo_height() / height)
        new_scale = fit * self.zoom_factor
        centered_x = (self.winfo_width() - width * new_scale) / 2.0
        centered_y = (self.winfo_height() - height * new_scale) / 2.0
        self.pan_x = canvas_x - image_x * new_scale - centered_x
        self.pan_y = canvas_y - image_y * new_scale - centered_y
        self._render()
        self._notify_view_changed()

    def _start_pan(self, event) -> None:
        self._pan_start = (event.x, event.y)
        self._pan_origin = (self.pan_x, self.pan_y)
        self.configure(cursor="fleur")

    def _move_pan(self, event) -> None:
        if self._pan_start is None or self._pan_origin is None:
            return
        self.pan_x = self._pan_origin[0] + event.x - self._pan_start[0]
        self.pan_y = self._pan_origin[1] + event.y - self._pan_start[1]
        self._render()

    def _end_pan(self, _event) -> None:
        self._pan_start = self._pan_origin = None
        self.configure(cursor="")

    def _render(self) -> None:
        self.delete("all")
        if self.image is None or self.winfo_width() < 2 or self.winfo_height() < 2:
            self.create_text(
                max(self.winfo_width() // 2, 1),
                max(self.winfo_height() // 2, 1),
                text="尚未加载图像",
                fill=COLORS["text_secondary"],
                font=("Microsoft YaHei UI", 13),
            )
            return
        source = self._render_source()
        height, width = source.shape[:2]
        self.fit_scale = min(self.winfo_width() / width, self.winfo_height() / height)
        self.scale = self.fit_scale * self.zoom_factor
        display_width = max(1, int(round(width * self.scale)))
        display_height = max(1, int(round(height * self.scale)))
        centered_x = (self.winfo_width() - display_width) / 2.0
        centered_y = (self.winfo_height() - display_height) / 2.0
        self.offset_x = centered_x + self.pan_x
        self.offset_y = centered_y + self.pan_y
        if display_width <= self.winfo_width():
            self.offset_x, self.pan_x = centered_x, 0.0
        else:
            self.offset_x = min(0.0, max(self.winfo_width() - display_width, self.offset_x))
            self.pan_x = self.offset_x - centered_x
        if display_height <= self.winfo_height():
            self.offset_y, self.pan_y = centered_y, 0.0
        else:
            self.offset_y = min(0.0, max(self.winfo_height() - display_height, self.offset_y))
            self.pan_y = self.offset_y - centered_y
        display = cv2.resize(source, (display_width, display_height), interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".png", display)
        if not ok:
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=data)
        self.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw", tags="image")
        self._draw_roi()

    def _render_source(self) -> np.ndarray:
        return self.image

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        if self.image is None:
            return 0, 0
        height, width = self.image.shape[:2]
        ix = int(round((x - self.offset_x) / max(self.scale, 1e-9)))
        iy = int(round((y - self.offset_y) / max(self.scale, 1e-9)))
        return min(max(ix, 0), width), min(max(iy, 0), height)

    def _press(self, event) -> None:
        if self.image is None:
            return
        self._drag_start = self._canvas_to_image(event.x, event.y)
        self.roi = (*self._drag_start, 0, 0)
        self._draw_roi()

    def _drag(self, event) -> None:
        if self._drag_start is None:
            return
        end_x, end_y = self._canvas_to_image(event.x, event.y)
        start_x, start_y = self._drag_start
        x, y = min(start_x, end_x), min(start_y, end_y)
        self.roi = (x, y, abs(end_x - start_x), abs(end_y - start_y))
        self._draw_roi()

    def _release(self, event) -> None:
        self._drag(event)
        self._drag_start = None
        self.event_generate("<<RoiChanged>>")

    def _draw_roi(self) -> None:
        self.delete("roi")
        if not self.roi:
            return
        x, y, width, height = self.roi
        x1, y1 = self.offset_x + x * self.scale, self.offset_y + y * self.scale
        x2, y2 = self.offset_x + (x + width) * self.scale, self.offset_y + (y + height) * self.scale
        self.create_rectangle(x1, y1, x2, y2, outline=COLORS["primary"], width=2, dash=(6, 3), tags="roi")
        self.create_text(x1 + 4, max(y1 - 12, 10), text=f"{width} x {height}", anchor="w", fill=COLORS["primary_hover"], tags="roi")


class MaskEditorCanvas(ImageCanvas):
    """Polygon and brush editor for an irregular template mask."""

    def __init__(self, master, image: np.ndarray, initial_mask: np.ndarray | None = None, **kwargs):
        super().__init__(master, selectable=False, **kwargs)
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8) if initial_mask is None else np.where(initial_mask > 0, 255, 0).astype(np.uint8)
        self.tool = "polygon"
        self.brush_radius = 12
        self.polygon_points: list[tuple[int, int]] = []
        self._painting = False
        self._erase = False
        self.bind("<ButtonPress-1>", self._press_tool)
        self.bind("<B1-Motion>", self._move_tool)
        self.bind("<ButtonRelease-1>", self._release_tool)
        self.bind("<ButtonPress-3>", self._press_erase)
        self.bind("<B3-Motion>", self._move_tool)
        self.bind("<ButtonRelease-3>", self._release_tool)
        self.bind("<Double-Button-1>", lambda _event: self.finish_polygon())
        self.set_image(image)

    def _render_source(self) -> np.ndarray:
        source = self.image.copy()
        selected = self.mask > 0
        if np.any(selected):
            tint = np.zeros_like(source)
            tint[:, :, 0] = 255
            source[selected] = cv2.addWeighted(source, 0.58, tint, 0.42, 0)[selected]
        return source

    def _render(self) -> None:
        super()._render()
        if len(self.polygon_points) >= 1:
            points = [(self.offset_x + x * self.scale, self.offset_y + y * self.scale) for x, y in self.polygon_points]
            flattened = [coordinate for point in points for coordinate in point]
            if len(points) >= 2:
                self.create_line(*flattened, fill=COLORS["primary"], width=2, tags="polygon")
            for x, y in points:
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=COLORS["primary"], outline="", tags="polygon")

    def _paint(self, event) -> None:
        x, y = self._canvas_to_image(event.x, event.y)
        value = 0 if self._erase else 255
        cv2.circle(self.mask, (x, y), int(self.brush_radius), value, -1, cv2.LINE_AA)
        self._render()

    def _press_tool(self, event) -> None:
        if self.tool == "polygon":
            self.polygon_points.append(self._canvas_to_image(event.x, event.y))
            self._render()
        else:
            self._painting, self._erase = True, False
            self._paint(event)

    def _press_erase(self, event) -> None:
        self._painting, self._erase = True, True
        self._paint(event)

    def _move_tool(self, event) -> None:
        if self._painting:
            self._paint(event)

    def _release_tool(self, _event) -> None:
        self._painting = False

    def finish_polygon(self) -> None:
        if len(self.polygon_points) >= 3:
            cv2.fillPoly(self.mask, [np.asarray(self.polygon_points, dtype=np.int32)], 255)
        self.polygon_points.clear()
        self._render()

    def clear_mask(self) -> None:
        self.mask.fill(0)
        self.polygon_points.clear()
        self._render()

    def fill_all(self) -> None:
        self.mask.fill(255)
        self.polygon_points.clear()
        self._render()


class MaskEditorDialog(tk.Toplevel):
    def __init__(self, master, image: np.ndarray, initial_mask: np.ndarray | None = None):
        super().__init__(master)
        self.configure(background=COLORS["app"])
        self.title("编辑不规则模板形状")
        self.geometry("920x680")
        self.minsize(700, 520)
        self.transient(master)
        self.result: np.ndarray | None = None
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="工具：").pack(side="left")
        self.tool_var = tk.StringVar(value="polygon")
        ttk.Radiobutton(controls, text="多边形", value="polygon", variable=self.tool_var, command=self._tool_changed).pack(side="left")
        ttk.Radiobutton(controls, text="画笔", value="brush", variable=self.tool_var, command=self._tool_changed).pack(side="left")
        ttk.Label(controls, text="画笔半径：").pack(side="left", padx=(16, 3))
        self.brush_var = tk.IntVar(value=12)
        ttk.Spinbox(controls, from_=2, to=100, textvariable=self.brush_var, width=6, command=self._brush_changed).pack(side="left")
        ttk.Button(controls, text="完成多边形", command=lambda: self.canvas.finish_polygon()).pack(side="left", padx=(16, 4))
        ttk.Button(controls, text="整个 ROI", command=lambda: self.canvas.fill_all()).pack(side="left", padx=4)
        ttk.Button(controls, text="清空", command=lambda: self.canvas.clear_mask()).pack(side="left", padx=4)
        ttk.Button(controls, text="确认", command=self._accept).pack(side="right", padx=4)
        ttk.Button(controls, text="取消", command=self.destroy).pack(side="right", padx=4)
        assist = ttk.Frame(self, padding=(8, 0, 8, 5))
        assist.pack(fill="x")
        ttk.Label(assist, text="算法辅助：").pack(side="left")
        self.assist_label_to_method = {label: method for method, label in METHOD_LABELS.items()}
        self.assist_method_var = tk.StringVar(value=METHOD_LABELS["textile_chroma"])
        ttk.Combobox(assist, textvariable=self.assist_method_var, values=list(self.assist_label_to_method), state="readonly", width=18).pack(side="left", padx=(0, 5))
        ttk.Button(assist, text="自动提取初始形状", command=self._run_assist).pack(side="left")
        self.assist_status_var = tk.StringVar(value="算法结果只作为初始 mask，请检查蓝色区域并用画笔修正。")
        ttk.Label(assist, textvariable=self.assist_status_var, style="PageHint.TLabel").pack(side="left", padx=10)
        ttk.Label(self, text="多边形：依次点击轮廓点，双击或点击“完成多边形”；画笔：左键添加，右键擦除。蓝色区域为模板有效形状。", style="PageHint.TLabel", padding=(10, 0)).pack(fill="x")
        self.canvas = MaskEditorCanvas(self, image, initial_mask)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()
        self.wait_visibility()
        self.focus_set()

    def _tool_changed(self) -> None:
        self.canvas.finish_polygon()
        self.canvas.tool = self.tool_var.get()

    def _brush_changed(self) -> None:
        try:
            self.canvas.brush_radius = max(2, int(self.brush_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _run_assist(self) -> None:
        method = self.assist_label_to_method.get(self.assist_method_var.get(), "auto")
        self.configure(cursor="watch")
        self.assist_status_var.set("正在进行图像处理……")
        self.update_idletasks()
        try:
            result = build_assisted_mask(self.canvas.image, method)
            if np.count_nonzero(result.mask) < 25:
                raise ValueError("算法没有提取到有效目标，请换一种算法或手工标注")
            self.canvas.mask = result.mask.copy()
            self.canvas.polygon_points.clear()
            self.canvas._render()
            self.assist_status_var.set(f"{result.method_label}：覆盖率 {result.coverage:.1%}，质量指标 {result.quality_score:.2f}；请人工复核。")
        except Exception as exc:
            self.assist_status_var.set("自动提取失败，可更换算法或手工标注。")
            messagebox.showwarning("辅助提取失败", str(exc), parent=self)
        finally:
            self.configure(cursor="")

    def _accept(self) -> None:
        self.canvas.finish_polygon()
        if int(np.count_nonzero(self.canvas.mask)) < 25:
            messagebox.showwarning("形状为空", "请标注至少 25 个有效像素。", parent=self)
            return
        self.result = self.canvas.mask.copy()
        self.destroy()


class TemplateParameterDialog(tk.Toplevel):
    """Edit the independent matching parameters of one library template."""

    FIELDS = (
        ("score_threshold", "匹配阈值"), ("angle_min", "最小角度"),
        ("angle_max", "最大角度"), ("angle_step", "角度步长"),
        ("scale_min", "最小尺度"), ("scale_max", "最大尺度"),
        ("scale_step", "尺度步长"), ("nms_iou_threshold", "NMS IoU"),
    )

    def __init__(self, master, template_name: str, parameters: MatchParameters):
        super().__init__(master)
        self.configure(background=COLORS["app"])
        self.title(f"编辑匹配参数 - {template_name}")
        self.resizable(False, False)
        self.transient(master)
        self.result: MatchParameters | None = None
        self.original = parameters
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        self.variables: dict[str, tk.StringVar] = {}
        for index, (key, label) in enumerate(self.FIELDS):
            ttk.Label(body, text=label).grid(row=index, column=0, sticky="e", padx=6, pady=4)
            variable = tk.StringVar(value=str(getattr(parameters, key)))
            self.variables[key] = variable
            ttk.Entry(body, textvariable=variable, width=18).grid(row=index, column=1, padx=6, pady=4)
        self.use_edges_var = tk.BooleanVar(value=parameters.use_edges)
        ttk.Label(body, text="特征模式").grid(row=len(self.FIELDS), column=0, sticky="e", padx=6, pady=4)
        self.feature_mode_var = tk.StringVar(value=FEATURE_MODE_LABELS.get(parameters.feature_mode, "兼容模式"))
        self.feature_label_to_mode = {label: mode for mode, label in FEATURE_MODE_LABELS.items()}
        ttk.Combobox(body, textvariable=self.feature_mode_var, values=list(self.feature_label_to_mode), state="readonly", width=16).grid(row=len(self.FIELDS), column=1, padx=6, pady=4)
        buttons = ttk.Frame(body)
        buttons.grid(row=len(self.FIELDS) + 1, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text="确定", command=self._accept).pack(side="left", padx=5)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="left", padx=5)
        self.grab_set()
        self.wait_visibility()

    def _accept(self) -> None:
        try:
            self.result = MatchParameters(
                score_threshold=float(self.variables["score_threshold"].get()),
                angle_min=float(self.variables["angle_min"].get()),
                angle_max=float(self.variables["angle_max"].get()),
                angle_step=float(self.variables["angle_step"].get()),
                scale_min=float(self.variables["scale_min"].get()),
                scale_max=float(self.variables["scale_max"].get()),
                scale_step=float(self.variables["scale_step"].get()),
                nms_iou_threshold=float(self.variables["nms_iou_threshold"].get()),
                use_edges=self.use_edges_var.get(),
                feature_mode=self.feature_label_to_mode.get(self.feature_mode_var.get(), "auto"),
                max_candidates_per_transform=self.original.max_candidates_per_transform,
                max_results=self.original.max_results,
                coarse_to_fine=self.original.coarse_to_fine,
                coarse_trigger_transforms=self.original.coarse_trigger_transforms,
                coarse_angle_step=self.original.coarse_angle_step,
                coarse_scale_step=self.original.coarse_scale_step,
                refine_transform_limit=self.original.refine_transform_limit,
            )
            self.result.validate()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        self.destroy()


class TemplateMatchingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FlexPose Vision - 柔性工件定位系统")
        self.geometry("1280x820")
        self.minsize(1120, 700)
        self.reference_path: Path | None = None
        self.reference_image: np.ndarray | None = None
        self.template_mask: np.ndarray | None = None
        self.detection_path: Path | None = None
        self.detection_image: np.ndarray | None = None
        self.annotated_image: np.ndarray | None = None
        self.multi_result: MultiTemplateResult | None = None
        self.multi_matcher: MultiTemplateMatcher | None = None
        self.data_root = _application_data_root()
        self.template_library = TemplateLibrary(self.data_root / "templates")
        self.library_load_error: str | None = None
        try:
            self.template_library.load()
        except Exception as exc:
            self.library_load_error = str(exc)
        self.worker_queue: queue.Queue = queue.Queue()
        self.validation_queue: queue.Queue = queue.Queue()
        self.validation_running = False
        self.detection_running = False
        self.template_audits: list[TemplateAudit] = []
        self.parameter_search_result: ParameterSearchResult | None = None
        # Keep only the newest processed frame. High-resolution raw + annotated
        # images are large and must not accumulate while the UI is repainting.
        self.live_queue: queue.Queue = queue.Queue(maxsize=1)
        self.live_source: OpenCVCameraSource | None = None
        self.live_session: ConveyorSession | None = None
        self.live_stop_event = threading.Event()
        self.live_frame_lock = threading.Lock()
        self.live_latest_frame: np.ndarray | None = None
        self.live_frame_sequence = 0
        self.live_running = False
        self.live_detection_roi: tuple[int, int, int, int] | None = None
        self.live_last_raw_frame: np.ndarray | None = None
        self.live_last_annotated_frame: np.ndarray | None = None
        self.live_last_result: ConveyorFrameResult | None = None
        self.live_runtime_report: RuntimeReport | None = None
        self._configure_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        self.style = configure_industrial_theme(self)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, height=56, padding=(16, 9), style="Header.TFrame")
        header.pack(fill="x")
        header.pack_propagate(False)
        ttk.Label(header, text="FP", style="HeaderBadge.TLabel").pack(side="left", padx=(0, SPACE["md"]))
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left", fill="y")
        ttk.Label(brand, text="FlexPose Vision", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(brand, text="INDUSTRIAL VISION WORKSTATION", style="HeaderSub.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"DESKTOP  ·  v{__version__}", style="HeaderChip.TLabel").pack(side="right")
        self.device_header_var = tk.StringVar(value="●  SYSTEM READY")
        ttk.Label(header, textvariable=self.device_header_var, style="HeaderSub.TLabel").pack(side="right", padx=(0, SPACE["lg"]))

        # These variables remain for the existing workflow state machine and
        # status messages, but the former large workflow banner is removed.
        self.workflow_title_var = tk.StringVar(value="第 1 步 / 3：导入基准图像")
        self.workflow_hint_var = tk.StringVar(value="从建立模板开始，页面会根据当前状态提示下一步。")

        self.status_var = tk.StringVar(value="就绪")
        footer = ttk.Frame(self, height=34, padding=(12, 0), style="Footer.TFrame")
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        self.footer_device_var = tk.StringVar(value="DEVICE  LOCAL")
        self.footer_image_var = tk.StringVar(value="IMAGE  —")
        self.footer_processing_var = tk.StringVar(value="PROCESSING  IDLE")
        for variable in (self.footer_device_var, self.footer_image_var, self.footer_processing_var):
            ttk.Label(footer, textvariable=variable, style="Footer.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.status_var, anchor="e", style="Footer.TLabel").pack(side="right", fill="x", expand=True)

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True)

        navigation = ttk.Frame(body, width=188, padding=(10, 14), style="Sidebar.TFrame")
        navigation.pack(side="left", fill="y")
        navigation.pack_propagate(False)
        ttk.Label(navigation, text="WORKSPACE", style="NavSection.TLabel").pack(anchor="w", padx=8, pady=(0, 8))

        self.notebook = ttk.Notebook(body, style="Workspace.TNotebook")
        self.notebook.pack(side="left", fill="both", expand=True, padx=(1, 0))
        self.template_tab = ttk.Frame(self.notebook, padding=SPACE["lg"], style="App.TFrame")
        self.detect_tab = ttk.Frame(self.notebook, padding=SPACE["md"], style="App.TFrame")
        self.notebook.add(self.template_tab, text="建立模板")
        self.notebook.add(self.detect_tab, text="模板检测")
        self._build_template_tab()
        self._build_detect_tab()
        self.live_tab = ttk.Frame(self.notebook, padding=SPACE["md"], style="App.TFrame")
        self.notebook.add(self.live_tab, text="实时视觉")
        self._build_live_tab()
        self.validation_tab = ttk.Frame(self.notebook, padding=SPACE["md"], style="App.TFrame")
        self.notebook.add(self.validation_tab, text="算法验证")
        self._build_validation_tab()

        self.nav_buttons: dict[tk.Widget, ttk.Button] = {}
        for label, page in (
            ("建立模板", self.template_tab),
            ("模板检测", self.detect_tab),
            ("实时视觉", self.live_tab),
            ("算法验证", self.validation_tab),
        ):
            button = ttk.Button(
                navigation,
                text=label,
                command=lambda target=page: self._select_page(target),
                style="Nav.TButton",
            )
            button.pack(fill="x", pady=2)
            self.nav_buttons[page] = button
        ttk.Separator(navigation).pack(fill="x", padx=8, pady=(SPACE["xl"], SPACE["md"]))
        ttk.Label(navigation, text="TOOLS", style="NavSection.TLabel").pack(anchor="w", padx=8, pady=(0, 8))
        ttk.Button(navigation, text="标定工具", command=self.open_calibration_workbench, style="Nav.TButton").pack(fill="x", pady=2)
        ttk.Button(navigation, text="设置", state="disabled", style="Nav.TButton").pack(fill="x", pady=2)
        ttk.Label(navigation, text="OpenCV / CPU", style="NavSection.TLabel").pack(side="bottom", anchor="w", padx=8, pady=8)

        self.notebook.bind("<<NotebookTabChanged>>", self._notebook_tab_changed)
        self._refresh_library_tree()
        self.refresh_template_audits(show_status=False)
        self._update_ui_state()
        if self.library_load_error:
            self.status_var.set(f"模板库加载失败：{self.library_load_error}")

    def _select_page(self, page: tk.Widget) -> None:
        self.notebook.select(page)

    def _refresh_navigation(self, selected: tk.Widget) -> None:
        for page, button in getattr(self, "nav_buttons", {}).items():
            button.configure(style="NavActive.TButton" if page is selected else "Nav.TButton")

    def _notebook_tab_changed(self, _event=None) -> None:
        self._update_ui_state()
        if not hasattr(self, "validation_tab") or self.validation_running:
            return
        try:
            selected = self.nametowidget(self.notebook.select())
        except (KeyError, tk.TclError):
            return
        self._refresh_navigation(selected)
        if selected is self.validation_tab:
            self.refresh_template_audits(show_status=False)

    def _valid_enabled_template_count(self) -> int:
        try:
            return sum(item.valid and item.entry.enabled for item in self.template_library.load_entries())
        except Exception:
            return 0

    def _update_ui_state(self) -> None:
        """Synchronize guidance and button availability with the current task state."""

        if not hasattr(self, "notebook"):
            return
        roi = self.reference_canvas.roi if hasattr(self, "reference_canvas") else None
        valid_roi = bool(roi and roi[2] >= 5 and roi[3] >= 5)
        has_name = bool(getattr(self, "template_name_var", None) and self.template_name_var.get().strip())
        template_title, template_hint = template_workflow_status(
            has_image=self.reference_image is not None,
            has_roi=valid_roi,
            has_mask=self.template_mask is not None,
            has_name=has_name,
        )
        if hasattr(self, "edit_mask_button"):
            self.edit_mask_button.configure(state="normal" if self.reference_image is not None and valid_roi else "disabled")
        template_can_save = self.reference_image is not None and valid_roi and has_name
        for name in ("save_template_button", "save_and_detect_button"):
            if hasattr(self, name):
                getattr(self, name).configure(state="normal" if template_can_save else "disabled")

        valid_count = self._valid_enabled_template_count()
        detect_title, detect_hint = detection_workflow_status(
            valid_template_count=valid_count,
            has_image=self.detection_image is not None,
            running=self.detection_running,
            has_result=self.multi_result is not None,
        )
        if hasattr(self, "detect_button"):
            ready = valid_count > 0 and self.detection_image is not None and not self.detection_running
            self.detect_button.configure(state="normal" if ready else "disabled")
        if hasattr(self, "save_button"):
            self.save_button.configure(
                state="normal" if self.multi_result is not None and self.annotated_image is not None and not self.detection_running else "disabled"
            )

        if hasattr(self, "parameter_search_button"):
            selected_audit = self._selected_audit_template_id()
            self.parameter_search_button.configure(
                state="normal" if selected_audit and not self.validation_running else "disabled"
            )
        if hasattr(self, "validation_benchmark_button"):
            self.validation_benchmark_button.configure(
                state="normal" if valid_count > 0 and not self.validation_running else "disabled"
            )
        if hasattr(self, "apply_search_button"):
            self.apply_search_button.configure(
                state="normal" if self.parameter_search_result is not None and not self.validation_running else "disabled"
            )
        if hasattr(self, "apply_recommendation_button"):
            self.apply_recommendation_button.configure(
                state="normal" if self._selected_audit_template_id() and not self.validation_running else "disabled"
            )

        try:
            selected = self.nametowidget(self.notebook.select())
        except (KeyError, tk.TclError):
            return
        if selected is self.template_tab:
            title, hint = template_title, template_hint
        elif selected is self.detect_tab:
            title, hint = detect_title, detect_hint
        elif hasattr(self, "live_tab") and selected is self.live_tab:
            title = "实时视觉：选择采集源后启动"
            hint = "先确认相机或视频源，再设置检测算法与 ROI；运行日志会记录采集和检测异常。"
        else:
            title = "算法验证：先选择模板，再执行诊断或压力测试"
            hint = "模式建议和参数搜索不会自动覆盖模板，只有点击应用后才会写入模板库。"
        if hasattr(self, "workflow_title_var"):
            self.workflow_title_var.set(title)
            self.workflow_hint_var.set(hint)
        self._refresh_navigation(selected)
        if hasattr(self, "template_step_items"):
            self._update_template_stepper()
        if hasattr(self, "inspector_roi_var"):
            self._update_template_inspector()

    def _build_template_tab(self) -> None:
        heading = ttk.Frame(self.template_tab, style="App.TFrame")
        heading.pack(fill="x", pady=(0, SPACE["md"]))
        ttk.Label(heading, text="建立模板", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            heading,
            text="从单工件参考图像建立可复用的 ROI 与 Mask，并加入持久模板库。",
            style="PageHint.TLabel",
        ).pack(anchor="w", pady=(SPACE["xs"], 0))

        workspace = ttk.Frame(self.template_tab, style="App.TFrame")
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(0, minsize=182)
        workspace.columnconfigure(1, weight=1, minsize=360)
        workspace.columnconfigure(2, minsize=284)
        workspace.rowconfigure(0, weight=1)

        # Left: compact workflow stepper.
        stepper = ttk.Frame(workspace, width=182, padding=(12, 16), style="Sidebar.TFrame")
        stepper.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE["md"]))
        stepper.pack_propagate(False)
        ttk.Label(stepper, text="WORKFLOW", style="NavSection.TLabel").pack(anchor="w", padx=6, pady=(0, SPACE["md"]))
        self.template_step_items: list[tuple[ttk.Frame, ttk.Label, ttk.Label]] = []
        for index, label in enumerate(("基准图像", "ROI", "Mask", "模板参数", "保存部署"), start=1):
            item = ttk.Frame(stepper, padding=(8, 10), style="StepperIdle.TFrame")
            item.pack(fill="x", pady=2)
            number = ttk.Label(item, text=f"{index:02d}", width=3, style="StepperIdleIndex.TLabel")
            number.pack(side="left")
            title = ttk.Label(item, text=label, style="StepperIdle.TLabel")
            title.pack(side="left", padx=(SPACE["sm"], 0))
            self.template_step_items.append((item, number, title))
        ttk.Label(
            stepper,
            text="蓝色表示当前步骤\n绿色表示已完成\n红色表示需要修正",
            style="NavSection.TLabel",
            justify="left",
        ).pack(side="bottom", anchor="w", padx=6, pady=6)

        # Center: the reference image is the visual focus.
        viewer = ttk.Frame(workspace, padding=(12, 10), style="Panel.TFrame")
        viewer.grid(row=0, column=1, sticky="nsew", padx=(0, SPACE["md"]))
        viewer_toolbar = ttk.Frame(viewer, style="Toolbar.TFrame")
        viewer_toolbar.pack(fill="x", pady=(0, SPACE["sm"]))
        ttk.Label(viewer_toolbar, text="REFERENCE IMAGE", style="InspectorTitle.TLabel").pack(side="left")
        self.reference_view_var = tk.StringVar(value="original")
        self.reference_view_buttons: dict[str, ttk.Button] = {}
        for mode, label in (("original", "Original"), ("mask", "Mask"), ("overlay", "Overlay")):
            button = ttk.Button(
                viewer_toolbar,
                text=label,
                command=lambda selected=mode: self._set_reference_view(selected),
                style="Ghost.TButton",
            )
            button.pack(side="left", padx=(SPACE["md"] if mode == "original" else 2, 0))
            self.reference_view_buttons[mode] = button
        ttk.Button(viewer_toolbar, text="放大", command=lambda: self.reference_canvas.zoom_in(), style="Icon.TButton").pack(side="right", padx=(2, 0))
        ttk.Button(viewer_toolbar, text="缩小", command=lambda: self.reference_canvas.zoom_out(), style="Icon.TButton").pack(side="right", padx=(2, 0))
        ttk.Button(viewer_toolbar, text="适应", command=lambda: self.reference_canvas.fit_to_window(), style="Ghost.TButton").pack(side="right", padx=(SPACE["sm"], 2))

        self.reference_canvas = ImageCanvas(viewer, selectable=True)
        self.reference_canvas.pack(fill="both", expand=True)
        self.reference_canvas.bind("<<RoiChanged>>", self._roi_changed)
        self.reference_canvas.on_view_changed = self._update_reference_view_status
        viewer_status = ttk.Frame(viewer, padding=(2, 7, 2, 0), style="Toolbar.TFrame")
        viewer_status.pack(fill="x")
        self.reference_image_info_var = tk.StringVar(value="— × —   RGB")
        self.reference_zoom_var = tk.StringVar(value="ZOOM  FIT")
        ttk.Label(viewer_status, textvariable=self.reference_image_info_var, style="InspectorLabel.TLabel").pack(side="left")
        ttk.Label(viewer_status, text="滚轮缩放  ·  中键平移  ·  左键框选 ROI", style="InspectorLabel.TLabel").pack(side="left", padx=SPACE["xl"])
        ttk.Label(viewer_status, textvariable=self.reference_zoom_var, style="InspectorValue.TLabel").pack(side="right")

        # Right: template properties and authoring actions.
        inspector = ttk.Frame(workspace, width=284, padding=(16, 16), style="Inspector.TFrame")
        inspector.grid(row=0, column=2, sticky="nsew")
        inspector.pack_propagate(False)
        ttk.Label(inspector, text="TEMPLATE", style="InspectorTitle.TLabel").pack(anchor="w", pady=(0, SPACE["lg"]))
        ttk.Label(inspector, text="Name", style="InspectorLabel.TLabel").pack(anchor="w")
        self.template_name_var = tk.StringVar(value="")
        self.template_name_var.trace_add("write", lambda *_args: self._update_ui_state())
        ttk.Entry(inspector, textvariable=self.template_name_var).pack(fill="x", pady=(SPACE["xs"], SPACE["lg"]))

        ttk.Label(inspector, text="SOURCE", style="InspectorTitle.TLabel").pack(anchor="w")
        self.inspector_source_var = tk.StringVar(value="No image loaded")
        ttk.Label(inspector, textvariable=self.inspector_source_var, style="InspectorValue.TLabel", wraplength=245, justify="left").pack(anchor="w", pady=(SPACE["xs"], SPACE["sm"]))
        ttk.Button(inspector, text="导入并自动分割", command=lambda: self.load_reference(auto_extract=True), style="Primary.TButton").pack(fill="x", pady=2)
        ttk.Button(inspector, text="手动加载图像", command=self.load_reference, style="Ghost.TButton").pack(fill="x", pady=2)

        ttk.Separator(inspector).pack(fill="x", pady=SPACE["lg"])
        self.roi_var = tk.StringVar(value="请先加载图像，然后框选单个工件")
        self.inspector_roi_var = tk.StringVar(value="Not selected")
        ttk.Label(inspector, text="ROI", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(inspector, textvariable=self.inspector_roi_var, style="InspectorValue.TLabel").pack(anchor="w", pady=(SPACE["xs"], SPACE["md"]))

        self.template_quality_var = tk.StringVar(value="Mask质量：尚未建立")
        self.inspector_mask_coverage_var = tk.StringVar(value="Coverage  —")
        self.inspector_mask_quality_var = tk.StringVar(value="NOT CREATED")
        ttk.Label(inspector, text="MASK", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(inspector, textvariable=self.inspector_mask_coverage_var, style="InspectorValue.TLabel").pack(anchor="w", pady=(SPACE["xs"], 2))
        self.inspector_quality_label = ttk.Label(inspector, textvariable=self.inspector_mask_quality_var, style="InspectorLabel.TLabel")
        self.inspector_quality_label.pack(anchor="w", pady=(0, SPACE["sm"]))
        self.edit_mask_button = ttk.Button(inspector, text="检查并修正分割轮廓", command=self.edit_template_shape, style="Secondary.TButton")
        self.edit_mask_button.pack(fill="x")

        ttk.Separator(inspector).pack(fill="x", pady=SPACE["lg"])
        ttk.Label(inspector, text="MATCH PARAMETERS", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(inspector, text="Threshold 0.72  ·  Angle 360°\nScale 0.50–1.80  ·  Auto feature", style="InspectorValue.TLabel", justify="left").pack(anchor="w", pady=(SPACE["xs"], 0))

        # Bottom: two clear primary outcomes, with the legacy shortcut retained
        # as a low-emphasis action.
        actions = ttk.Frame(self.template_tab, padding=(16, 12), style="Panel.TFrame")
        actions.pack(fill="x", pady=(SPACE["md"], 0))
        ttk.Label(actions, text="模板将保存到电脑端持久模板库，可直接用于检测。", style="Hint.TLabel").pack(side="left")
        self.save_template_button = ttk.Button(actions, text="保存模板", command=self.save_template, style="Primary.TButton")
        self.save_template_button.pack(side="right", padx=(0, SPACE["sm"]))
        self.save_and_detect_button = ttk.Button(
            actions,
            text="保存并进入检测",
            command=lambda: self.save_template(continue_to_detection=True),
            style="Ghost.TButton",
        )
        self.save_and_detect_button.pack(side="right", padx=(0, SPACE["sm"]))
        self._set_reference_view("original")

    def _set_reference_view(self, mode: str) -> None:
        if mode not in {"original", "mask", "overlay"}:
            mode = "original"
        if mode != "original" and self.template_mask is None:
            mode = "original"
        self.reference_view_var.set(mode)
        for key, button in getattr(self, "reference_view_buttons", {}).items():
            button.configure(style="Primary.TButton" if key == mode else "Ghost.TButton")
        self._refresh_reference_view(reset_view=False)

    def _reference_view_image(self) -> np.ndarray | None:
        if self.reference_image is None:
            return None
        mode = self.reference_view_var.get() if hasattr(self, "reference_view_var") else "original"
        if mode == "original" or self.template_mask is None or not self.reference_canvas.roi:
            return self.reference_image
        x, y, width, height = self.reference_canvas.roi
        mask = self.template_mask
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0
        if mode == "mask":
            display = np.zeros_like(self.reference_image)
            crop = display[y : y + height, x : x + width]
            crop[selected] = (255, 255, 255)
            return display
        display = self.reference_image.copy()
        crop = display[y : y + height, x : x + width]
        tint = np.zeros_like(crop)
        tint[:, :, 0] = 255
        blended = cv2.addWeighted(crop, 0.58, tint, 0.42, 0)
        crop[selected] = blended[selected]
        return display

    def _refresh_reference_view(self, *, reset_view: bool = False) -> None:
        if not hasattr(self, "reference_canvas"):
            return
        roi = self.reference_canvas.roi
        self.reference_canvas.set_image(
            self._reference_view_image(),
            clear_roi=False,
            reset_view=reset_view,
        )
        self.reference_canvas.roi = roi
        self.reference_canvas._draw_roi()
        self._update_reference_view_status()

    def _update_reference_view_status(self) -> None:
        if not hasattr(self, "reference_image_info_var"):
            return
        if self.reference_image is None:
            self.reference_image_info_var.set("— × —   RGB")
            self.reference_zoom_var.set("ZOOM  FIT")
            return
        height, width = self.reference_image.shape[:2]
        channels = "GRAY" if self.reference_image.ndim == 2 else "RGB"
        self.reference_image_info_var.set(f"{width} × {height}   {channels}")
        self.reference_zoom_var.set(f"ZOOM  {self.reference_canvas.zoom_factor * 100:.0f}%")

    def _update_template_inspector(self) -> None:
        roi = self.reference_canvas.roi if hasattr(self, "reference_canvas") else None
        if roi:
            self.inspector_roi_var.set(f"{roi[2]} × {roi[3]}   at {roi[0]}, {roi[1]}")
        else:
            self.inspector_roi_var.set("Not selected")
        if self.reference_path is None:
            self.inspector_source_var.set("No image loaded")
        else:
            path_text = str(self.reference_path).lower()
            self.inspector_source_var.set(f"本地图像\n{self.reference_path.name}")
        if self.template_mask is None:
            self.inspector_mask_coverage_var.set("Coverage  —")
            self.inspector_mask_quality_var.set("NOT CREATED")
            self.inspector_quality_label.configure(style="InspectorLabel.TLabel")
            return
        quality = analyze_template_mask(self.template_mask)
        self.inspector_mask_coverage_var.set(f"Coverage  {quality.coverage:.1%}")
        self.inspector_mask_quality_var.set("GOOD" if quality.valid else "CHECK REQUIRED")
        self.inspector_quality_label.configure(style="StatusSuccess.TLabel" if quality.valid else "StatusError.TLabel")

    def _update_template_stepper(self) -> None:
        image_ready = self.reference_image is not None
        roi = self.reference_canvas.roi if hasattr(self, "reference_canvas") else None
        roi_ready = bool(roi and roi[2] >= 5 and roi[3] >= 5)
        mask_ready = self.template_mask is not None
        mask_error = False
        if mask_ready:
            try:
                mask_error = not analyze_template_mask(self.template_mask).valid
            except Exception:
                mask_error = True
        name_ready = bool(self.template_name_var.get().strip())
        if not image_ready:
            states = ("Active", "Idle", "Idle", "Idle", "Idle")
        elif not roi_ready:
            states = ("Done", "Active", "Idle", "Idle", "Idle")
        elif not mask_ready:
            states = ("Done", "Done", "Active", "Idle", "Idle")
        elif mask_error:
            states = ("Done", "Done", "Error", "Idle", "Idle")
        elif not name_ready:
            states = ("Done", "Done", "Done", "Active", "Idle")
        else:
            states = ("Done", "Done", "Done", "Done", "Active")
        for state, (frame, number, label) in zip(states, self.template_step_items):
            frame.configure(style=f"Stepper{state}.TFrame")
            number.configure(style=f"Stepper{state}Index.TLabel")
            label.configure(style=f"Stepper{state}.TLabel")

    def _build_detect_tab(self) -> None:
        top = ttk.Frame(self.detect_tab, padding=(12, 10), style="Card.TFrame")
        top.pack(fill="x", pady=(0, 8))
        top_title = ttk.Frame(top, style="Card.TFrame")
        top_title.pack(side="left", fill="x", expand=True)
        ttk.Label(top_title, text="模板库与离线验证", style="Section.TLabel").pack(anchor="w")
        ttk.Label(top_title, text="选择模板管理参数，选择图像后运行当前启用的全部模板。", style="Hint.TLabel").pack(anchor="w", pady=(2, 0))
        management = ttk.Frame(top, style="Card.TFrame")
        management.pack(side="right")
        ttk.Button(management, text="启用/停用", command=self.toggle_selected_template, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(management, text="编辑参数", command=self.edit_selected_template, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(management, text="重命名", command=self.rename_selected_template, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(management, text="移出", command=self.remove_selected_template, style="Danger.TButton").pack(side="left", padx=3)
        self.loaded_detection_var = tk.StringVar(value="未加载检测图像")
        self.load_detection_button = ttk.Button(management, text="选择检测图像", command=self.load_detection_image, style="Primary.TButton")
        self.load_detection_button.pack(side="left", padx=(10, 3))
        ttk.Label(management, textvariable=self.loaded_detection_var, style="Hint.TLabel").pack(side="left", padx=(5, 0))

        library_frame = ttk.LabelFrame(self.detect_tab, text="持久模板库", padding=7, style="Card.TLabelframe")
        library_frame.pack(fill="x", pady=(0, 6))
        library_columns = ("enabled", "name", "size", "score", "angle", "scale", "method", "status")
        self.library_tree = ttk.Treeview(library_frame, columns=library_columns, show="headings", height=5, selectmode="browse")
        library_labels = {"enabled": "启用", "name": "工件类型", "size": "模板尺寸", "score": "阈值", "angle": "角度范围", "scale": "尺度范围", "method": "方式", "status": "状态"}
        library_widths = {"enabled": 48, "name": 120, "size": 80, "score": 55, "angle": 130, "scale": 125, "method": 70, "status": 180}
        for column in library_columns:
            self.library_tree.heading(column, text=library_labels[column])
            self.library_tree.column(column, width=library_widths[column], anchor="center")
        library_scroll = ttk.Scrollbar(library_frame, orient="vertical", command=self.library_tree.yview)
        self.library_tree.configure(yscrollcommand=library_scroll.set)
        self.library_tree.tag_configure("normal", background=COLORS["panel"], foreground=COLORS["text"])
        self.library_tree.tag_configure("disabled", background=COLORS["sidebar"], foreground=COLORS["text_disabled"])
        self.library_tree.tag_configure("error", background="#2A171C", foreground=COLORS["error"])
        self.library_tree.pack(side="left", fill="x", expand=True)
        library_scroll.pack(side="right", fill="y")
        self.library_tree.bind("<Double-1>", lambda _event: self.toggle_selected_template())
        self.library_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_ui_state())

        params = ttk.LabelFrame(
            self.detect_tab,
            text="检测操作｜左侧参数仅用于新建或导入模板，已有模板请使用“编辑参数”",
            padding=(8, 5),
            style="Card.TLabelframe",
        )
        params.pack(fill="x", pady=(0, 6))
        defaults = {
            "阈值": ("score", "0.72"), "最小角度": ("angle_min", "-180"),
            "最大角度": ("angle_max", "178"), "角度步长": ("angle_step", "2"),
            "最小尺度": ("scale_min", "0.5"), "最大尺度": ("scale_max", "1.8"),
            "尺度步长": ("scale_step", "0.05"), "NMS IoU": ("nms", "0.25"),
        }
        self.param_vars: dict[str, tk.StringVar] = {}
        for column, (label, (key, value)) in enumerate(defaults.items()):
            ttk.Label(params, text=label).grid(row=0, column=column, padx=3)
            variable = tk.StringVar(value=value)
            self.param_vars[key] = variable
            ttk.Entry(params, textvariable=variable, width=9).grid(row=1, column=column, padx=3)
        ttk.Label(params, text="特征模式").grid(row=0, column=8, padx=4)
        self.feature_mode_var = tk.StringVar(value=FEATURE_MODE_LABELS["auto"])
        ttk.Combobox(params, textvariable=self.feature_mode_var, values=list(FEATURE_MODE_LABELS.values()), state="readonly", width=10).grid(row=1, column=8, padx=4)
        self.use_edges_var = tk.BooleanVar(value=False)
        self.detect_button = ttk.Button(params, text="▶ 开始检测", command=self.start_detection, style="Primary.TButton")
        self.detect_button.grid(row=0, column=9, rowspan=2, padx=4)
        self.save_button = ttk.Button(params, text="保存结果", command=self.save_results, state="disabled", style="Secondary.TButton")
        self.save_button.grid(row=0, column=10, rowspan=2, padx=4)
        self.progress = ttk.Progressbar(params, mode="indeterminate", length=100)
        self.progress.grid(row=0, column=11, rowspan=2, padx=7)

        pane = ttk.Panedwindow(self.detect_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        image_frame = ttk.Frame(pane, padding=8, style="Card.TFrame")
        result_frame = ttk.Frame(pane, padding=8, style="Card.TFrame")
        pane.add(image_frame, weight=4)
        pane.add(result_frame, weight=2)
        self.detection_canvas = ImageCanvas(image_frame)
        image_toolbar = ttk.Frame(image_frame, style="Card.TFrame")
        image_toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(image_toolbar, text="检测图像｜滚轮缩放，中键拖动", style="Hint.TLabel").pack(side="left")
        ttk.Button(image_toolbar, text="放大 ＋", command=self.detection_canvas.zoom_in, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(image_toolbar, text="缩小 －", command=self.detection_canvas.zoom_out, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(image_toolbar, text="适应窗口", command=self.detection_canvas.fit_to_window, style="Secondary.TButton").pack(side="right")
        self.detection_canvas.pack(fill="both", expand=True)

        summary = ttk.Frame(result_frame, style="Card.TFrame")
        summary.pack(fill="x", padx=(8, 0), pady=(0, 8))
        self.count_var = tk.StringVar(value="目标数量：0")
        self.total_metric_var = tk.StringVar(value="0")
        self.ambiguous_metric_var = tk.StringVar(value="0")
        self.type_metric_var = tk.StringVar(value="0")
        for label, variable in (
            ("当前目标", self.total_metric_var),
            ("待确认", self.ambiguous_metric_var),
            ("明确类型", self.type_metric_var),
        ):
            card = ttk.Frame(summary, padding=(12, 7), style="Metric.TFrame")
            card.pack(side="left", fill="x", expand=True, padx=(0, 6))
            ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w")
            ttk.Label(card, text=label, style="MetricLabel.TLabel").pack(anchor="w")
        columns = ("id", "status", "template", "x", "y", "angle", "score", "scale", "candidates")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=16)
        labels = {"id": "编号", "status": "识别状态", "template": "目标类型", "x": "中心 X", "y": "中心 Y", "angle": "角度°", "score": "得分", "scale": "尺度", "candidates": "候选类型"}
        widths = {"id": 42, "status": 72, "template": 82, "x": 70, "y": 70, "angle": 65, "score": 65, "scale": 52, "candidates": 140}
        for column in columns:
            self.result_tree.heading(column, text=labels[column])
            self.result_tree.column(column, width=widths[column], anchor="center")
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.tag_configure("confirmed", background="#10261B", foreground=COLORS["success"])
        self.result_tree.tag_configure("ambiguous", background="#2B2110", foreground=COLORS["warning"])
        self.result_tree.pack(side="left", fill="both", expand=True, padx=(8, 0))
        scrollbar.pack(side="right", fill="y")

    def _build_live_tab(self) -> None:
        controls = ttk.LabelFrame(self.live_tab, text="相机 / 视频源与计数设置", padding=8, style="Card.TLabelframe")
        controls.pack(fill="x", pady=(0, 7))
        self.live_width_var = tk.StringVar(value="1920")
        self.live_height_var = tk.StringVar(value="1080")
        self.live_fps_var = tk.StringVar(value="15")
        self.live_exposure_var = tk.StringVar(value="")
        # DirectShow is generally more stable than MSMF for USB cameras on Windows.
        self.live_backend_var = tk.StringVar(value="dshow")
        self.live_fourcc_var = tk.StringVar(value="MJPG")
        self.live_max_edge_var = tk.StringVar(value="1920")
        self.live_detector_mode_var = tk.StringVar(value="荧光工件分割")
        ttk.Label(controls, text="视频源（相机编号或文件）").grid(row=0, column=0, padx=4, sticky="w")
        self.live_source_var = tk.StringVar(value="0")
        ttk.Entry(controls, textvariable=self.live_source_var, width=28).grid(row=1, column=0, padx=4, sticky="ew")
        ttk.Button(controls, text="选择视频", command=self._browse_live_source, style="Secondary.TButton").grid(row=1, column=1, padx=4)
        ttk.Label(controls, text="计数线方向").grid(row=0, column=2, padx=4)
        self.live_axis_var = tk.StringVar(value="竖线（从左到右）")
        ttk.Combobox(controls, textvariable=self.live_axis_var, values=("竖线（从左到右）", "横线（从上到下）"), state="readonly", width=16).grid(row=1, column=2, padx=4)
        ttk.Label(controls, text="位置比例").grid(row=0, column=3, padx=4)
        self.live_line_position_var = tk.StringVar(value="0.50")
        ttk.Entry(controls, textvariable=self.live_line_position_var, width=8).grid(row=1, column=3, padx=4)
        ttk.Label(controls, text="计数方向").grid(row=0, column=4, padx=4)
        self.live_direction_var = tk.StringVar(value="正向")
        ttk.Combobox(controls, textvariable=self.live_direction_var, values=("正向", "反向", "双向"), state="readonly", width=8).grid(row=1, column=4, padx=4)
        self.live_quality_gate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="质量不合格时跳过检测", variable=self.live_quality_gate_var).grid(row=1, column=5, padx=7)
        self.live_start_button = ttk.Button(controls, text="▶ 启动实时检测", command=self.start_live_detection, style="Primary.TButton")
        self.live_start_button.grid(row=0, column=6, rowspan=2, padx=5)
        self.live_stop_button = ttk.Button(controls, text="■ 停止", command=self.stop_live_detection, state="disabled", style="Danger.TButton")
        self.live_stop_button.grid(row=0, column=7, rowspan=2, padx=5)
        ttk.Button(controls, text="清零累计", command=self.reset_live_counts, style="Secondary.TButton").grid(row=0, column=8, rowspan=2, padx=5)
        camera_fields = (
            ("分辨率宽", self.live_width_var, 2),
            ("分辨率高", self.live_height_var, 3),
            ("采集 FPS", self.live_fps_var, 4),
            ("曝光（留空自动）", self.live_exposure_var, 5),
        )
        for label, variable, column in camera_fields:
            ttk.Label(controls, text=label).grid(row=2, column=column, padx=4, pady=(8, 0))
            ttk.Entry(controls, textvariable=variable, width=10).grid(row=3, column=column, padx=4)
        ttk.Label(controls, text="相机后端").grid(row=2, column=0, padx=4, pady=(8, 0), sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.live_backend_var,
            values=("auto", "dshow", "msmf"),
            state="readonly",
            width=10,
        ).grid(row=3, column=0, padx=4, sticky="w")
        ttk.Label(controls, text="视频格式").grid(row=2, column=1, padx=4, pady=(8, 0))
        ttk.Entry(controls, textvariable=self.live_fourcc_var, width=8).grid(row=3, column=1, padx=4)
        ttk.Label(controls, text="检测最长边").grid(row=2, column=6, padx=4, pady=(8, 0))
        ttk.Entry(controls, textvariable=self.live_max_edge_var, width=10).grid(row=3, column=6, padx=4)
        ttk.Label(controls, text="检测算法").grid(row=2, column=7, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.live_detector_mode_var,
            values=("荧光工件分割", "多模板匹配"),
            state="readonly",
            width=13,
        ).grid(row=3, column=7, padx=4)
        controls.columnconfigure(0, weight=1)

        pane = ttk.Panedwindow(self.live_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        image_frame = ttk.Frame(pane, padding=8, style="Card.TFrame")
        dashboard = ttk.Frame(pane, padding=12, style="Card.TFrame")
        pane.add(image_frame, weight=4)
        pane.add(dashboard, weight=2)
        toolbar = ttk.Frame(image_frame, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="实时画面｜拖框设置检测 ROI", style="Hint.TLabel").pack(side="left")
        self.live_canvas = ImageCanvas(image_frame, selectable=True)
        self.live_canvas.bind("<<RoiChanged>>", self._live_roi_changed)
        self.live_roi_var = tk.StringVar(value="检测 ROI：整幅图像")
        ttk.Label(toolbar, textvariable=self.live_roi_var, style="Hint.TLabel").pack(side="left", padx=(16, 0))
        ttk.Button(toolbar, text="清除 ROI", command=self._clear_live_roi, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="保存当前诊断", command=self.save_live_diagnostic, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="放大 ＋", command=self.live_canvas.zoom_in, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="缩小 －", command=self.live_canvas.zoom_out, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="适应窗口", command=self.live_canvas.fit_to_window, style="Secondary.TButton").pack(side="right")
        self.live_canvas.pack(fill="both", expand=True)

        self.live_count_var = tk.StringVar(value="当前画面工件：0")
        self.live_breakdown_var = tk.StringVar(value="当前分类：-")
        self.live_cumulative_var = tk.StringVar(value="过线累计（辅助）：0")
        self.live_quality_var = tk.StringVar(value="画面质量：未启动")
        self.live_performance_var = tk.StringVar(value="处理耗时：-")
        ttk.Label(dashboard, textvariable=self.live_count_var, style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(dashboard, textvariable=self.live_breakdown_var, style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(dashboard, textvariable=self.live_cumulative_var, style="Hint.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(dashboard, textvariable=self.live_quality_var, style="Hint.TLabel", wraplength=360).pack(anchor="w", pady=(0, 4))
        ttk.Label(dashboard, textvariable=self.live_performance_var, style="Hint.TLabel").pack(anchor="w", pady=(0, 12))
        columns = ("track", "template", "x", "y", "angle", "pick", "safe", "counted")
        self.live_tree = ttk.Treeview(dashboard, columns=columns, show="headings", height=15)
        labels = {
            "track": "轨迹", "template": "类型", "x": "中心X", "y": "中心Y",
            "angle": "角度°", "pick": "吸取点", "safe": "安全半径", "counted": "状态",
        }
        widths = {
            "track": 50, "template": 105, "x": 62, "y": 62, "angle": 62,
            "pick": 105, "safe": 70, "counted": 60,
        }
        for column in columns:
            self.live_tree.heading(column, text=labels[column])
            self.live_tree.column(column, width=widths[column], anchor="center")
        self.live_tree.pack(fill="both", expand=True)

    def _build_validation_tab(self) -> None:
        controls = ttk.Frame(self.validation_tab, padding=(14, 10), style="Card.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="模板健康检查与受控压力测试", style="Section.TLabel").pack(side="left")
        ttk.Button(
            controls, text="刷新健康检查", command=self.refresh_template_audits,
            style="Secondary.TButton",
        ).pack(side="left", padx=(14, 4))
        self.apply_recommendation_button = ttk.Button(
            controls, text="应用模式建议", command=self.apply_audit_recommendation,
            style="Secondary.TButton",
        )
        self.apply_recommendation_button.pack(side="left", padx=4)
        self.validation_scene_var = tk.StringVar(value="3")
        self.validation_profile_var = tk.StringVar(value="balanced")
        ttk.Label(controls, text="场景数").pack(side="left", padx=(18, 3))
        ttk.Entry(controls, textvariable=self.validation_scene_var, width=5).pack(side="left")
        ttk.Label(controls, text="压力等级").pack(side="left", padx=(10, 3))
        ttk.Combobox(
            controls,
            textvariable=self.validation_profile_var,
            values=("clean", "balanced", "harsh"),
            state="readonly",
            width=10,
        ).pack(side="left")
        self.parameter_search_button = ttk.Button(
            controls, text="自动搜索所选模板", command=self.start_parameter_search,
            style="Primary.TButton",
        )
        self.parameter_search_button.pack(side="right", padx=(4, 0))
        self.validation_benchmark_button = ttk.Button(
            controls, text="运行模板库压力测试", command=self.start_validation_benchmark,
            style="Primary.TButton",
        )
        self.validation_benchmark_button.pack(side="right", padx=4)

        pane = ttk.Panedwindow(self.validation_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        audit_card = ttk.Frame(pane, padding=10, style="Card.TFrame")
        detail_card = ttk.Frame(pane, padding=12, style="Card.TFrame")
        pane.add(audit_card, weight=3)
        pane.add(detail_card, weight=2)
        ttk.Label(audit_card, text="模板库诊断", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        columns = ("enabled", "name", "score", "current", "recommended", "mask", "status")
        self.audit_tree = ttk.Treeview(audit_card, columns=columns, show="headings", height=15, selectmode="browse")
        labels = {
            "enabled": "启用", "name": "模板名称", "score": "健康分",
            "current": "当前模式", "recommended": "推荐模式", "mask": "Mask覆盖率", "status": "状态",
        }
        widths = {"enabled": 48, "name": 125, "score": 65, "current": 95, "recommended": 95, "mask": 85, "status": 72}
        for column in columns:
            self.audit_tree.heading(column, text=labels[column])
            self.audit_tree.column(column, width=widths[column], anchor="center")
        audit_scroll = ttk.Scrollbar(audit_card, orient="vertical", command=self.audit_tree.yview)
        self.audit_tree.configure(yscrollcommand=audit_scroll.set)
        self.audit_tree.tag_configure("good", background="#10261B", foreground=COLORS["success"])
        self.audit_tree.tag_configure("warning", background="#2B2110", foreground=COLORS["warning"])
        self.audit_tree.tag_configure("error", background="#2A171C", foreground=COLORS["error"])
        self.audit_tree.pack(side="left", fill="both", expand=True)
        audit_scroll.pack(side="right", fill="y")
        self.audit_tree.bind("<<TreeviewSelect>>", self._show_selected_audit)

        ttk.Label(detail_card, text="诊断与搜索结果", style="Section.TLabel").pack(anchor="w")
        self.audit_detail_text = tk.Text(
            detail_card,
            height=20,
            wrap="word",
            relief="flat",
            background=COLORS["elevated"],
            foreground=COLORS["text"],
            insertbackground=COLORS["text"],
            font=("Microsoft YaHei UI", 10),
            padx=10,
            pady=10,
        )
        self.audit_detail_text.pack(fill="both", expand=True, pady=(8, 8))
        self.audit_detail_text.insert("1.0", "选择左侧模板查看诊断信息。")
        self.audit_detail_text.configure(state="disabled")
        action_row = ttk.Frame(detail_card, style="Card.TFrame")
        action_row.pack(fill="x")
        self.apply_search_button = ttk.Button(
            action_row,
            text="应用搜索结果",
            command=self.apply_parameter_search,
            state="disabled",
            style="Primary.TButton",
        )
        self.apply_search_button.pack(side="right")
        self.validation_progress = ttk.Progressbar(action_row, mode="indeterminate", length=180)
        self.validation_progress.pack(side="left", padx=(0, 10))
        self.validation_summary_var = tk.StringVar(value="尚未运行压力测试")
        ttk.Label(action_row, textvariable=self.validation_summary_var, style="Hint.TLabel").pack(side="left")

    def _roi_changed(self, _event=None) -> None:
        roi = self.reference_canvas.roi
        self.template_mask = None
        self.template_quality_var.set("Mask质量：ROI已改变，请重新提取轮廓")
        self.roi_var.set(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}" if roi else "未选择 ROI")
        if hasattr(self, "reference_view_var"):
            self.reference_view_var.set("original")
            self._set_reference_view("original")
        self._update_ui_state()

    def load_reference(self, auto_extract: bool = False) -> None:
        path = filedialog.askopenfilename(title="选择参考图像", filetypes=IMAGE_TYPES)
        if not path:
            return
        self.load_reference_path(path, auto_extract=auto_extract)

    def load_reference_path(self, path: str | Path, auto_extract: bool = False) -> None:
        """Load a reference image selected from the local filesystem."""

        try:
            self.reference_image = read_image(path)
            self.reference_path = Path(path)
            self.template_mask = None
            self.template_quality_var.set("Mask质量：尚未建立")
            self.reference_view_var.set("original")
            self._refresh_reference_view(reset_view=True)
            self.footer_image_var.set(f"IMAGE  {self.reference_image.shape[1]}×{self.reference_image.shape[0]}")
            self.status_var.set(f"已加载参考图像：{path}")
            self._update_ui_state()
            if auto_extract:
                self.after(80, self.auto_locate_template)
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc), parent=self)

    def open_calibration_workbench(self) -> None:
        CalibrationWorkbench(self)

    def auto_locate_template(self) -> None:
        if self.reference_image is None:
            messagebox.showwarning("缺少图像", "请先导入基准图像。", parent=self)
            return
        self.configure(cursor="watch")
        self.status_var.set("正在自动定位工件并绘制模板……")
        self.update_idletasks()
        try:
            selection = locate_assisted_template(self.reference_image, "auto")
            self.reference_canvas.roi = selection.roi_xywh
            self.reference_canvas._draw_roi()
            x, y, width, height = selection.roi_xywh
            crop = self.reference_image[y : y + height, x : x + width].copy()
            self.roi_var.set(
                f"自动定位 ROI {width} x {height}，整图覆盖 {selection.full_image_coverage:.1%}；请检查绿色轮廓"
            )
            dialog = MaskEditorDialog(self, crop, selection.mask)
            self.wait_window(dialog)
            if dialog.result is not None:
                self.template_mask = dialog.result
                quality = analyze_template_mask(self.template_mask)
                self.roi_var.set(f"自动模板已确认，有效区域 {quality.coverage:.1%}（ROI {width} x {height}）")
                self.template_quality_var.set(f"Mask质量：{quality.message}")
                self._set_reference_view("overlay")
                self.status_var.set("自动模板绘制完成，可设置名称并保存到模板库")
            else:
                self.status_var.set("已取消自动模板编辑，仍可手动框选")
        except Exception as exc:
            self.status_var.set("自动模板绘制失败")
            messagebox.showerror("自动绘制失败", str(exc), parent=self)
        finally:
            self.configure(cursor="")
            self._update_ui_state()

    def edit_template_shape(self) -> None:
        if self.reference_image is None or not self.reference_canvas.roi:
            messagebox.showwarning("缺少选区", "请先加载参考图像并框选目标的大致范围。", parent=self)
            return
        x, y, width, height = self.reference_canvas.roi
        if width < 5 or height < 5:
            messagebox.showwarning("选区过小", "粗 ROI 至少需要 5 x 5 像素。", parent=self)
            return
        crop = self.reference_image[y : y + height, x : x + width].copy()
        dialog = MaskEditorDialog(self, crop, self.template_mask)
        self.wait_window(dialog)
        if dialog.result is not None:
            self.template_mask = dialog.result
            quality = analyze_template_mask(self.template_mask)
            self.roi_var.set(f"不规则模板已标注，有效区域 {quality.coverage:.1%}（ROI {width} x {height}）")
            self.template_quality_var.set(f"Mask质量：{quality.message}")
            self._set_reference_view("overlay")

        self._update_ui_state()

    def save_template(self, continue_to_detection: bool = False) -> None:
        if self.reference_image is None or not self.reference_canvas.roi:
            messagebox.showwarning("缺少选区", "请先加载参考图像并框选目标。", parent=self)
            return
        name = self.template_name_var.get().strip()
        if not name:
            messagebox.showwarning("缺少模板名称", "请填写能表示工件类别的模板名称，例如 Pink_Textile_01。", parent=self)
            return
        if len(name) > 48:
            messagebox.showwarning("名称过长", "模板名称最多48个字符。", parent=self)
            return
        if name.casefold() in {"t1", "target", "template", "object"}:
            if not messagebox.askyesno(
                "名称过于笼统",
                f"模板名称“{name}”不利于分类和现场调试，建议改成明确工件类型。\n\n仍要继续保存吗？",
                parent=self,
            ):
                return
        if self.template_mask is None:
            if not messagebox.askyesno(
                "尚未提取轮廓",
                "当前将使用整个矩形ROI作为模板，可能包含较多背景。\n建议先点击“编辑 / 算法辅助提取”。\n\n仍要继续保存吗？",
                parent=self,
            ):
                return
        else:
            quality = analyze_template_mask(self.template_mask)
            if not quality.valid and not messagebox.askyesno(
                "Mask质量警告",
                quality.message + "。这可能降低定位和分类稳定性。\n\n仍要继续保存吗？",
                parent=self,
            ):
                return
        try:
            model = TemplateModel.from_roi(
                self.reference_image,
                self.reference_canvas.roi,
                name,
                str(self.reference_path) if self.reference_path else None,
                self.template_mask,
                tighten_mask=self.template_mask is not None,
            )
        except Exception as exc:
            messagebox.showerror("模板无效", str(exc), parent=self)
            return
        try:
            parameters = self._parameters()
            parameters.validate()
            entry = self.template_library.add_model(model, parameters)
            self._invalidate_matcher()
            self._refresh_library_tree(select_id=entry.template_id)
            saved = self.template_library.root / entry.template_file
            model_height, model_width = model.image.shape[:2]
            self.status_var.set(f"模板已加入模板库：{model.name}（紧裁剪 {model_width}x{model_height}）")
            self._update_ui_state()
            if continue_to_detection:
                self.notebook.select(self.detect_tab)
                self.workflow_title_var.set("第 1 步 / 2：选择检测图像")
                self.workflow_hint_var.set(f"模板“{model.name}”已保存并启用，请选择一张现场图像进行验证。")
            else:
                messagebox.showinfo(
                    "模板已建立",
                    f"工件类型：{model.name}\n有效模板尺寸：{model_width} x {model_height}\n\n已保存并加入模板库：\n{saved}",
                    parent=self,
                )
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)

    def import_template(self) -> None:
        path = filedialog.askopenfilename(title="导入模板到模板库", filetypes=[("模板元数据", "*.json")])
        if not path:
            return
        try:
            parameters = self._parameters()
            parameters.validate()
            entry = self.template_library.import_template(path, parameters)
            self._invalidate_matcher()
            self._refresh_library_tree(select_id=entry.template_id)
            self.status_var.set(f"已导入模板：{entry.name}")
        except Exception as exc:
            messagebox.showerror("模板导入失败", str(exc), parent=self)

    def _selected_template_id(self) -> str | None:
        selection = self.library_tree.selection()
        return selection[0] if selection else None

    def _invalidate_matcher(self) -> None:
        self.multi_matcher = None

    def _refresh_library_tree(self, select_id: str | None = None) -> None:
        if not hasattr(self, "library_tree"):
            return
        self.library_tree.delete(*self.library_tree.get_children())
        for loaded in self.template_library.load_entries():
            entry = loaded.entry
            size = "-"
            if loaded.valid:
                size = f"{loaded.model.image.shape[1]}x{loaded.model.image.shape[0]}"
            params = entry.parameters
            values = (
                "✓" if entry.enabled else "—",
                entry.name,
                size,
                f"{params.score_threshold:.2f}",
                f"{params.angle_min:g}~{params.angle_max:g}/{params.angle_step:g}",
                f"{params.scale_min:g}~{params.scale_max:g}/{params.scale_step:g}",
                FEATURE_MODE_LABELS.get(params.feature_mode, "边缘" if params.use_edges else "灰度"),
                "正常" if loaded.valid else f"异常：{loaded.error}",
            )
            row_tag = "error" if not loaded.valid else ("normal" if entry.enabled else "disabled")
            self.library_tree.insert("", "end", iid=entry.template_id, values=values, tags=(row_tag,))
        if select_id and self.library_tree.exists(select_id):
            self.library_tree.selection_set(select_id)
            self.library_tree.see(select_id)
        self._update_ui_state()

    def _set_audit_detail(self, text: str) -> None:
        self.audit_detail_text.configure(state="normal")
        self.audit_detail_text.delete("1.0", "end")
        self.audit_detail_text.insert("1.0", text)
        self.audit_detail_text.configure(state="disabled")

    def _selected_audit_template_id(self) -> str | None:
        selection = self.audit_tree.selection()
        return selection[0] if selection else None

    def refresh_template_audits(self, show_status: bool = True) -> None:
        try:
            previous = self._selected_audit_template_id() if hasattr(self, "audit_tree") else None
            self.template_audits = audit_template_library(self.template_library)
            if not hasattr(self, "audit_tree"):
                return
            self.audit_tree.delete(*self.audit_tree.get_children())
            for audit in self.template_audits:
                mask = f"{audit.mask_coverage:.1%}" if audit.mask_coverage is not None else "-"
                status = {"good": "正常", "warning": "需检查", "error": "异常"}.get(audit.status, audit.status)
                self.audit_tree.insert(
                    "", "end", iid=audit.template_id,
                    values=(
                        "✓" if audit.enabled else "—",
                        audit.template_name,
                        audit.health_score,
                        FEATURE_MODE_LABELS.get(audit.current_mode, audit.current_mode),
                        FEATURE_MODE_LABELS.get(audit.recommended_mode or "", audit.recommended_mode or "-"),
                        mask,
                        status,
                    ),
                    tags=(audit.status,),
                )
            report = self.data_root / "outputs" / "algorithm_validation" / "template_audit_latest.json"
            write_template_audit_report(report, self.template_audits)
            if previous and self.audit_tree.exists(previous):
                self.audit_tree.selection_set(previous)
            elif self.template_audits:
                self.audit_tree.selection_set(self.template_audits[0].template_id)
            self._show_selected_audit()
            if show_status and hasattr(self, "status_var"):
                warning_count = sum(item.status != "good" for item in self.template_audits)
                self.status_var.set(f"模板健康检查完成：{len(self.template_audits)} 项，{warning_count} 项需要检查")
        except Exception as exc:
            if show_status:
                messagebox.showerror("模板检查失败", str(exc), parent=self)

    def _show_selected_audit(self, _event=None) -> None:
        template_id = self._selected_audit_template_id()
        audit = next((item for item in self.template_audits if item.template_id == template_id), None)
        if audit is None:
            self._set_audit_detail("选择左侧模板查看诊断信息。")
            self._update_ui_state()
            return
        lines = [
            f"模板：{audit.template_name}",
            f"健康分：{audit.health_score}/100",
            f"当前模式：{FEATURE_MODE_LABELS.get(audit.current_mode, audit.current_mode)}",
            f"推荐模式：{FEATURE_MODE_LABELS.get(audit.recommended_mode or '', audit.recommended_mode or '-')}",
        ]
        if audit.recommendation_confidence is not None:
            lines.append(f"推荐置信度：{audit.recommendation_confidence:.1%}")
        if audit.recommendation_reason:
            lines.append(f"依据：{audit.recommendation_reason}")
        if audit.mask_coverage is not None:
            lines.extend((
                "",
                f"Mask覆盖率：{audit.mask_coverage:.1%}",
                f"明显连通区域：{audit.mask_component_count}",
                f"主体占比：{audit.mask_main_component_ratio:.1%}",
                f"接触边界：{'是' if audit.mask_touches_border else '否'}",
            ))
        lines.append("\n发现的问题：")
        lines.extend(f"• {item}" for item in audit.issues or ("未发现明显配置问题",))
        lines.append("\n建议操作：")
        lines.extend(f"• {item}" for item in audit.actions)
        self._set_audit_detail("\n".join(lines))
        if self.parameter_search_result is not None and self.parameter_search_result.template_id != template_id:
            self.parameter_search_result = None
        self._update_ui_state()

    def apply_audit_recommendation(self) -> None:
        template_id = self._selected_audit_template_id()
        audit = next((item for item in self.template_audits if item.template_id == template_id), None)
        if audit is None or not audit.recommended_mode:
            messagebox.showinfo("请选择模板", "请先选择一个有效模板。", parent=self)
            return
        entry = self.template_library.get(template_id)
        if entry.parameters.feature_mode == audit.recommended_mode:
            self.status_var.set(f"{entry.name} 已经使用推荐特征模式")
            return
        if not messagebox.askyesno(
            "应用模式建议",
            f"将“{entry.name}”的特征模式从\n{FEATURE_MODE_LABELS.get(entry.parameters.feature_mode, entry.parameters.feature_mode)}\n改为\n{FEATURE_MODE_LABELS.get(audit.recommended_mode, audit.recommended_mode)}？",
            parent=self,
        ):
            return
        self.template_library.update_parameters(
            template_id,
            replace(entry.parameters, feature_mode=audit.recommended_mode),
        )
        self._invalidate_matcher()
        self._refresh_library_tree(select_id=template_id)
        self.refresh_template_audits(show_status=False)
        self.audit_tree.selection_set(template_id)
        self._show_selected_audit()
        self.status_var.set(f"已为 {entry.name} 应用推荐特征模式")

    def _validation_scene_count(self) -> int:
        count = int(self.validation_scene_var.get())
        if not 1 <= count <= 20:
            raise ValueError("场景数必须在1到20之间")
        return count

    def _start_validation_worker(self, task_name: str, function) -> None:
        if self.validation_running:
            messagebox.showinfo("任务进行中", "请等待当前算法验证任务结束。", parent=self)
            return
        self.validation_running = True
        self._update_ui_state()
        self.validation_progress.start(12)
        self.validation_summary_var.set(f"正在执行：{task_name}……")

        def worker() -> None:
            try:
                self.validation_queue.put(("success", task_name, function()))
            except Exception as exc:
                self.validation_queue.put(("error", task_name, exc))

        threading.Thread(target=worker, daemon=True, name=f"validation-{task_name}").start()
        self.after(100, self._poll_validation_queue)

    def start_parameter_search(self) -> None:
        template_id = self._selected_audit_template_id()
        if not template_id:
            messagebox.showinfo("请选择模板", "请先在诊断表中选择一个模板。", parent=self)
            return
        try:
            config = ParameterSearchConfig(
                scene_count=self._validation_scene_count(),
                profile=self.validation_profile_var.get(),
                max_trials=6,
            )
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.data_root / "outputs" / "algorithm_validation" / f"search_{timestamp}_{template_id[:8]}.json"

        def work():
            result = search_template_parameters(self.template_library, template_id, config)
            write_parameter_search_report(output, result)
            return result, output

        self._start_validation_worker("参数搜索", work)

    def start_validation_benchmark(self) -> None:
        try:
            count = self._validation_scene_count()
            profile = self.validation_profile_var.get()
            SyntheticStressConfig(profile=profile).validate()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.data_root / "outputs" / "algorithm_validation" / f"benchmark_{timestamp}"

        def work():
            return run_synthetic_benchmark(
                self.template_library.root,
                output,
                scene_count=count,
                config=SyntheticStressConfig(profile=profile, min_objects=1, max_objects=3),
                thresholds=EvaluationThresholds(),
                save_images=True,
            )

        self._start_validation_worker("模板库压力测试", work)

    def _poll_validation_queue(self) -> None:
        try:
            event = self.validation_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_validation_queue)
            return
        self.validation_running = False
        self.validation_progress.stop()
        self._update_ui_state()
        if event[0] == "error":
            self.validation_summary_var.set(f"{event[1]}失败")
            messagebox.showerror(f"{event[1]}失败", str(event[2]), parent=self)
            return
        task_name, payload = event[1], event[2]
        if task_name == "参数搜索":
            self.parameter_search_result, output = payload
            best = self.parameter_search_result.best_trial
            self.validation_summary_var.set(f"搜索完成：F1 {best.f1:.3f}，报告已保存")
            self._set_audit_detail(
                f"参数搜索完成｜{self.parameter_search_result.template_name}\n\n"
                f"推荐模式：{FEATURE_MODE_LABELS.get(best.feature_mode, best.feature_mode)}\n"
                f"推荐阈值：{best.score_threshold:.2f}\n"
                f"角度步长：{best.angle_step:g}°\n"
                f"尺度范围：{best.scale_min:g} ~ {best.scale_max:g}\n\n"
                f"Precision：{best.precision:.3f}\nRecall：{best.recall:.3f}\nF1：{best.f1:.3f}\n"
                f"平均中心误差：{best.mean_center_error_px if best.mean_center_error_px is not None else '-'} px\n"
                f"平均角度误差：{best.mean_angle_error_deg if best.mean_angle_error_deg is not None else '-'}°\n"
                f"平均耗时：{best.mean_elapsed_ms:.1f} ms\n\n"
                f"报告：{output}\n\n搜索结果基于合成扰动，应用后仍需现场图片验证。"
            )
            self.apply_search_button.configure(state="normal")
            self._update_ui_state()
            return
        output, summary = payload
        self.validation_summary_var.set(f"压力测试完成：F1 {summary.f1:.3f}，准确计数 {summary.exact_count_scenes}/{summary.scene_count}")
        self._set_audit_detail(
            "模板库压力测试完成\n\n"
            f"场景数：{summary.scene_count}\n真值目标：{summary.ground_truth_count}\n预测目标：{summary.prediction_count}\n"
            f"Precision：{summary.precision:.3f}\nRecall：{summary.recall:.3f}\nF1：{summary.f1:.3f}\n"
            f"分类准确率：{summary.classification_accuracy:.3f}\n"
            f"中心平均误差：{summary.mean_center_error_px if summary.mean_center_error_px is not None else '-'} px\n"
            f"角度平均误差：{summary.mean_angle_error_deg if summary.mean_angle_error_deg is not None else '-'}°\n"
            f"平均耗时：{summary.mean_elapsed_ms:.1f} ms\n\n输出目录：{output}"
        )

    def apply_parameter_search(self) -> None:
        result = self.parameter_search_result
        if result is None:
            return
        if not messagebox.askyesno(
            "应用搜索结果",
            f"将自动搜索结果写入模板“{result.template_name}”？\n原参数会被替换，但模板图像和Mask不会改变。",
            parent=self,
        ):
            return
        self.template_library.update_parameters(result.template_id, result.best_parameters)
        self._invalidate_matcher()
        self._refresh_library_tree(select_id=result.template_id)
        self.refresh_template_audits(show_status=False)
        if self.audit_tree.exists(result.template_id):
            self.audit_tree.selection_set(result.template_id)
        self.apply_search_button.configure(state="disabled")
        self.parameter_search_result = None
        self.status_var.set(f"已应用 {result.template_name} 的参数搜索结果")
        self._update_ui_state()

    def reload_library(self) -> None:
        try:
            self.template_library.load()
            self._invalidate_matcher()
            self._refresh_library_tree()
            self.status_var.set(f"模板库已重新加载：{len(self.template_library.entries)} 个模板")
        except Exception as exc:
            messagebox.showerror("模板库加载失败", str(exc), parent=self)

    def toggle_selected_template(self) -> None:
        template_id = self._selected_template_id()
        if not template_id:
            messagebox.showinfo("请选择模板", "请先在模板库中选择一项。", parent=self)
            return
        entry = self.template_library.get(template_id)
        self.template_library.set_enabled(template_id, not entry.enabled)
        self._invalidate_matcher()
        self._refresh_library_tree(select_id=template_id)

    def edit_selected_template(self) -> None:
        template_id = self._selected_template_id()
        if not template_id:
            messagebox.showinfo("请选择模板", "请先在模板库中选择一项。", parent=self)
            return
        entry = self.template_library.get(template_id)
        dialog = TemplateParameterDialog(self, entry.name, entry.parameters)
        self.wait_window(dialog)
        if dialog.result is not None:
            self.template_library.update_parameters(template_id, dialog.result)
            self._invalidate_matcher()
            self._refresh_library_tree(select_id=template_id)
            self.status_var.set(f"已更新模板参数：{entry.name}")

    def rename_selected_template(self) -> None:
        template_id = self._selected_template_id()
        if not template_id:
            messagebox.showwarning("未选择模板", "请先在模板库中选择一个工件类型。", parent=self)
            return
        entry = self.template_library.get(template_id)
        name = simpledialog.askstring(
            "重命名工件类型",
            "请输入用于检测界面和分类输出的模板名称：",
            initialvalue=entry.name,
            parent=self,
        )
        if name is None:
            return
        try:
            self.template_library.rename(template_id, name)
            self._invalidate_matcher()
            self._refresh_library_tree(select_id=template_id)
            self.status_var.set(f"工件类型已重命名：{name.strip()}")
        except Exception as exc:
            messagebox.showerror("重命名失败", str(exc), parent=self)

    def remove_selected_template(self) -> None:
        template_id = self._selected_template_id()
        if not template_id:
            messagebox.showinfo("请选择模板", "请先在模板库中选择一项。", parent=self)
            return
        entry = self.template_library.get(template_id)
        answer = messagebox.askyesno("移出模板库", f"确定将“{entry.name}”移出模板库吗？\n模板图片和 mask 文件会保留。", parent=self)
        if not answer:
            return
        self.template_library.remove(template_id)
        self._invalidate_matcher()
        self._refresh_library_tree()
        self.status_var.set(f"已移出模板库：{entry.name}")

    def load_detection_image(self) -> None:
        path = filedialog.askopenfilename(title="选择检测图像", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            self.detection_image = read_image(path)
            self.detection_path = Path(path)
            self.annotated_image = None
            self.multi_result = None
            self.loaded_detection_var.set(Path(path).name)
            self.detection_canvas.set_image(self.detection_image)
            self._show_matches([])
            self.status_var.set(f"已加载检测图像：{path}")
            self._update_ui_state()
        except Exception as exc:
            messagebox.showerror("图像加载失败", str(exc), parent=self)

    def load_demo_case(self) -> None:
        try:
            demo = prepare_template_demo(self.template_library, self.data_root / "samples" / "template_demo")
            self._invalidate_matcher()
            self._refresh_library_tree(select_id=demo.template_ids[0])
            self.detection_path = demo.detection_path
            self.detection_image = read_image(demo.detection_path)
            self.annotated_image = None
            self.multi_result = None
            self.loaded_detection_var.set(f"合成自检：{demo.detection_path.name}")
            self.detection_canvas.set_image(self.detection_image)
            self._show_matches([])
            self.status_var.set("内置合成自检已加载：仅用于验证 UI 和结果链路，不代表真实现场精度……")
            self.after(120, self.start_detection)
        except Exception as exc:
            messagebox.showerror("合成自检加载失败", str(exc), parent=self)

    def _parameters(self) -> MatchParameters:
        return MatchParameters(
            score_threshold=float(self.param_vars["score"].get()),
            angle_min=float(self.param_vars["angle_min"].get()),
            angle_max=float(self.param_vars["angle_max"].get()),
            angle_step=float(self.param_vars["angle_step"].get()),
            scale_min=float(self.param_vars["scale_min"].get()),
            scale_max=float(self.param_vars["scale_max"].get()),
            scale_step=float(self.param_vars["scale_step"].get()),
            nms_iou_threshold=float(self.param_vars["nms"].get()),
            use_edges=self.use_edges_var.get(),
            feature_mode={label: mode for mode, label in FEATURE_MODE_LABELS.items()}.get(self.feature_mode_var.get(), "pose_tolerant"),
        )

    def start_detection(self) -> None:
        if self.detection_image is None:
            messagebox.showwarning("缺少图像", "请先加载检测图像。", parent=self)
            return
        try:
            if self.multi_matcher is None:
                self.multi_matcher = MultiTemplateMatcher.from_library(self.template_library)
            if self.multi_matcher.valid_enabled_count == 0:
                raise ValueError("模板库中没有有效且已启用的模板")
        except Exception as exc:
            messagebox.showerror("无法开始检测", str(exc), parent=self)
            return
        self.detect_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.detection_running = True
        self._update_ui_state()
        self.progress.start(12)
        self.status_var.set(f"正在使用 {self.multi_matcher.valid_enabled_count} 个模板进行匹配，请稍候……")
        image = self.detection_image.copy()
        matcher = self.multi_matcher

        def work() -> None:
            try:
                result = matcher.match(image)
                self.worker_queue.put(("success", result, draw_multi_template_matches(image, result)))
            except Exception as exc:
                self.worker_queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll_worker)

    def _poll_worker(self) -> None:
        try:
            event = self.worker_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_worker)
            return
        self.progress.stop()
        self.detection_running = False
        if event[0] == "error":
            self.status_var.set("检测失败")
            self._update_ui_state()
            messagebox.showerror("检测失败", str(event[1]), parent=self)
            return
        self.multi_result = event[1]
        self.annotated_image = event[2]
        self.detection_canvas.set_image(self.annotated_image)
        self._show_matches(list(self.multi_result.objects))
        self._update_ui_state()
        if self.multi_result.processing_scale < 0.999:
            source = self.multi_result.source_image_size
            processed = self.multi_result.processing_image_size
            self.status_var.set(
                f"检测完成：找到 {self.multi_result.object_count} 个目标；"
                f"大图已自动预处理 {source[0]}x{source[1]} → {processed[0]}x{processed[1]}，"
                "坐标已映射回原图。"
            )
            return
        error_note = f"，{len(self.multi_result.template_errors)} 个模板异常" if self.multi_result.template_errors else ""
        self.status_var.set(f"检测完成：找到 {self.multi_result.object_count} 个目标，待确认 {self.multi_result.ambiguous_count} 个{error_note}")

    def _show_matches(self, matches: list[RecognizedObject]) -> None:
        self.result_tree.delete(*self.result_tree.get_children())
        for match in matches:
            candidates = " / ".join(f"{item.template_name}:{item.score:.3f}" for item in match.candidate_templates)
            status = "已确认" if match.classification_status == "confirmed" else "待确认"
            self.result_tree.insert(
                "", "end",
                values=(match.object_id, status, match.template_name, f"{match.center_x:.2f}", f"{match.center_y:.2f}", f"{match.angle_deg:.2f}", f"{match.score:.4f}", f"{match.scale:.3f}", candidates),
                tags=(match.classification_status,),
            )
        if self.multi_result is None:
            self.count_var.set("目标数量：0")
            self.total_metric_var.set("0")
            self.ambiguous_metric_var.set("0")
            self.type_metric_var.set("0")
            return
        breakdown = "，".join(f"{name}:{count}" for name, count in self.multi_result.counts_by_template.items())
        self.count_var.set(f"总数：{len(matches)}  待确认：{self.multi_result.ambiguous_count}" + (f"  {breakdown}" if breakdown else ""))
        self.total_metric_var.set(str(len(matches)))
        self.ambiguous_metric_var.set(str(self.multi_result.ambiguous_count))
        self.type_metric_var.set(str(sum(1 for count in self.multi_result.counts_by_template.values() if count > 0)))

    def _browse_live_source(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")],
        )
        if path:
            self.live_source_var.set(path)

    def _live_roi_changed(self, _event=None) -> None:
        roi = self.live_canvas.roi
        with self.live_frame_lock:
            self.live_detection_roi = roi
        if roi is None:
            self.live_roi_var.set("检测 ROI：整幅图像")
        else:
            self.live_roi_var.set(f"检测 ROI：x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")

    def _clear_live_roi(self) -> None:
        self.live_canvas.roi = None
        self.live_canvas._draw_roi()
        self._live_roi_changed()

    @staticmethod
    def _optional_number(value: str, converter):
        normalized = value.strip()
        return None if not normalized else converter(normalized)

    def _live_camera_settings(self) -> CameraSettings:
        exposure = self._optional_number(self.live_exposure_var.get(), float)
        return CameraSettings(
            width=self._optional_number(self.live_width_var.get(), int),
            height=self._optional_number(self.live_height_var.get(), int),
            fps=self._optional_number(self.live_fps_var.get(), float),
            exposure=exposure,
            auto_exposure=exposure is None,
            backend=self.live_backend_var.get(),
            fourcc=self.live_fourcc_var.get().strip() or None,
            buffer_size=1,
        )

    def _live_tracking_config(self) -> TrackingConfig:
        axis = "x" if self.live_axis_var.get().startswith("竖线") else "y"
        direction = {"正向": "positive", "反向": "negative", "双向": "both"}[self.live_direction_var.get()]
        config = TrackingConfig(
            line_axis=axis,
            line_position=float(self.live_line_position_var.get()),
            direction=direction,
        )
        config.validate()
        return config

    def start_live_detection(self) -> None:
        if self.live_running:
            return
        try:
            if self.live_detector_mode_var.get() == "荧光工件分割":
                matcher = FluorescentTextileDetector(FluorescentTextileParameters())
            else:
                matcher = MultiTemplateMatcher.from_library(self.template_library)
                if matcher.valid_enabled_count == 0:
                    raise ValueError("模板库中没有有效且已启用的模板")
            max_edge = int(self.live_max_edge_var.get())
            if max_edge < 640:
                raise ValueError("检测最长边不能小于 640 像素")
            matcher.max_processing_edge = max_edge
            matcher.max_processing_pixels = max_edge * max_edge
            tracking_config = self._live_tracking_config()
            report = RuntimeReport(self.data_root / "outputs" / "runtime_reports", "live")
            source_text = self.live_source_var.get().strip()
            source = OpenCVCameraSource.from_text(source_text, self._live_camera_settings())
            source.open()
            actual_camera = source.actual_settings()
            report.record(
                "device_connect",
                details={"mode": "opencv_camera_or_video", "source": source_text, **actual_camera},
            )
            report.record(
                "input_mode",
                details={"mode": "opencv_camera_or_video", "source": source_text},
            )
        except Exception as exc:
            if "report" in locals():
                report.record_error("device_connect", exc)
                report.close("startup_error")
            messagebox.showerror("实时检测无法启动", str(exc), parent=self)
            return
        self.live_runtime_report = report
        self.live_source = source
        self.live_session = ConveyorSession(
            matcher,
            tracking_config=tracking_config,
            reject_bad_frames=self.live_quality_gate_var.get(),
        )
        self.live_stop_event.clear()
        with self.live_frame_lock:
            self.live_latest_frame = None
            self.live_frame_sequence = 0
        while not self.live_queue.empty():
            try:
                self.live_queue.get_nowait()
            except queue.Empty:
                break
        self.live_running = True
        self.live_start_button.configure(state="disabled")
        self.live_stop_button.configure(state="normal")
        self.status_var.set(
            f"实时检测已启动：{self.live_source_var.get()}｜"
            f"实际采集 {actual_camera['width']:.0f}x{actual_camera['height']:.0f} "
            f"@ {actual_camera['fps']:.1f} FPS｜报告：{report.directory}"
        )

        def capture_worker() -> None:
            frame_interval = 0.0
            if isinstance(source, OpenCVCameraSource) and isinstance(source.source, str) and source.capture is not None:
                fps = float(source.capture.get(cv2.CAP_PROP_FPS))
                if 1.0 <= fps <= 240.0:
                    frame_interval = 1.0 / fps
            try:
                while not self.live_stop_event.is_set():
                    frame = source.read()
                    report.record("camera_capture")
                    with self.live_frame_lock:
                        self.live_latest_frame = frame
                        self.live_frame_sequence += 1
                    if frame_interval and self.live_stop_event.wait(frame_interval):
                        break
            except EOFError:
                self._put_live_event(("stopped", "视频播放结束"))
                self.live_stop_event.set()
            except Exception as exc:
                if not self.live_stop_event.is_set():
                    report.record_error("camera_capture", exc)
                    self._put_live_event(("error", exc))
                    self.live_stop_event.set()

        def detection_worker() -> None:
            last_sequence = -1
            try:
                while not self.live_stop_event.is_set():
                    with self.live_frame_lock:
                        sequence = self.live_frame_sequence
                        frame = None if self.live_latest_frame is None else self.live_latest_frame.copy()
                        detection_roi = self.live_detection_roi
                    if frame is None or sequence == last_sequence:
                        self.live_stop_event.wait(0.01)
                        continue
                    last_sequence = sequence
                    detection_started = perf_counter()
                    result = self.live_session.process_frame(frame, detection_roi=detection_roi)
                    report.record(
                        "detection",
                        duration_ms=(perf_counter() - detection_started) * 1000.0,
                        details={"count": result.current_count, "skipped": result.skipped},
                    )
                    annotated = draw_conveyor_frame(frame, result, tracking_config)
                    self._put_live_event(("frame", result, frame, annotated))
            except Exception as exc:
                if not self.live_stop_event.is_set():
                    report.record_error("detection", exc)
                    self._put_live_event(("error", exc))
                    self.live_stop_event.set()

        threading.Thread(target=capture_worker, daemon=True, name="camera-capture").start()
        threading.Thread(target=detection_worker, daemon=True, name="continuous-detection").start()
        self.after(60, self._poll_live_queue)

    def _put_live_event(self, event: tuple) -> None:
        try:
            self.live_queue.put_nowait(event)
        except queue.Full:
            try:
                self.live_queue.get_nowait()
            except queue.Empty:
                pass
            self.live_queue.put_nowait(event)

    def _poll_live_queue(self) -> None:
        events: list[tuple] = []
        while True:
            try:
                events.append(self.live_queue.get_nowait())
            except queue.Empty:
                break
        for event in events:
            if event[0] == "frame":
                self._update_live_dashboard(event[1], event[2], event[3])
            elif event[0] == "error":
                self.stop_live_detection(status="实时检测失败")
                messagebox.showerror("实时检测失败", str(event[1]), parent=self)
                return
            elif event[0] == "stopped":
                self.stop_live_detection(status=str(event[1]))
                return
        if self.live_running:
            self.after(60, self._poll_live_queue)

    def _update_live_dashboard(
        self,
        result: ConveyorFrameResult,
        raw_frame: np.ndarray,
        annotated: np.ndarray,
    ) -> None:
        self.live_last_result = result
        self.live_last_raw_frame = raw_frame.copy()
        self.live_last_annotated_frame = annotated.copy()
        self.live_canvas.set_image(annotated, clear_roi=False, reset_view=False)
        if self.live_runtime_report is not None:
            self.live_runtime_report.record("display")
        self.live_tree.delete(*self.live_tree.get_children())
        objects = result.detection.objects if result.detection is not None else ()
        for item in objects:
            self.live_tree.insert(
                "",
                "end",
                values=(
                    f"#{item.object_id}", item.template_name, f"{item.center_x:.1f}",
                    f"{item.center_y:.1f}", f"{item.angle_deg:.1f}",
                    (
                        f"{item.pick_point_x:.0f},{item.pick_point_y:.0f}"
                        if item.pick_point_x is not None and item.pick_point_y is not None else "-"
                    ),
                    f"{item.safe_radius_px:.1f}" if item.safe_radius_px is not None else "-",
                    "可吸取" if item.auto_pick_allowed else (
                        "待确认" if item.classification_status == "ambiguous" else "仅定位"
                    ),
                ),
            )
        counts = result.detection.counts_by_template if result.detection is not None else {}
        breakdown = "，".join(f"{name}:{count}" for name, count in counts.items()) or "-"
        self.live_count_var.set(f"当前画面工件：{result.current_count}")
        self.live_breakdown_var.set(f"当前分类：{breakdown}")
        self.live_cumulative_var.set(f"过线累计（辅助）：{result.tracking.cumulative_total}")
        issue_labels = {
            "underexposed": "欠曝", "overexposed": "过曝", "blurred": "模糊",
            "uneven_illumination": "光照不均",
        }
        quality = "正常" if result.quality.passed else "、".join(issue_labels.get(item, item) for item in result.quality.issues)
        self.live_quality_var.set(
            f"画面质量：{quality}｜亮度 {result.quality.mean_brightness:.1f}｜清晰度 {result.quality.sharpness:.1f}"
        )
        self.live_performance_var.set(
            f"处理耗时：{result.elapsed_ms:.0f} ms｜活动轨迹：{result.tracking.active_track_count}"
            + ("｜本帧已跳过" if result.skipped else "")
        )

    def save_live_diagnostic(self) -> None:
        if self.live_last_raw_frame is None or self.live_last_annotated_frame is None or self.live_last_result is None:
            messagebox.showwarning("没有可保存结果", "请先启动实时检测并等待一帧结果。", parent=self)
            return
        try:
            run_dir = self.data_root / "outputs" / "workstation_diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir.mkdir(parents=True, exist_ok=False)
            write_image(run_dir / "01_original.png", self.live_last_raw_frame)
            write_image(run_dir / "02_annotated.png", self.live_last_annotated_frame)
            result = self.live_last_result
            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "quality": asdict(result.quality),
                "elapsed_ms": result.elapsed_ms,
                "skipped": result.skipped,
                "input_image_size": result.input_image_size,
                "detection_roi": result.detection_roi,
                "current_count": result.current_count,
                "detection": result.detection.to_output_dict() if result.detection is not None else None,
            }
            (run_dir / "results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if result.detection is not None and result.detection.debug_images:
                debug_dir = run_dir / "debug"
                for name, debug_image in result.detection.debug_images.items():
                    write_image(debug_dir / f"{name}.png", debug_image)
            self.status_var.set(f"实时诊断已保存：{run_dir}")
            messagebox.showinfo("诊断已保存", str(run_dir), parent=self)
        except Exception as exc:
            messagebox.showerror("保存诊断失败", str(exc), parent=self)

    def reset_live_counts(self) -> None:
        if self.live_session is not None:
            self.live_session.reset_counts()
        self.live_count_var.set("当前画面工件：0")
        self.live_breakdown_var.set("当前分类：-")
        self.live_cumulative_var.set("过线累计（辅助）：0")
        self.status_var.set("实时累计计数已清零")

    def stop_live_detection(self, status: str = "实时检测已停止") -> None:
        self.live_stop_event.set()
        source, self.live_source = self.live_source, None
        if source is not None:
            source.close()
        self.live_running = False
        report, self.live_runtime_report = self.live_runtime_report, None
        if report is not None:
            report.close(status)
        if hasattr(self, "live_start_button"):
            self.live_start_button.configure(state="normal")
            self.live_stop_button.configure(state="disabled")
        self.status_var.set(status)

    def _on_close(self) -> None:
        self.stop_live_detection()
        self.destroy()

    def save_results(self) -> None:
        if self.annotated_image is None or self.detection_path is None or self.multi_result is None:
            return
        folder = filedialog.askdirectory(title="选择结果保存目录", initialdir=str(self.data_root / "outputs"))
        if not folder:
            return
        try:
            run_dir = write_multi_template_result(folder, self.detection_path, self.annotated_image, self.multi_result)
            self.status_var.set(f"结果已保存：{run_dir}")
            messagebox.showinfo("保存完成", f"标注图、JSON 和 CSV 已保存至：\n{run_dir}", parent=self)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Industrial SegPose template matching UI")
    parser.add_argument("--check", action="store_true", help="Create and close the UI for environment verification")
    args = parser.parse_args(argv)
    app = TemplateMatchingApp()
    if args.check:
        app.withdraw()
        if (
            not hasattr(app, "library_tree")
            or not hasattr(app, "reference_canvas")
            or not hasattr(app, "detection_canvas")
            or not isinstance(app.template_library, TemplateLibrary)
        ):
            raise RuntimeError("Template library UI check failed")
        editor = MaskEditorCanvas(app, np.zeros((40, 60, 3), dtype=np.uint8))
        editor.fill_all()
        if np.count_nonzero(editor.mask) != editor.mask.size:
            raise RuntimeError("Irregular mask editor check failed")
        editor.destroy()
        app.update_idletasks()
        app.destroy()
        print("Template matching UI check passed")
        return 0
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
