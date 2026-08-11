"""Tk desktop UI for creating templates and detecting rotated objects."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from . import __version__
from .camera import OpenCVCameraSource
from .calibration_ui import CalibrationWorkbench
from .io.image_reader import read_image
from .k230_workbench import (
    K230CaptureImage,
    build_workbench_deployment,
    scan_k230_captures,
    validate_workbench_images,
)
from .runtime import ConveyorSession, ConveyorFrameResult, draw_conveyor_frame
from .template_matching import (
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
from .template_demo import prepare_template_demo
from .tracking import TrackingConfig
from .wpd_sync import sync_canmv_captures, sync_canmv_diagnostics


IMAGE_TYPES = [("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")]


def _application_data_root() -> Path:
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.home() / "IndustrialSegPose"


FEATURE_MODE_LABELS = {
    "pose_tolerant": "柔性姿态",
    "dark_textile": "暗色纹理",
    "textile_chroma": "纺织色度",
    "edges": "轮廓边缘",
    "gray": "灰度纹理",
    "auto": "兼容模式",
}


class ImageCanvas(tk.Canvas):
    """Fit-to-window image canvas with an optional draggable ROI."""

    def __init__(self, master, selectable: bool = False, **kwargs):
        super().__init__(master, background="#202124", highlightthickness=0, **kwargs)
        self.selectable = selectable
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

    def zoom_in(self) -> None:
        self._zoom_at(1.25, self.winfo_width() / 2, self.winfo_height() / 2)

    def zoom_out(self) -> None:
        self._zoom_at(0.8, self.winfo_width() / 2, self.winfo_height() / 2)

    def fit_to_window(self) -> None:
        self.zoom_factor = 1.0
        self.pan_x = self.pan_y = 0.0
        self._render()

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
                fill="#b0b3b8",
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
        self.create_rectangle(x1, y1, x2, y2, outline="#00e676", width=2, dash=(6, 3), tags="roi")
        self.create_text(x1 + 4, max(y1 - 12, 10), text=f"{width} x {height}", anchor="w", fill="#00e676", tags="roi")


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
            tint[:, :, 1] = 255
            source[selected] = cv2.addWeighted(source, 0.58, tint, 0.42, 0)[selected]
        return source

    def _render(self) -> None:
        super()._render()
        if len(self.polygon_points) >= 1:
            points = [(self.offset_x + x * self.scale, self.offset_y + y * self.scale) for x, y in self.polygon_points]
            flattened = [coordinate for point in points for coordinate in point]
            if len(points) >= 2:
                self.create_line(*flattened, fill="#ffd740", width=2, tags="polygon")
            for x, y in points:
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffd740", outline="", tags="polygon")

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
        self.assist_status_var = tk.StringVar(value="算法结果只作为初始 mask，请检查绿色区域并用画笔修正。")
        ttk.Label(assist, textvariable=self.assist_status_var, foreground="#555555").pack(side="left", padx=10)
        ttk.Label(self, text="多边形：依次点击轮廓点，双击或点击“完成多边形”；画笔：左键添加，右键擦除。绿色区域为模板有效形状。", foreground="#555555", padding=(10, 0)).pack(fill="x")
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
        self.minsize(980, 650)
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
        self.live_queue: queue.Queue = queue.Queue(maxsize=3)
        self.live_source: OpenCVCameraSource | None = None
        self.live_session: ConveyorSession | None = None
        self.live_stop_event = threading.Event()
        self.live_frame_lock = threading.Lock()
        self.live_latest_frame: np.ndarray | None = None
        self.live_frame_sequence = 0
        self.live_running = False
        self._configure_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.configure(background="#F3F6FA")
        style.configure("TFrame", background="#F3F6FA")
        style.configure("Header.TFrame", background="#17233C")
        style.configure("HeaderTitle.TLabel", background="#17233C", foreground="#FFFFFF", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("HeaderSub.TLabel", background="#17233C", foreground="#AFC2E3", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background="#F3F6FA", foreground="#15213A", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Section.TLabel", background="#FFFFFF", foreground="#24324A", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Hint.TLabel", background="#FFFFFF", foreground="#68758A", font=("Microsoft YaHei UI", 9))
        style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        style.configure("Card.TLabelframe", background="#FFFFFF", bordercolor="#DDE4EE", relief="solid")
        style.configure("Card.TLabelframe.Label", background="#FFFFFF", foreground="#24324A", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(10, 7), borderwidth=0)
        style.configure("Primary.TButton", background="#2563EB", foreground="#FFFFFF", font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Primary.TButton", background=[("active", "#1D4ED8"), ("disabled", "#9BB7EE")])
        style.configure("Secondary.TButton", background="#E8EEF8", foreground="#244064")
        style.map("Secondary.TButton", background=[("active", "#D9E4F4")])
        style.configure("Danger.TButton", background="#FEECEC", foreground="#B42318")
        style.map("Danger.TButton", background=[("active", "#FDDADA")])
        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF", foreground="#26354D", rowheight=30, borderwidth=0, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", background="#EAF0F8", foreground="#34445F", relief="flat", font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#DCE8FF")], foreground=[("selected", "#173B78")])
        style.configure("TNotebook", background="#F3F6FA", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Microsoft YaHei UI", 10, "bold"), background="#E6EBF3", foreground="#5B687D")
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#1D4ED8")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(20, 14), style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="FlexPose Vision", style="HeaderTitle.TLabel").pack(side="left")
        ttk.Label(header, text="纺织工件 · 模板建立 · 多目标识别定位计数", style="HeaderSub.TLabel").pack(side="left", padx=20, pady=(6, 0))
        ttk.Label(header, text=f"v{__version__}", style="HeaderSub.TLabel").pack(side="right", pady=(6, 0))
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.template_tab = ttk.Frame(self.notebook, padding=8)
        self.detect_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.template_tab, text=" 1. 建立模板 ")
        self.notebook.add(self.detect_tab, text=" 2. 模板检测 ")
        self._build_template_tab()
        self._build_detect_tab()
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6), background="#E8EEF7", foreground="#40516D").pack(fill="x")
        self._refresh_library_tree()
        if self.library_load_error:
            self.status_var.set(f"模板库加载失败：{self.library_load_error}")

    def _build_template_tab(self) -> None:
        source_card = ttk.Frame(self.template_tab, padding=(16, 12), style="Card.TFrame")
        source_card.pack(fill="x", pady=(0, 8))
        source_text = ttk.Frame(source_card, style="Card.TFrame")
        source_text.pack(side="left", fill="x", expand=True)
        ttk.Label(source_text, text="01  选择基准图像", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            source_text,
            text="推荐先自动分割；需要使用设备采集图时，再进入 K230 图像工作台。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        source_actions = ttk.Frame(source_card, style="Card.TFrame")
        source_actions.pack(side="right")
        ttk.Button(
            source_actions,
            text="导入图片并自动分割",
            command=lambda: self.load_reference(auto_extract=True),
            style="Primary.TButton",
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            source_actions,
            text="从 K230 选择采集图",
            command=self.open_k230_workbench,
            style="Secondary.TButton",
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            source_actions,
            text="手动加载图片",
            command=self.load_reference,
            style="Secondary.TButton",
        ).pack(side="left")

        workspace = ttk.Frame(self.template_tab)
        workspace.pack(fill="both", expand=True, pady=(0, 8))
        sidebar = ttk.Frame(workspace, width=290, padding=16, style="Card.TFrame")
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="02  模板信息", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(sidebar, text="模板名称（工件类型）", style="Hint.TLabel").pack(anchor="w")
        self.template_name_var = tk.StringVar(value="")
        ttk.Entry(sidebar, textvariable=self.template_name_var).pack(fill="x", pady=(4, 12))
        self.roi_var = tk.StringVar(value="请先加载图像，然后框选单个工件")
        ttk.Label(sidebar, textvariable=self.roi_var, style="Hint.TLabel", wraplength=250).pack(anchor="w")
        ttk.Separator(sidebar).pack(fill="x", pady=16)
        ttk.Label(sidebar, text="03  轮廓与 Mask", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Button(
            sidebar,
            text="检查并修正分割轮廓",
            command=self.edit_template_shape,
            style="Secondary.TButton",
        ).pack(fill="x", pady=(0, 10))
        self.template_quality_var = tk.StringVar(value="Mask 质量：尚未建立")
        ttk.Label(
            sidebar,
            textvariable=self.template_quality_var,
            style="Hint.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            sidebar,
            text="提示：粗 ROI 四周保留少量背景；自动结果不完整时再用画笔修正。",
            style="Hint.TLabel",
            wraplength=250,
            justify="left",
        ).pack(side="bottom", anchor="w")

        canvas_card = ttk.Frame(workspace, padding=10, style="Card.TFrame")
        canvas_card.pack(side="left", fill="both", expand=True)
        canvas_header = ttk.Frame(canvas_card, style="Card.TFrame")
        canvas_header.pack(fill="x", pady=(0, 8))
        ttk.Label(canvas_header, text="基准图像与粗 ROI", style="Section.TLabel").pack(side="left")
        self.reference_canvas = ImageCanvas(canvas_card, selectable=True)
        ttk.Button(canvas_header, text="放大 ＋", command=self.reference_canvas.zoom_in, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(canvas_header, text="缩小 －", command=self.reference_canvas.zoom_out, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(canvas_header, text="适应窗口", command=self.reference_canvas.fit_to_window, style="Secondary.TButton").pack(side="right", padx=(8, 0))
        self.reference_canvas.pack(fill="both", expand=True)
        self.reference_canvas.bind("<<RoiChanged>>", self._roi_changed)

        action_card = ttk.Frame(self.template_tab, padding=(16, 12), style="Card.TFrame")
        action_card.pack(fill="x")
        action_text = ttk.Frame(action_card, style="Card.TFrame")
        action_text.pack(side="left", fill="x", expand=True)
        ttk.Label(action_text, text="04  保存与部署", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            action_text,
            text="保存后会加入电脑端模板库；部署包会一次性包含模板库中的全部有效模板。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        action_buttons = ttk.Frame(action_card, style="Card.TFrame")
        action_buttons.pack(side="right")
        ttk.Button(
            action_buttons,
            text="保存并加入模板库",
            command=self.save_template,
            style="Primary.TButton",
        ).pack(side="left", padx=(0, 8))
        self.deploy_button = ttk.Button(
            action_buttons,
            text="生成完整 K230 部署包",
            command=self.build_k230_package,
            style="Primary.TButton",
        )
        self.deploy_button.pack(side="left")

    def _build_detect_tab(self) -> None:
        top = ttk.Frame(self.detect_tab, padding=(12, 10), style="Card.TFrame")
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="模板库与离线验证", style="Section.TLabel").pack(side="left", padx=(0, 14))
        ttk.Button(top, text="启用/停用", command=self.toggle_selected_template, style="Secondary.TButton").pack(side="left", padx=4)
        ttk.Button(top, text="编辑参数", command=self.edit_selected_template, style="Secondary.TButton").pack(side="left", padx=4)
        ttk.Button(top, text="重命名", command=self.rename_selected_template, style="Secondary.TButton").pack(side="left", padx=4)
        ttk.Button(top, text="移出模板库", command=self.remove_selected_template, style="Danger.TButton").pack(side="left", padx=4)
        self.loaded_detection_var = tk.StringVar(value="未加载检测图像")
        ttk.Button(top, text="选择检测图像", command=self.load_detection_image, style="Primary.TButton").pack(side="right")
        ttk.Label(top, textvariable=self.loaded_detection_var, style="Hint.TLabel").pack(side="right", padx=8)

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
        self.library_tree.pack(side="left", fill="x", expand=True)
        library_scroll.pack(side="right", fill="y")
        self.library_tree.bind("<Double-1>", lambda _event: self.toggle_selected_template())

        params = ttk.LabelFrame(self.detect_tab, text="默认匹配参数", padding=(8, 5), style="Card.TLabelframe")
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
        self.feature_mode_var = tk.StringVar(value="柔性姿态")
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

        summary = ttk.Frame(result_frame)
        summary.pack(fill="x", padx=(8, 0), pady=(0, 5))
        self.count_var = tk.StringVar(value="目标数量：0")
        ttk.Label(summary, textvariable=self.count_var, style="Title.TLabel").pack(side="left")
        columns = ("id", "status", "template", "x", "y", "angle", "score", "scale", "candidates")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=16)
        labels = {"id": "编号", "status": "识别状态", "template": "目标类型", "x": "中心 X", "y": "中心 Y", "angle": "角度°", "score": "得分", "scale": "尺度", "candidates": "候选类型"}
        widths = {"id": 42, "status": 72, "template": 82, "x": 70, "y": 70, "angle": 65, "score": 65, "scale": 52, "candidates": 140}
        for column in columns:
            self.result_tree.heading(column, text=labels[column])
            self.result_tree.column(column, width=widths[column], anchor="center")
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True, padx=(8, 0))
        scrollbar.pack(side="right", fill="y")

    def _build_live_tab(self) -> None:
        controls = ttk.LabelFrame(self.live_tab, text="相机 / 视频源与计数设置", padding=8, style="Card.TLabelframe")
        controls.pack(fill="x", pady=(0, 7))
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
        self.live_quality_gate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="质量不合格时跳过检测", variable=self.live_quality_gate_var).grid(row=1, column=5, padx=7)
        self.live_start_button = ttk.Button(controls, text="▶ 启动实时检测", command=self.start_live_detection, style="Primary.TButton")
        self.live_start_button.grid(row=0, column=6, rowspan=2, padx=5)
        self.live_stop_button = ttk.Button(controls, text="■ 停止", command=self.stop_live_detection, state="disabled", style="Danger.TButton")
        self.live_stop_button.grid(row=0, column=7, rowspan=2, padx=5)
        ttk.Button(controls, text="清零累计", command=self.reset_live_counts, style="Secondary.TButton").grid(row=0, column=8, rowspan=2, padx=5)
        controls.columnconfigure(0, weight=1)

        pane = ttk.Panedwindow(self.live_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        image_frame = ttk.Frame(pane, padding=8, style="Card.TFrame")
        dashboard = ttk.Frame(pane, padding=12, style="Card.TFrame")
        pane.add(image_frame, weight=4)
        pane.add(dashboard, weight=2)
        toolbar = ttk.Frame(image_frame, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="实时画面｜黄色线为计数线", style="Hint.TLabel").pack(side="left")
        self.live_canvas = ImageCanvas(image_frame)
        ttk.Button(toolbar, text="放大 ＋", command=self.live_canvas.zoom_in, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="缩小 －", command=self.live_canvas.zoom_out, style="Secondary.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="适应窗口", command=self.live_canvas.fit_to_window, style="Secondary.TButton").pack(side="right")
        self.live_canvas.pack(fill="both", expand=True)

        self.live_count_var = tk.StringVar(value="累计计数：0")
        self.live_breakdown_var = tk.StringVar(value="分类计数：-")
        self.live_quality_var = tk.StringVar(value="画面质量：未启动")
        self.live_performance_var = tk.StringVar(value="处理耗时：-")
        ttk.Label(dashboard, textvariable=self.live_count_var, style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(dashboard, textvariable=self.live_breakdown_var, style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(dashboard, textvariable=self.live_quality_var, style="Hint.TLabel", wraplength=360).pack(anchor="w", pady=(0, 4))
        ttk.Label(dashboard, textvariable=self.live_performance_var, style="Hint.TLabel").pack(anchor="w", pady=(0, 12))
        columns = ("track", "template", "x", "y", "angle", "counted")
        self.live_tree = ttk.Treeview(dashboard, columns=columns, show="headings", height=15)
        labels = {"track": "轨迹", "template": "类型", "x": "中心X", "y": "中心Y", "angle": "角度°", "counted": "计数"}
        widths = {"track": 55, "template": 100, "x": 70, "y": 70, "angle": 70, "counted": 55}
        for column in columns:
            self.live_tree.heading(column, text=labels[column])
            self.live_tree.column(column, width=widths[column], anchor="center")
        self.live_tree.pack(fill="both", expand=True)

    def _roi_changed(self, _event=None) -> None:
        roi = self.reference_canvas.roi
        self.template_mask = None
        self.template_quality_var.set("Mask质量：ROI已改变，请重新提取轮廓")
        self.roi_var.set(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}" if roi else "未选择 ROI")

    def load_reference(self, auto_extract: bool = False) -> None:
        path = filedialog.askopenfilename(title="选择参考图像", filetypes=IMAGE_TYPES)
        if not path:
            return
        self.load_reference_path(path, auto_extract=auto_extract)

    def load_reference_path(self, path: str | Path, auto_extract: bool = False) -> None:
        """Load a reference selected by a file dialog or the K230 workbench."""

        try:
            self.reference_image = read_image(path)
            self.reference_path = Path(path)
            self.template_mask = None
            self.template_quality_var.set("Mask质量：尚未建立")
            self.reference_canvas.set_image(self.reference_image)
            self.status_var.set(f"已加载参考图像：{path}")
            if auto_extract:
                self.after(80, self.auto_locate_template)
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc), parent=self)

    def open_k230_workbench(self) -> None:
        K230TemplateWorkbench(self)

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
                self.status_var.set("自动模板绘制完成，可设置名称并保存到模板库")
            else:
                self.status_var.set("已取消自动模板编辑，仍可手动框选")
        except Exception as exc:
            self.status_var.set("自动模板绘制失败")
            messagebox.showerror("自动绘制失败", str(exc), parent=self)
        finally:
            self.configure(cursor="")

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

    def save_template(self) -> None:
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
            self.library_tree.insert("", "end", iid=entry.template_id, values=values)
        if select_id and self.library_tree.exists(select_id):
            self.library_tree.selection_set(select_id)
            self.library_tree.see(select_id)

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
        self.detect_button.configure(state="normal")
        if event[0] == "error":
            self.status_var.set("检测失败")
            messagebox.showerror("检测失败", str(event[1]), parent=self)
            return
        self.multi_result = event[1]
        self.annotated_image = event[2]
        self.detection_canvas.set_image(self.annotated_image)
        self._show_matches(list(self.multi_result.objects))
        self.save_button.configure(state="normal")
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
            self.result_tree.insert("", "end", values=(match.object_id, status, match.template_name, f"{match.center_x:.2f}", f"{match.center_y:.2f}", f"{match.angle_deg:.2f}", f"{match.score:.4f}", f"{match.scale:.3f}", candidates))
        if self.multi_result is None:
            self.count_var.set("目标数量：0")
            return
        breakdown = "，".join(f"{name}:{count}" for name, count in self.multi_result.counts_by_template.items())
        self.count_var.set(f"总数：{len(matches)}  待确认：{self.multi_result.ambiguous_count}" + (f"  {breakdown}" if breakdown else ""))

    def _browse_live_source(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")],
        )
        if path:
            self.live_source_var.set(path)

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
            matcher = MultiTemplateMatcher.from_library(self.template_library)
            if matcher.valid_enabled_count == 0:
                raise ValueError("模板库中没有有效且已启用的模板")
            tracking_config = self._live_tracking_config()
            source = OpenCVCameraSource.from_text(self.live_source_var.get())
            source.open()
        except Exception as exc:
            messagebox.showerror("实时检测无法启动", str(exc), parent=self)
            return
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
        self.status_var.set(f"实时检测已启动：{self.live_source_var.get()}")

        def capture_worker() -> None:
            frame_interval = 0.0
            if isinstance(source.source, str) and source.capture is not None:
                fps = float(source.capture.get(cv2.CAP_PROP_FPS))
                if 1.0 <= fps <= 240.0:
                    frame_interval = 1.0 / fps
            try:
                while not self.live_stop_event.is_set():
                    frame = source.read()
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
                    self._put_live_event(("error", exc))
                    self.live_stop_event.set()

        def detection_worker() -> None:
            last_sequence = -1
            try:
                while not self.live_stop_event.is_set():
                    with self.live_frame_lock:
                        sequence = self.live_frame_sequence
                        frame = None if self.live_latest_frame is None else self.live_latest_frame.copy()
                    if frame is None or sequence == last_sequence:
                        self.live_stop_event.wait(0.01)
                        continue
                    last_sequence = sequence
                    result = self.live_session.process_frame(frame)
                    annotated = draw_conveyor_frame(frame, result, tracking_config)
                    self._put_live_event(("frame", result, annotated))
            except Exception as exc:
                if not self.live_stop_event.is_set():
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
                self._update_live_dashboard(event[1], event[2])
            elif event[0] == "error":
                self.stop_live_detection(status="实时检测失败")
                messagebox.showerror("实时检测失败", str(event[1]), parent=self)
                return
            elif event[0] == "stopped":
                self.stop_live_detection(status=str(event[1]))
                return
        if self.live_running:
            self.after(60, self._poll_live_queue)

    def _update_live_dashboard(self, result: ConveyorFrameResult, annotated: np.ndarray) -> None:
        self.live_canvas.set_image(annotated, reset_view=False)
        self.live_tree.delete(*self.live_tree.get_children())
        for item in result.tracking.observations:
            self.live_tree.insert(
                "",
                "end",
                values=(
                    f"T{item.track_id}", item.template_name, f"{item.center_x:.1f}",
                    f"{item.center_y:.1f}", f"{item.angle_deg:.1f}", "新计数" if item.counted_now else ("已计" if item.counted else "-"),
                ),
            )
        counts = result.tracking.counts_by_template
        breakdown = "，".join(f"{name}:{count}" for name, count in counts.items()) or "-"
        self.live_count_var.set(f"累计计数：{result.tracking.cumulative_total}")
        self.live_breakdown_var.set(f"分类计数：{breakdown}")
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

    def reset_live_counts(self) -> None:
        if self.live_session is not None:
            self.live_session.reset_counts()
        self.live_count_var.set("累计计数：0")
        self.live_breakdown_var.set("分类计数：-")
        self.status_var.set("实时累计计数已清零")

    def stop_live_detection(self, status: str = "实时检测已停止") -> None:
        self.live_stop_event.set()
        source, self.live_source = self.live_source, None
        if source is not None:
            source.close()
        self.live_running = False
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

    def build_k230_package(self) -> None:
        """Export the complete persistent library from the primary authoring page."""
        output = self.data_root / "build" / "k230_sdcard"
        self.deploy_button.configure(state="disabled")
        self.status_var.set("正在导出完整模板库并生成K230部署包……")

        def worker() -> None:
            try:
                result = build_workbench_deployment(self.data_root, output)
                self.after(0, lambda result=result: done(result, None))
            except Exception as exc:
                self.after(0, lambda exc=exc: done(None, exc))

        def done(result, error) -> None:
            self.deploy_button.configure(state="normal")
            if error is not None:
                self.status_var.set("K230部署包生成失败")
                messagebox.showerror("生成失败", str(error), parent=self)
                return
            app_root, check = result
            detail = (
                f"已导出 {check.exported_templates} 个有效模板"
                f"（启用 {check.enabled_templates}，停用 {check.disabled_templates}）"
            )
            if check.invalid_templates:
                detail += f"，跳过 {len(check.invalid_templates)} 个异常模板"
            self.status_var.set(f"{detail}：{app_root}")
            messagebox.showinfo(
                "部署包已生成",
                f"{detail}\n\n停用模板也会写入部署包，可在K230模板库中重新启用。"
                f"\n\n复制此目录到SD卡：\n{app_root}",
                parent=self,
            )

        threading.Thread(target=worker, daemon=True, name="k230-deployment-build").start()


class K230TemplateWorkbench(tk.Toplevel):
    """Desktop authoring bridge for images captured by a K230 module."""

    def __init__(self, parent: "TemplateMatchingApp"):
        super().__init__(parent)
        self.parent = parent
        self.title("K230采集图像与模板工作台")
        self.geometry("1120x720")
        self.minsize(900, 600)
        self.transient(parent)
        self.captures: list[K230CaptureImage] = []
        self.source_var = tk.StringVar(value=str(parent.data_root / "captures"))
        self.summary_var = tk.StringVar(value="选择K230的SD卡、industrial_vision目录或已复制的captures目录")
        self._build()

    def _build(self) -> None:
        header = ttk.Frame(self, padding=14, style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="K230 模板工作台", style="HeaderTitle.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="采集图像 → 自动分割/人工修正 → 批量回放",
            style="HeaderSub.TLabel",
        ).pack(side="left", padx=18, pady=(6, 0))

        source = ttk.Frame(self, padding=(12, 10), style="Card.TFrame")
        source.pack(fill="x", padx=10, pady=10)
        ttk.Label(source, text="采集目录", style="Section.TLabel").pack(side="left", padx=(0, 8))
        ttk.Entry(source, textvariable=self.source_var).pack(side="left", fill="x", expand=True)
        ttk.Button(source, text="浏览", command=self._browse, style="Secondary.TButton").pack(side="left", padx=6)
        ttk.Button(source, text="从已连接K230同步", command=self._sync_connected, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(source, text="同步分割诊断", command=self._sync_diagnostics, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(source, text="扫描图像", command=self._scan, style="Primary.TButton").pack(side="left")

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10)
        list_card = ttk.Frame(pane, padding=8, style="Card.TFrame")
        preview_card = ttk.Frame(pane, padding=8, style="Card.TFrame")
        pane.add(list_card, weight=3)
        pane.add(preview_card, weight=4)

        ttk.Label(list_card, text="采集图像（可多选用于验证）", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        columns = ("session", "file", "dimensions", "size")
        self.tree = ttk.Treeview(list_card, columns=columns, show="headings", selectmode="extended")
        labels = {"session": "批次", "file": "文件", "dimensions": "尺寸", "size": "大小"}
        widths = {"session": 100, "file": 180, "dimensions": 90, "size": 75}
        for column in columns:
            self.tree.heading(column, text=labels[column])
            self.tree.column(column, width=widths[column], anchor="center")
        scroll = ttk.Scrollbar(list_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._show_selected)

        preview_header = ttk.Frame(preview_card, style="Card.TFrame")
        preview_header.pack(fill="x", pady=(0, 6))
        ttk.Label(preview_header, text="图像预览", style="Section.TLabel").pack(side="left")
        self.preview_canvas = ImageCanvas(preview_card)
        ttk.Button(preview_header, text="放大 ＋", command=self.preview_canvas.zoom_in, style="Secondary.TButton").pack(side="right", padx=3)
        ttk.Button(preview_header, text="缩小 －", command=self.preview_canvas.zoom_out, style="Secondary.TButton").pack(side="right", padx=3)
        ttk.Button(preview_header, text="适应窗口", command=self.preview_canvas.fit_to_window, style="Secondary.TButton").pack(side="right")
        self.preview_canvas.pack(fill="both", expand=True)

        actions = ttk.Frame(self, padding=10, style="Card.TFrame")
        actions.pack(fill="x", padx=10, pady=10)
        ttk.Button(actions, text="① 设为基准图并自动分割", command=self._use_as_reference, style="Primary.TButton").pack(side="left")
        ttk.Button(actions, text="② 批量验证所选图像", command=self._validate_selected, style="Secondary.TButton").pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.summary_var, style="Hint.TLabel", wraplength=430).pack(side="right")

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="选择K230采集目录或SD卡根目录", initialdir=self.source_var.get())
        if path:
            self.source_var.set(path)
            self._scan()

    def _sync_connected(self) -> None:
        self.configure(cursor="watch")
        self.summary_var.set("正在通过Windows便携设备接口同步K230采集图像……")
        self.update_idletasks()
        try:
            path = sync_canmv_captures(self.parent.data_root / "build" / "k230_capture_cache")
            self.source_var.set(str(path))
            self._scan()
            self.summary_var.set(f"已从CanMV同步到本地缓存：{path}")
        except Exception as exc:
            messagebox.showerror("K230同步失败", str(exc), parent=self)
        finally:
            self.configure(cursor="")

    def _sync_diagnostics(self) -> None:
        self.configure(cursor="watch")
        self.summary_var.set("正在同步K230最近一次模板分割诊断……")
        self.update_idletasks()
        try:
            path = sync_canmv_diagnostics(self.parent.data_root / "build" / "k230_diagnostic_cache")
            self.source_var.set(str(path))
            self._scan()
            self.summary_var.set(f"诊断文件已同步：{path}；可将整个目录交给Codex分析")
        except Exception as exc:
            messagebox.showerror("诊断同步失败", str(exc), parent=self)
        finally:
            self.configure(cursor="")

    def _scan(self) -> None:
        try:
            self.captures = scan_k230_captures(self.source_var.get())
            self.tree.delete(*self.tree.get_children())
            for index, capture in enumerate(self.captures):
                self.tree.insert(
                    "", "end", iid=str(index),
                    values=(capture.session, capture.path.name, capture.dimensions, f"{capture.size_bytes / 1024:.0f} KB"),
                )
            self.tree.selection_set("0")
            self.tree.focus("0")
            self._show_selected()
            self.summary_var.set(f"已读取 {len(self.captures)} 张图像；文件保持在原目录，不复制进部署包")
        except Exception as exc:
            messagebox.showerror("扫描失败", str(exc), parent=self)

    def _selected_captures(self) -> list[K230CaptureImage]:
        return [self.captures[int(item)] for item in self.tree.selection() if item.isdigit()]

    def _show_selected(self, _event=None) -> None:
        selected = self._selected_captures()
        if not selected:
            return
        try:
            self.preview_canvas.set_image(read_image(selected[0].path))
            self.summary_var.set(str(selected[0].path))
        except Exception as exc:
            self.summary_var.set(f"预览失败：{exc}")

    def _use_as_reference(self) -> None:
        selected = self._selected_captures()
        if not selected:
            messagebox.showwarning("未选择图像", "请先选择一张采集图像。", parent=self)
            return
        # Windows/Tk forbids iconifying a transient Toplevel.  Transfer focus
        # to the main editor and close this workbench after handing over the
        # selected path; the scheduled mask dialog belongs to the main window.
        self.withdraw()
        self.parent.notebook.select(self.parent.template_tab)
        self.parent.load_reference_path(selected[0].path, auto_extract=True)
        self.parent.lift()
        self.parent.focus_force()
        self.destroy()

    def _run_background(self, label: str, task, done) -> None:
        self.summary_var.set(label)

        def worker() -> None:
            try:
                result = task()
                self.after(0, lambda: done(result, None))
            except Exception as exc:
                self.after(0, lambda exc=exc: done(None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_selected(self) -> None:
        selected = self._selected_captures()
        if not selected:
            messagebox.showwarning("未选择验证图像", "请在左侧选择一张或多张图像。", parent=self)
            return
        output = self.parent.data_root / "build" / "k230_sdcard"
        report_root = self.parent.data_root / "reports" / "k230_workbench" / datetime.now().strftime("%Y%m%d_%H%M%S")

        def task():
            app_root, check = build_workbench_deployment(self.parent.data_root, output)
            report = validate_workbench_images(app_root, [item.path for item in selected], report_root)
            return report, check

        def done(result, error) -> None:
            if error:
                self.summary_var.set("批量验证失败")
                messagebox.showerror("验证失败", str(error), parent=self)
                return
            report, check = result
            self.summary_var.set(f"已用 {check.enabled_templates} 个模板验证 {len(selected)} 张图像：{report}")
            messagebox.showinfo("验证完成", f"验证报告和标注预览已保存：\n{report.parent}", parent=self)

        self._run_background("正在生成K230模板并批量回放……", task, done)

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
            or not hasattr(app, "deploy_button")
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
