"""Tk 稳定版界面：提供从主窗口启动的平面标定工作台。"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..calibration.tool import run_calibration_job
from ..ui import COLORS


class CalibrationWorkbench(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.configure(background=COLORS["app"])
        self.title("相机—机械坐标标定")
        self.geometry("760x430")
        self.transient(master)
        self.point_file = tk.StringVar()
        self.output_directory = tk.StringVar(value=str(Path.cwd() / "calibration"))
        self.result = tk.StringVar(value="请选择包含像素坐标和局部机械坐标的JSON点位文件。")
        self._build()

    def _build(self):
        frame = ttk.Frame(self, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="平面视觉标定", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w", pady=(0, 18))
        ttk.Label(frame, text="点位文件").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(5, 15))
        ttk.Entry(row, textvariable=self.point_file).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="选择文件", command=self._browse_points).pack(side="left", padx=(8, 0))
        ttk.Label(frame, text="输出目录").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(5, 18))
        ttk.Entry(row, textvariable=self.output_directory).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="选择目录", command=self._browse_output).pack(side="left", padx=(8, 0))
        ttk.Button(frame, text="计算标定并导出报告", command=self._run).pack(anchor="w", pady=(0, 18))
        ttk.Label(frame, textvariable=self.result, wraplength=700, justify="left").pack(anchor="w")

    def _browse_points(self):
        path = filedialog.askopenfilename(parent=self, filetypes=(("JSON", "*.json"), ("所有文件", "*.*")))
        if path:
            self.point_file.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.output_directory.set(path)

    def _run(self):
        try:
            outputs = run_calibration_job(self.point_file.get(), self.output_directory.get())
            calibration = outputs["calibration"]
            self.result.set("标定完成。标定文件、逐点CSV和Markdown报告已保存到：\n" + str(calibration.parent))
        except Exception as error:
            messagebox.showerror("标定失败", str(error), parent=self)
