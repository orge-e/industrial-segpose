# PySide6 正式迁移说明

## 环境

Qt UI 使用独立环境，避免影响当前 Tkinter 稳定版：

```powershell
cd D:\study-code\industrial-segpose
C:\Users\KeanuReeves\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv-qt
.\.venv-qt\Scripts\python.exe -m pip install -r requirements-qt.txt
```

启动阶段性 Qt UI：

```powershell
.\.venv-qt\Scripts\python.exe -m industrial_segpose.ui_qt.app
```

也可以直接运行：

```powershell
.\scripts\start_qt_ui.ps1
```

原稳定版入口保持不变：

```powershell
python -m industrial_segpose.ui_tk.app
```

## 边界

- `ui_qt` 只负责展示、交互、状态与任务调度。
- 模板文件、模板库和分割算法继续调用既有模块。
- 当前仅正式迁移“建立模板”页面；检测、实时视觉和验证页面是明确占位。
