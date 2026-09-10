# FlexPose Vision PySide6 建立模板阶段迁移报告

## 结论

“建立模板”页面已从独立高保真 Prototype 迁移为正式 PySide6 页面，并通过 Service/Adapter 调用现有模板、分割和模板库逻辑。Tkinter 稳定版入口及既有业务格式未被替换。

## 独立环境

| 组件 | 版本 |
|---|---:|
| Python | 3.12 |
| PySide6 Essentials | 6.10.3 |
| Shiboken6 | 6.10.3 |
| NumPy | 2.2.6 |
| OpenCV | 4.12.0 |

环境位于 `.venv-qt`，未复用 Anaconda 中存在 ABI 冲突的旧 PySide6。实测导入 QtCore、QtSvg、NumPy 和 OpenCV 时无 Shiboken/NumPy ABI 警告。

## Qt 目录结构

```text
industrial_segpose/ui_qt/
├─ app.py                     Qt 阶段性入口
├─ main_window.py             应用壳、导航、状态栏
├─ theme/
│  ├─ tokens.py               冻结的颜色与间距 Token
│  ├─ stylesheet.py           全局 QSS
│  └─ icons.py                SVG 图标
├─ components/
│  ├─ image_viewer.py         QGraphicsView/Scene 图像画布
│  ├─ workflow_stepper.py     垂直建模步骤
│  └─ status_badge.py         状态显示
├─ models/
│  └─ template_state.py       单一 TemplatePageState
├─ services/
│  ├─ interfaces.py           Camera/Detection/Calibration/Repository 边界
│  ├─ template_repository.py  既有 TemplateLibrary 适配器
│  ├─ template_service.py     图像、分割、保存用例
│  └─ workers.py              QThread 工作对象
├─ dialogs/
│  └─ mask_editor.py          Qt Mask 添加/擦除编辑器
└─ pages/
   ├─ template_page.py        正式建立模板页面
   └─ placeholder_page.py     后续页面明确占位
```

## 已接入真实业务

- Unicode 图像读取：复用 `io.image_reader.read_image`。
- ROI：场景坐标始终等于图像像素坐标，不受缩放、平移与 DPI 影响。
- 自动分割：复用 `build_assisted_mask`，支持智能组合、纺织色度、暗色纹理、白色裁剪缝等既有方法。
- Mask 质量：复用 `analyze_template_mask`，显示覆盖率、主体占比和质量信息。
- Mask 修正：Qt 画笔支持添加、擦除、笔刷大小与清空。
- 匹配参数：阈值、360°角度范围、角度步长、尺度范围、尺度步长、NMS 和特征模式。
- 模板保存：复用 `TemplateModel.from_roi` 和 `TemplateLibrary.add_model`，不改变 JSON/PNG/Mask 格式。
- 自动分割在 QThread 中运行，不阻塞 Qt GUI 线程。

## 当前占位范围

- 模板检测页面
- 实时视觉页面
- 算法验证页面
- 标定工具与设置入口
- CameraService、DetectionService 和 CalibrationService 目前只有接口边界，未在本阶段接入。

## 验证结果

- 全量测试：`184 passed in 12.96s`。
- 真实参考图 ROI 分割：纺织色度方法得到有效 Mask，覆盖率 25.13%。
- 1440×900：控件无裁切，中文显示正常。
- 1920×1080 工作区：控件无异常拉伸；Windows 可用工作区截图为 1920×1061。
- Windows 125% DPI：中文、Viewer、Inspector 和底部状态栏均无裁切；物理截图为 1800×1061。

## 截图

- `docs/design/ui_redesign/screenshots/qt_template_page.png`
- `docs/design/ui_redesign/screenshots/qt_template_page_1920x1080.png`
- `docs/design/ui_redesign/screenshots/qt_template_page_125pct.png`

## 未修改的业务资产

本阶段没有改写：模板匹配算法、Mask 分割算法、模板文件格式、模板库清单格式和 Tkinter 正式入口。`industrial-segpose-ui` 仍指向 Tkinter；新增入口为 `industrial-segpose-ui-qt`。

## 已知问题与技术债

- 自动分割结果仍取决于工况；Qt 页面提供方法选择和 Mask 人工修正，但本阶段按要求没有调整算法本身。
- Mask 编辑器是第一版基础画笔，尚无撤销栈、多边形或边缘吸附。
- 保存并进入检测会进入明确的迁移占位页，真实 Qt 检测业务尚未接入。
- 现有 Tkinter 业务仍承担正式生产回退，Qt 入口尚未提升为默认入口。

## 下一阶段：实时视觉迁移

待本阶段 Design Freeze 后，再实现 CameraService 和 DetectionService 的具体适配器、相机线程、最新帧缓冲、检测 Worker、结果模型、生产指标及错误报告。下一阶段不应回头改变本报告冻结的整体视觉结构。
