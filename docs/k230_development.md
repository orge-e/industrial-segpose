# K230 脱机运行开发说明

## 当前交付边界

桌面端负责模板编辑、批量验证和部署包生成；当前K230生产配置只负责加载已验证模板，并独立完成候选分割、多模板分类、中心/角度计算、跨帧跟踪、计数和结果显示。板端模板建立入口与代码暂时保留，但默认禁用。

已经按雅博 SD 卡 v1.4.3 中的 `libs.PipeLine`、GC2093 通道配置、ST7701 显示和 `TOUCH(0)` 接口完成适配。当前入口是安全的 `dry-run`：只在屏幕和控制台输出结果，不控制吸盘。实际帧率、阈值和角度精度仍需连接真实模块后标定。

## 目录

- `industrial_segpose/k230_export.py`：桌面模板库导出器。
- `industrial_segpose/k230_deploy.py`：生成 SD 卡目录结构的本地部署工具。
- `shared_protocol/`：电脑、K230 和控制器共用的 JSON Lines 消息协议。
- `k230_runtime/`：硬件适配、模板建立、检测、跟踪、触摸 UI 和运行循环。
- `k230_runtime/device_config.example.json`：设备配置示例。

## 电脑端验证

在项目根目录执行：

```powershell
python -m pytest
python -m k230_runtime.selftest
python -m industrial_segpose.k230_deploy --project . --output build/k230_sdcard --overwrite
```

本地部署包位于 `build/k230_sdcard/industrial_vision/`，默认不会提交 Git。连接模块后，将整个 `industrial_vision` 目录复制到 `/sdcard/`，不要覆盖根目录原有 `main.py`：

```text
industrial_vision/
  device_config.json
  templates/
    template_library.json
    <template-id>/
      metadata.json
      template_gray.png
      template_mask.png
      template_edge.png
      template_pose.pgm
  k230_runtime/
  shared_protocol/
```

在 CanMV 中手动运行 `/sdcard/industrial_vision/main.py`。设备界面名称默认为 `FlexPose Vision｜柔性工件定位系统`，可在 `device_config.json` 的 `ui` 节点修改。检测页底部提供放大、缩小、适应画面、拍摄原图和计数清零按钮。当前 `template_authoring.enabled=false`：模板库页面仍可查看模板和启用/停用模板，但不会启动板端拍摄、冻结或分割流程。模板应在电脑端建立、验证并随部署包导入。

检测框标签使用`TYPE:<模板名称>`显示工件类别，并显示目标流水号、状态、中心坐标、角度和置信度。画面右侧固定的`DETECTION RESULT`面板逐项显示`TYPE`模板类型、`POS X/Y`图像中心坐标、`ANGLE`旋转角度和`SCORE`置信度，避免随目标移动的标签被边界裁切。内部模板 UUID 不在屏幕上显示；模板名称缺失时显示`Unnamed_Template`，避免把内部哈希误认为类别。完整模板 ID 仍保留在控制台心跳的`current_objects`和`pick_target`协议中。`NOW`是当前画面识别数量，`LINE TOTAL`是工件通过计数线后的累计数量。

“拍摄原图”保存未经标注的相机帧，目录按日期自动归档：

```text
/sdcard/industrial_vision/captures/session_NNNN/IMG_NNNNNN.jpg
```

建议每类工件至少采集正曝光、欠曝光、过曝光、不同位置和不同角度的图像，用于桌面端回放标定。板端检测使用曝光容差 LAB 分割、旋转不变主轴尺寸、面积尺度一致性和模板独立阈值；输出角度为适合吸盘末端旋转的 `[-90°, 90°)` 主轴角。

## 开机自动进入软件

真机手动运行和稳定性验证完成后，再执行：

```text
/sdcard/industrial_vision/install_autostart.py
```

安装器会先把原 `/sdcard/main.py` 保存为 `/sdcard/industrial_vision/recovery/original_main.py`，然后安装安全启动器。下次开机自动进入“建立模板 / 图像检测”主页。

- 开机时按住板载按键：跳过本软件并运行原程序。
- 创建 `/sdcard/industrial_vision/disable_autostart`：持续禁用自动启动。
- 执行 `/sdcard/industrial_vision/restore_original.py`：恢复原根启动程序。
- 启动异常会写入 `/sdcard/industrial_vision/logs/startup_error.log`，然后尝试回退原程序。

## 上板前必须完成

1. 在 K230 上运行本地部署包，验证屏幕叠加层、触摸坐标和内存占用。
2. 使用粉色件和黑色布片分别标定 LAB 阈值、中心、角度、置信度和耗时。
3. 用传送带视频验证跟踪距离、计数线位置和重复计数抑制。
4. 核对 UART 引脚、电平、波特率以及控制器接收协议。
5. 完成相机内参/畸变和像素到工作台坐标的标定。
6. 增加 PLC 或吸盘控制器握手、超时、急停和失联保护。
7. 先以“只输出、不动作”模式跑传送带数据，再逐级开放执行器。

## 安全原则

视觉结果只有在图像质量合格、识别状态明确、置信度达标、目标仍处于有效抓取窗口且控制器完成握手时才允许进入执行链路。待确认目标、过期目标和重复轨迹不得触发抓取。

## 推荐的生产模板流程

当前采用“电脑端精细建库与验证、K230端采集与执行”：

1. K230只负责拍摄现场样本，原图保存到`captures/session_NNNN/`。
2. 将整个会话目录复制到电脑，在桌面UI中修订mask、名称、吸取点和匹配参数。
3. 在电脑端导出带LAB前景/背景统计、区分通道和形状特征的轻量模板包。
4. 先用现场原图离线验证，再把通过验证的整个`templates/`目录复制到K230。
5. 当前阶段不在K230端建立模板；所有模板必须在电脑端查看Mask、完成批量回放验证，再导出到模块模板库。

### 桌面图形化流程

启动`python -m industrial_segpose.template_ui`，在“建立模板”页打开“K230采集图像 / 模板工作台”：

1. 目录框可以选择SD卡根目录、`industrial_vision`、`captures`或复制到电脑的单个会话目录。
2. “设为基准图并自动分割”把所选原图送入电脑端自动定位、初始Mask和画笔修正流程。
3. 保存模板后，多选不同曝光、位置和角度的图像，点击“批量验证所选图像”。报告位于`reports/k230_workbench/<时间>/`。
4. 点击“生成完整K230部署包”，程序检查至少存在一个有效且启用的模板，然后生成`build/k230_sdcard/industrial_vision/`。

工作台直接读取源照片但不会复制它们。最终部署包只包含运行时、设备配置和压缩后的模板资产，避免占用模块约511 MB的存储空间。

生成并验证模板包：

```powershell
python -m industrial_segpose.k230_export --library templates --output build/k230_templates --overwrite
python -m industrial_segpose.k230_validation --bundle build/k230_templates/template_library.json --images <采集图片目录> --output reports/k230_validation
```

`validation_report.json`记录每张图的类别、置信度、中心、角度和尺度，同时生成轮廓叠加预览。只有覆盖实际曝光、位置、角度和材料状态的样本均通过后，才更新模块模板包。
