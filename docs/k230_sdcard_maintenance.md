# K230 SD卡文件与空间维护

## 当前设备审计结论

Windows将当前K230显示为CanMV便携设备（WPD/MTP）和COM串口，而不是带盘符的磁盘。桌面UI不能用普通文件路径直接打开它，因此项目通过Windows便携设备接口把`/sdcard/industrial_vision/captures`同步到`build/k230_capture_cache/`。同步只读取设备并写入电脑缓存，不修改SD卡。

当前SD卡中的`kmodel/`约占274.5 MB，是最主要的可回收空间来源；其中人脸识别、视线、手部关键点、语音、人体分割和OCR等模型属于雅博/CanMV示例，并非本项目传统视觉检测所必需。但删除模型会让对应示例失效。

## 必须保留或不建议随意修改

- `boot.py`、根目录`main.py`：启动链和应用入口。替换错误会导致设备无法进入预期程序。
- `micropython`：板端运行时文件，不删除。
- `libs/`、`ybMain/`、`ybUtils/`、`configs/`、`res/`、`resources/`：雅博界面、驱动、字体和公共资源。删除可能导致原厂桌面或示例无法启动。
- `industrial_vision/`：本项目程序、模板库、标定、日志和采集图像。
- `System Volume Information/`：系统维护目录，保持不动。

## 可迁移或清理的内容

以下项目都应先完整复制到电脑备份，再由操作者确认删除：

1. `industrial_vision/captures/session_*/`和旧的`1970-01-01/`采集目录。图片同步到电脑并确认可打开后可从SD卡清理。
2. `industrial_vision/logs/`中的历史错误报告和轮转日志。建议保留当前日志及最近一次故障报告。
3. 已完成升级后遗留的`k230_runtime_update/`、`apply_runtime_update.py`及重复的松散运行时文件。只有确认`industrial_vision/k230_runtime/`和当前部署包可正常启动后才清理。
4. 不使用的`kmodel/*.kmodel`以及对应示例资源。当前工业视觉程序不依赖这些AI模型，理论上可回收约274.5 MB；若还需要原厂演示，应保留或将整个目录迁移到电脑备份。

## 推荐维护顺序

1. 先备份整个SD卡或至少备份`boot.py`、`main.py`、`industrial_vision/`和`kmodel/`。
2. 优先同步并清理旧采集图片与旧日志。
3. 空间仍不足时，再迁移不使用的`kmodel`示例模型。
4. 每次只清理一类文件，重启并验证相机、触摸UI、模板加载和检测后再继续。

本项目不会自动删除SD卡内容；清理操作必须由操作者明确确认。
