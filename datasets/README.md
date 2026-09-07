# Algorithm datasets

- `synthetic/`: generated scenes and complete ground truth.
- `public/`: imported public datasets.
- `historical/`: normalized historical project images.
- `real/`: reserved for future on-site captures.
- `annotation_images/`: original images waiting for algorithm pre-annotation.
- `annotation_workbench/`: Labelme JSON, copied source images and pre-label previews.

Large generated images and database files are local experiment artifacts and
should not be committed.

## 真实标注集格式

经人工审查的现场数据放在一个独立目录中：

```text
real/<批次名称>/
├─ images/                  原始图片
└─ label_maps/              实例标签图
   └─ <原图文件名>.labels.png
```

标签图中 `0` 表示背景，每个工件使用不同的正整数编号。项目的 Labelme
转换工具会直接生成这种格式。完成至少一批审查后可运行：

```powershell
industrial-segpose validate-real --dataset datasets/real/<批次名称>
```

输出包含四种传统算法的数值指标和一张多图、多算法实际效果对照图。
