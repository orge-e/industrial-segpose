"""验证真实标注集评测流程及可视化对照图输出。"""

from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.evaluation import run_real_dataset_validation


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)


def test_real_dataset_validation_writes_metrics_and_visual_board(tmp_path):
    dataset = tmp_path / "中文真实标注集"
    image = np.zeros((180, 260, 3), np.uint8)
    labels = np.zeros((180, 260), np.uint16)
    cv2.rectangle(image, (28, 38), (105, 132), (235, 235, 235), -1)
    cv2.ellipse(image, (190, 90), (35, 52), 18, 0, 360, (220, 220, 220), -1)
    cv2.rectangle(labels, (28, 38), (105, 132), 1, -1)
    cv2.ellipse(labels, (190, 90), (35, 52), 18, 0, 360, 2, -1)
    _write_png(dataset / "images" / "现场样图一.png", image)
    _write_png(dataset / "label_maps" / "现场样图一.labels.png", labels)

    result = run_real_dataset_validation(
        dataset,
        tmp_path / "outputs",
        algorithms=["threshold"],
        minimum_iou=0.20,
        max_visual_images=4,
        run_name="fixed_run",
    )

    assert result.dataset_id > 0
    assert set(result.summaries) == {"threshold"}
    assert result.summaries["threshold"].image_count == 1
    assert result.summaries["threshold"].mean_axis_angle_error_deg is not None
    assert result.comparison.image_count == 1
    assert result.comparison.output_path.is_file()
    assert result.comparison.manifest_path.is_file()
    assert (result.run_dir / "metrics" / "summary.csv").is_file()
    assert (result.run_dir / "metrics" / "summary.md").is_file()
    assert (result.run_dir / "validation.db").is_file()
