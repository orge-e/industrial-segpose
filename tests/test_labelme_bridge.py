import json
from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.datasets.labelme_bridge import labelme_json_to_instance_map
from industrial_segpose.io.image_reader import write_image


def test_labelme_polygons_convert_to_separate_instances(tmp_path: Path):
    image = np.full((100, 160, 3), 45, np.uint8)
    image_path = tmp_path / "现场原图.jpg"
    write_image(image_path, image)
    annotation = {
        "version": "5.0.0",
        "flags": {},
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 160,
        "shapes": [
            {"label": "workpiece", "points": [[10, 10], [60, 10], [60, 60], [10, 60]], "shape_type": "polygon"},
            {"label": "workpiece", "points": [[90, 20], [145, 20], [145, 75], [90, 75]], "shape_type": "polygon"},
        ],
    }
    json_path = tmp_path / "现场原图.json"
    json_path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    result = labelme_json_to_instance_map(json_path, tmp_path / "converted")
    label_map = cv2.imdecode(np.fromfile(result.label_map_path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert result.object_count == 2
    assert set(np.unique(label_map)) == {0, 1, 2}
    assert result.overlay_path.is_file()
    assert result.metadata_path.is_file()
