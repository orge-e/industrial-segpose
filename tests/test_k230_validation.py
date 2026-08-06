import json
from pathlib import Path

import cv2
import numpy as np

from industrial_segpose.k230_validation import validate_image, validate_paths


def test_offline_k230_validation_detects_colored_workpiece_without_background_flood():
    image = np.full((240, 360, 3), 150, np.uint8)
    polygon = np.array([[70, 90], [285, 78], [315, 125], [245, 155], [100, 165], [55, 130]], np.int32)
    cv2.fillPoly(image, [polygon], (105, 105, 135))
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    (_, _), (first, second), _ = cv2.minAreaRect(contour)
    major, minor = max(first, second), min(first, second)
    samples = contour.reshape(-1, 2).astype(np.float32)
    _, vectors, _ = cv2.PCACompute2(samples, mean=None)
    reference_angle = float(np.degrees(np.arctan2(vectors[0][1], vectors[0][0])))
    area = float(cv2.contourArea(contour))
    template = {
        "template_id": "pink",
        "name": "Pink",
        "enabled": True,
        "width": 270,
        "height": 90,
        "mask_area_px": area,
        "reference_angle_deg": reference_angle,
        "shape_features": {
            "major_axis_length_px": major,
            "minor_axis_length_px": minor,
            "axis_fill_ratio": area / (major * minor),
        },
        "parameters": {"scale_min": 0.8, "scale_max": 1.2, "score_threshold": 0.7},
        "segmentation": {
            "mode": "color",
            "foreground_lab_median": [48.0, 12.0, 5.0],
            "lab_thresholds": [[20, 80, -5, 35, -10, 25]],
            "exposure_tolerance_l": 10,
            "chroma_tolerance": 3,
            "area_threshold": 200,
            "discriminative_channel": {
                "enabled": True,
                "channel": "a",
                "foreground_side": "above",
                "cutoff": 5.0,
            },
        },
    }

    detections, generated_masks = validate_image(image, [template])

    assert len(detections) == 1
    assert detections[0]["template_id"] == "pink"
    assert detections[0]["confidence"] > 0.9
    assert np.count_nonzero(generated_masks["pink"]) < image.shape[0] * image.shape[1] * 0.5


def test_validate_paths_keeps_duplicate_source_stems(tmp_path):
    bundle = tmp_path / "templates"
    bundle.mkdir()
    metadata = {"template_id": "t1", "name": "Part", "enabled": True, "segmentation": {"lab_thresholds": []}}
    (bundle / "t1.json").write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "template_library.json").write_text(
        json.dumps({"templates": [{"metadata_file": "t1.json"}]}), encoding="utf-8"
    )
    first = tmp_path / "a" / "IMG_000001.jpg"
    second = tmp_path / "b" / "IMG_000001.jpg"
    first.parent.mkdir()
    second.parent.mkdir()
    assert cv2.imwrite(str(first), np.zeros((32, 32, 3), np.uint8))
    assert cv2.imwrite(str(second), np.zeros((32, 32, 3), np.uint8))

    report_path = validate_paths(bundle / "template_library.json", [first, second], tmp_path / "report")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(report["images"]) == 2
    assert Path(report["images"][0]["preview"]).name.startswith("0001_")
    assert Path(report["images"][1]["preview"]).name.startswith("0002_")
