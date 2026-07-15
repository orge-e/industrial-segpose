import json
from pathlib import Path
import cv2
import numpy as np
from industrial_segpose.cli import main


def test_cli_json_csv_masks(tmp_path):
    image = np.zeros((120, 180, 3), np.uint8); cv2.rectangle(image, (40, 40), (130, 75), (255, 255, 255), -1)
    image_path = tmp_path / "sample.png"; cv2.imencode(".png", image)[1].tofile(image_path)
    output = tmp_path / "outputs"; assert main(["infer", "--input", str(image_path), "--config", "configs/default.yaml", "--output", str(output)]) == 0
    run = next(output.glob("run_*")); payload = json.loads(next((run / "json").glob("*.json")).read_text(encoding="utf-8"))
    assert payload["object_count"] == 1; assert next((run / "csv").glob("*.csv")).is_file(); assert len(list((run / "masks").glob("*.png"))) == 1; assert (run / "run_summary.json").is_file()


def test_cli_missing_input_returns_two(tmp_path):
    assert main(["infer", "--input", str(tmp_path / "missing.png"), "--config", "configs/default.yaml"]) == 2


def test_recursive_duplicate_names_do_not_overwrite(tmp_path):
    image = np.zeros((80, 120, 3), np.uint8); cv2.circle(image, (60, 40), 20, (255, 255, 255), -1)
    source = tmp_path / "inputs"
    for folder in (source / "a", source / "b"):
        folder.mkdir(parents=True); cv2.imencode(".png", image)[1].tofile(folder / "same.png")
    output = tmp_path / "out"; assert main(["infer", "--input", str(source), "--recursive", "--config", "configs/default.yaml", "--output", str(output)]) == 0
    run = next(output.glob("run_*")); assert len(list((run / "json").glob("*.json"))) == 2; assert len(list((run / "annotated").glob("*.png"))) == 2
