import json

import cv2
import numpy as np

from industrial_segpose.template_matching.audit import (
    ParameterSearchConfig,
    audit_template_library,
    search_template_parameters,
    write_parameter_search_report,
    write_template_audit_report,
)
from industrial_segpose.template_matching import MatchParameters, TemplateLibrary, TemplateModel


def _dark_library(root):
    image = np.full((70, 100, 3), 205, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    points = np.array([[12, 14], [82, 10], [90, 48], [55, 60], [15, 53]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    image[mask > 0] = (24, 24, 24)
    cv2.line(image, (25, 22), (70, 45), (48, 48, 48), 2)
    library = TemplateLibrary(root)
    entry = library.add_model(
        TemplateModel("暗色裁片", image, mask=mask),
        MatchParameters(
            score_threshold=0.65,
            angle_min=-20,
            angle_max=20,
            angle_step=4,
            scale_min=0.9,
            scale_max=1.1,
            scale_step=0.1,
            feature_mode="pose_tolerant",
            use_edges=False,
            max_results=10,
        ),
    )
    return library, entry


def test_audit_detects_feature_mode_mismatch_and_writes_report(tmp_path):
    library, entry = _dark_library(tmp_path / "templates")
    audits = audit_template_library(library)
    assert len(audits) == 1
    assert audits[0].template_id == entry.template_id
    assert audits[0].recommended_mode == "dark_textile"
    assert audits[0].current_mode == "pose_tolerant"
    assert audits[0].issues
    report = write_template_audit_report(tmp_path / "audit.json", audits)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["template_count"] == 1
    assert payload["templates"][0]["recommended_mode"] == "dark_textile"


def test_parameter_search_is_bounded_reproducible_and_does_not_mutate_library(tmp_path):
    library, entry = _dark_library(tmp_path / "templates")
    original = entry.parameters
    config = ParameterSearchConfig(
        scene_count=1,
        seed=9,
        profile="clean",
        max_trials=2,
        score_thresholds=(0.50, 0.65),
        center_tolerance_px=35,
        angle_tolerance_deg=12,
    )
    first = search_template_parameters(library, entry.template_id, config)
    second = search_template_parameters(library, entry.template_id, config)
    assert first.best_parameters == second.best_parameters
    assert first.best_trial.f1 == second.best_trial.f1
    assert first.best_trial.mean_center_error_px == second.best_trial.mean_center_error_px
    assert first.best_trial.mean_angle_error_deg == second.best_trial.mean_angle_error_deg
    assert first.evaluated_configurations <= 2
    assert len(first.trials) <= 2 * 2
    assert library.get(entry.template_id).parameters == original
    report = write_parameter_search_report(tmp_path / "search.json", first)
    assert json.loads(report.read_text(encoding="utf-8"))["template_name"] == "暗色裁片"
