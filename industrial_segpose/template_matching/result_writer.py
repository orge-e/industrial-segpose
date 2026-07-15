"""JSON, CSV, and annotated-image export for multi-template recognition."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from ..io.image_reader import write_image
from .multi_matcher import MultiTemplateResult


def write_multi_template_result(
    output_root: str | Path,
    source_image: str | Path,
    annotated_image: np.ndarray,
    result: MultiTemplateResult,
    run_name: str | None = None,
) -> Path:
    root = Path(output_root)
    name = run_name or f"template_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_image(run_dir / "annotated.png", annotated_image)
    payload = {"image": str(source_image), **result.to_dict()}
    (run_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object_id", "classification_status", "template_id", "template_name", "center_x", "center_y", "angle_deg", "score", "normalized_score", "scale", "candidate_templates"])
        for item in result.objects:
            candidates = json.dumps(
                [{"template_id": candidate.template_id, "template_name": candidate.template_name, "score": candidate.score, "normalized_score": candidate.normalized_score} for candidate in item.candidate_templates],
                ensure_ascii=False,
            )
            writer.writerow([item.object_id, item.classification_status, item.template_id or "", item.template_name, item.center_x, item.center_y, item.angle_deg, item.score, item.normalized_score, item.scale, candidates])
    return run_dir

