"""算法评估：组织多种工况配置的离线合成测试套件。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .runner import ALGORITHM_NAMES, BenchmarkRunner
from ..datasets import SyntheticConfig, SyntheticDatasetGenerator
from ..experiment_db import ExperimentDatabase


PRODUCTION_PROFILES = (
    "clean",
    "rotation",
    "scale_variation",
    "color_variation",
    "lighting",
    "noise",
    "dense_separated",
    "edge_partial",
    "production_mixed",
)
STRESS_PROFILES = ("touching", "mixed")
DEFAULT_PROFILES = PRODUCTION_PROFILES


def run_synthetic_suite(
    project_root: str | Path,
    *,
    images_per_profile: int = 8,
    random_seed: int = 20260821,
    database_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    profiles: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    if images_per_profile < 1:
        raise ValueError("images_per_profile must be positive")
    root = Path(project_root)
    selected_profiles = tuple(profiles or DEFAULT_PROFILES)
    if not selected_profiles:
        raise ValueError("At least one benchmark profile is required")
    database_file = Path(database_path) if database_path else root / "data" / "benchmark" / "segpose_production_benchmark.db"
    report_root = Path(output_dir) if output_dir else root / "reports" / "offline_benchmark_production"
    dataset_root = root / "datasets" / "synthetic" / "production_nonoverlap"
    if database_file.exists() and not overwrite:
        raise FileExistsError(
            f"Benchmark database already exists: {database_file}. Use overwrite=True for a clean rerun."
        )
    database = ExperimentDatabase(database_file)
    database.rebuild()
    all_details: dict[str, dict[str, list[dict[str, Any]]]] = {}
    profile_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for profile_index, profile in enumerate(selected_profiles):
        profile_dataset_root = dataset_root / profile
        min_objects, max_objects = (3, 4) if profile == "dense_separated" else (1, 4)
        min_scale, max_scale = {
            "dense_separated": (0.68, 0.82),
            "scale_variation": (0.55, 1.40),
            "production_mixed": (0.70, 1.10),
        }.get(profile, (0.75, 1.25))
        generator = SyntheticDatasetGenerator(
            SyntheticConfig(
                image_width=512,
                image_height=384,
                image_count=images_per_profile,
                min_objects=min_objects,
                max_objects=max_objects,
                min_scale=min_scale,
                max_scale=max_scale,
                profile=profile,
                random_seed=random_seed + profile_index * 1000,
            )
        )
        generated = generator.generate(
            profile_dataset_root,
            database,
            dataset_name=f"phase5_{profile}",
            dataset_version="initial",
        )
        profile_output = report_root / profile
        summaries = BenchmarkRunner(database, profile_output).run(generated.dataset_id)
        profile_summaries[profile] = {
            name: _jsonable_summary(summary) for name, summary in summaries.items()
        }
        all_details[profile] = json.loads(
            (profile_output / "details.json").read_text(encoding="utf-8")
        )
    combined = _combine_details(all_details, selected_profiles)
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "profiles": list(selected_profiles),
            "images_per_profile": images_per_profile,
            "random_seed": random_seed,
            "database_path": str(database_file),
        },
        "overall": combined,
        "by_profile": profile_summaries,
    }
    (report_root / "suite_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(report_root / "suite_summary.csv", combined)
    _write_markdown(report_root / "suite_summary.md", payload)
    return combined


def _combine_details(details_by_profile, profiles):
    combined: dict[str, dict[str, Any]] = {}
    for algorithm in ALGORITHM_NAMES:
        rows = [row for profile in profiles for row in details_by_profile[profile][algorithm]]
        matches = [match for row in rows for match in row["matches"]]
        gt_count = sum(row["gt_count"] for row in rows)
        prediction_count = sum(row["prediction_count"] for row in rows)
        tp = len(matches)
        fp = sum(row["false_positive"] for row in rows)
        fn = sum(row["false_negative"] for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        times = [float(row["processing_time_ms"]) for row in rows]
        per_profile = {}
        for profile in profiles:
            selected = details_by_profile[profile][algorithm]
            profile_tp = sum(len(row["matches"]) for row in selected)
            profile_fp = sum(row["false_positive"] for row in selected)
            profile_fn = sum(row["false_negative"] for row in selected)
            profile_precision = profile_tp / max(profile_tp + profile_fp, 1)
            profile_recall = profile_tp / max(profile_tp + profile_fn, 1)
            per_profile[profile] = {
                "f1": 2 * profile_precision * profile_recall / max(profile_precision + profile_recall, 1e-12),
                "mean_iou": sum(match["iou"] for row in selected for match in row["matches"]) / max(sum(row["gt_count"] for row in selected), 1),
                "exact_count_accuracy": float(np.mean([row["exact_count_accuracy"] for row in selected])),
                "failure_rate": float(np.mean([row["failure"] for row in selected])),
            }
        worst_profile = min(per_profile, key=lambda key: (per_profile[key]["f1"], per_profile[key]["mean_iou"]))
        mean_time = float(np.mean(times)) if times else 0.0
        combined[algorithm] = {
            "algorithm": algorithm,
            "image_count": len(rows),
            "ground_truth_count": gt_count,
            "prediction_count": prediction_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "mean_iou": sum(item["iou"] for item in matches) / max(gt_count, 1),
            "mean_dice": sum(item["dice"] for item in matches) / max(gt_count, 1),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "mean_count_error": float(np.mean([row["count_error"] for row in rows])),
            "count_accuracy": float(np.mean([row["count_accuracy"] for row in rows])),
            "exact_count_accuracy": float(np.mean([row["exact_count_accuracy"] for row in rows])),
            "mean_center_error_px": _optional_mean([item["center_error_px"] for item in matches]),
            "mean_axis_angle_error_deg": _optional_mean([item["axis_angle_error_deg"] for item in matches]),
            "mean_directed_angle_error_deg": _optional_mean([item["directed_angle_error_deg"] for item in matches]),
            "mean_processing_time_ms": mean_time,
            "p95_processing_time_ms": float(np.percentile(times, 95)) if times else 0.0,
            "fps": 1000.0 / mean_time if mean_time else 0.0,
            "failure_rate": float(np.mean([row["failure"] for row in rows])),
            "worst_profile": worst_profile,
            "by_profile": per_profile,
        }
    return combined


def _write_csv(path: Path, combined) -> None:
    fields = [key for key in next(iter(combined.values())) if key != "by_profile"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in combined.values():
            writer.writerow({key: value for key, value in row.items() if key != "by_profile"})


def _write_markdown(path: Path, payload) -> None:
    rows = [
        "# Production Non-overlap Synthetic Benchmark",
        "",
        f"Profiles: {', '.join(payload['config']['profiles'])}",
        f"Images per profile: {payload['config']['images_per_profile']}",
        "",
        "| Algorithm | IoU | Dice | Precision | Recall | F1 | Exact Count | Center px | Axis deg | Mean ms | P95 ms | Worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["overall"].values():
        rows.append(
            f"| {item['algorithm']} | {item['mean_iou']:.3f} | {item['mean_dice']:.3f} | "
            f"{item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | "
            f"{item['exact_count_accuracy']:.3f} | {_format(item['mean_center_error_px'])} | "
            f"{_format(item['mean_axis_angle_error_deg'])} | {item['mean_processing_time_ms']:.2f} | "
            f"{item['p95_processing_time_ms']:.2f} | {item['worst_profile']} |"
        )
    rows.extend(["", "## Profiles", ""])
    for algorithm, item in payload["overall"].items():
        rows.append(f"### {algorithm}")
        rows.append("")
        rows.append("| Profile | F1 | IoU | Exact Count | Failure Rate |")
        rows.append("|---|---:|---:|---:|---:|")
        for profile, metrics in item["by_profile"].items():
            rows.append(
                f"| {profile} | {metrics['f1']:.3f} | {metrics['mean_iou']:.3f} | "
                f"{metrics['exact_count_accuracy']:.3f} | {metrics['failure_rate']:.3f} |"
            )
        rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def _jsonable_summary(summary) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(summary)


def _optional_mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _format(value) -> str:
    return "N/A" if value is None else f"{value:.3f}"
