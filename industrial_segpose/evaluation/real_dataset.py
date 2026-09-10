"""真实标注集评测：导入人工审查结果并生成指标和视觉对照图。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..datasets import MaskDirectoryAdapter
from ..experiment_db import ExperimentDatabase
from .runner import REAL_ALGORITHM_NAMES, AlgorithmBenchmarkSummary, BenchmarkRunner
from .visualizer import ComparisonBoardResult, generate_mask_directory_comparison_board


@dataclass(frozen=True)
class RealDatasetValidationResult:
    run_dir: Path
    dataset_id: int
    summaries: dict[str, AlgorithmBenchmarkSummary]
    comparison: ComparisonBoardResult


def run_real_dataset_validation(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    algorithms: Iterable[str] = REAL_ALGORITHM_NAMES,
    minimum_iou: float = 0.30,
    max_visual_images: int = 12,
    run_name: str | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> RealDatasetValidationResult:
    """Run repeatable metrics and visual QA for a reviewed mask directory."""
    source = Path(dataset_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {source}")
    algorithm_names = tuple(algorithms)
    if not algorithm_names:
        raise ValueError("At least one algorithm is required")
    stamp = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    database = ExperimentDatabase(run_dir / "validation.db")
    dataset_id = MaskDirectoryAdapter(source_type="real").import_dataset(
        source, database, dataset_name=f"real_validation_{stamp}"
    )
    report_dir = run_dir / "metrics"
    overrides = {
        name: {"input": {"roi": list(roi) if roi else None}}
        for name in algorithm_names
    }
    summaries = BenchmarkRunner(database, report_dir, minimum_iou=minimum_iou).run(
        dataset_id, algorithm_names, overrides
    )
    comparison = generate_mask_directory_comparison_board(
        source,
        run_dir / "visual_comparison.jpg",
        max_images=max_visual_images,
        algorithms=algorithm_names,
        minimum_iou=minimum_iou,
        config_overrides=overrides,
    )
    return RealDatasetValidationResult(run_dir, dataset_id, summaries, comparison)
