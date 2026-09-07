"""Run the database-driven benchmark from an existing registered dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from industrial_segpose.evaluation.runner import ALGORITHM_NAMES, BenchmarkRunner
from industrial_segpose.experiment_db import ExperimentDatabase, default_database_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id", type=int)
    parser.add_argument("--database", default=str(default_database_path()))
    parser.add_argument("--output", default="reports/algorithm_benchmark")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=ALGORITHM_NAMES)
    args = parser.parse_args()
    summaries = BenchmarkRunner(ExperimentDatabase(args.database), args.output).run(
        args.dataset_id, args.algorithms
    )
    for name, summary in summaries.items():
        print(name, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
