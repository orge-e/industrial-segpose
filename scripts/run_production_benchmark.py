"""Run the non-overlapping production synthetic benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_segpose.evaluation.suite import PRODUCTION_PROFILES, run_synthetic_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--images-per-profile", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summaries = run_synthetic_suite(
        args.project_root,
        images_per_profile=args.images_per_profile,
        random_seed=args.seed,
        overwrite=args.overwrite,
        profiles=PRODUCTION_PROFILES,
    )
    for algorithm, summary in summaries.items():
        print(
            f"{algorithm}: F1={summary['f1']:.3f}, IoU={summary['mean_iou']:.3f}, "
            f"exact_count={summary['exact_count_accuracy']:.3f}, "
            f"mean_ms={summary['mean_processing_time_ms']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
