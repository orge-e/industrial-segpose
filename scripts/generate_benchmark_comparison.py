"""Generate a reusable side-by-side visual comparison of segmentation algorithms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_segpose.evaluation.runner import ALGORITHM_NAMES
from industrial_segpose.evaluation.visualizer import generate_comparison_board


DEFAULT_PROFILES = ("rotation", "lighting", "dense_separated", "production_mixed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render multiple images and algorithms into one comparison board."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "datasets" / "synthetic" / "production_nonoverlap"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "offline_benchmark_production" / "visual_comparison.png"),
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--images-per-profile", type=int, default=1)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    args = parser.parse_args()

    result = generate_comparison_board(
        args.dataset_root,
        args.output,
        profiles=args.profiles,
        images_per_profile=args.images_per_profile,
        algorithms=args.algorithms,
        tile_width=args.tile_width,
        minimum_iou=args.minimum_iou,
    )
    print(f"comparison image: {result.output_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"images: {result.image_count}; algorithms: {', '.join(result.algorithms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
