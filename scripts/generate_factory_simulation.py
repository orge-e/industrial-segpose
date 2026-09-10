"""Create factory-photo augmentations or labelled pickup-state simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_segpose.datasets.factory_simulation import (
    PhotoAugmentationConfig,
    PickupSimulationConfig,
    VisualTwinConfig,
    export_workpiece_assets,
    generate_photo_augmentations,
    generate_pickup_scenes,
    generate_visual_twin_scenes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    augment = subparsers.add_parser("augment", help="camera and lighting variants")
    augment.add_argument("--image", required=True)
    augment.add_argument("--label-map")
    augment.add_argument("--output", required=True)
    augment.add_argument("--count", type=int, default=12)
    augment.add_argument("--seed", type=int, default=20260824)

    pickup = subparsers.add_parser("pickup", help="simulate random removed workpieces")
    pickup.add_argument("--image", required=True)
    pickup.add_argument("--label-map", required=True)
    pickup.add_argument("--output", required=True)
    pickup.add_argument("--count", type=int, default=12)
    pickup.add_argument("--seed", type=int, default=20260824)
    pickup.add_argument("--removal-ratios", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.8])

    assets = subparsers.add_parser("assets", help="export reviewed instances as digital assets")
    assets.add_argument("--image", required=True)
    assets.add_argument("--label-map", required=True)
    assets.add_argument("--asset-metadata")
    assets.add_argument("--output", required=True)
    assets.add_argument("--padding", type=int, default=6)

    twin = subparsers.add_parser("compose", help="compose non-overlapping visual-twin scenes")
    twin.add_argument("--image", required=True, help="reviewed source image")
    twin.add_argument("--label-map", required=True, help="reviewed instance label map")
    twin.add_argument("--background", help="optional clean background image")
    twin.add_argument("--asset-metadata", help="optional JSON with class and reference angles")
    twin.add_argument("--output", required=True)
    twin.add_argument("--count", type=int, default=100)
    twin.add_argument("--seed", type=int, default=20260831)
    twin.add_argument("--min-objects", type=int, default=1)
    twin.add_argument("--max-objects", type=int, default=12)
    twin.add_argument("--min-scale", type=float, default=0.85)
    twin.add_argument("--max-scale", type=float, default=1.15)
    twin.add_argument("--min-gap", type=int, default=6)
    twin.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    twin.add_argument("--profile", choices=["clean", "balanced", "harsh"], default="balanced")

    args = parser.parse_args()
    if args.mode == "augment":
        result = generate_photo_augmentations(
            args.image,
            args.output,
            label_map_path=args.label_map,
            config=PhotoAugmentationConfig(variant_count=args.count, random_seed=args.seed),
        )
    elif args.mode == "pickup":
        result = generate_pickup_scenes(
            args.image,
            args.label_map,
            args.output,
            config=PickupSimulationConfig(
                scene_count=args.count,
                random_seed=args.seed,
                removal_ratios=tuple(args.removal_ratios),
            ),
        )
    elif args.mode == "assets":
        result = export_workpiece_assets(
            args.image,
            args.label_map,
            args.output,
            asset_metadata_path=args.asset_metadata,
            padding_px=args.padding,
        )
    else:
        result = generate_visual_twin_scenes(
            args.image,
            args.label_map,
            args.output,
            background_image=args.background,
            asset_metadata_path=args.asset_metadata,
            config=VisualTwinConfig(
                scene_count=args.count,
                random_seed=args.seed,
                min_objects=args.min_objects,
                max_objects=args.max_objects,
                min_scale=args.min_scale,
                max_scale=args.max_scale,
                min_gap_px=args.min_gap,
                profile=args.profile,
                placement_roi=None if args.roi is None else tuple(args.roi),
            ),
        )
    print(f"output: {result.output_dir}")
    if hasattr(result, "asset_count"):
        print(f"assets: {result.asset_count}")
    else:
        print(f"images: {result.image_count}")
        print(f"source instances: {result.source_instance_count}")
    print(f"manifest: {result.manifest_path}")
    print(f"preview: {result.preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
