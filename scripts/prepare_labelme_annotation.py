"""Create project-assisted Labelme annotations and convert reviewed JSON to Label Maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_segpose.datasets.labelme_bridge import (
    create_fluorescent_labelme_preannotation,
    labelme_json_to_instance_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prelabel = subparsers.add_parser("prelabel", help="auto-create editable Labelme polygons")
    prelabel.add_argument("--image", required=True)
    prelabel.add_argument("--output", required=True)
    prelabel.add_argument("--label", default="workpiece")
    convert = subparsers.add_parser("convert", help="convert reviewed Labelme JSON")
    convert.add_argument("--json", required=True)
    convert.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "prelabel":
        result = create_fluorescent_labelme_preannotation(
            args.image, args.output, label=args.label
        )
        print(f"objects: {result.object_count}")
        print(f"labelme json: {result.json_path}")
        print(f"preview: {result.overlay_path}")
    else:
        result = labelme_json_to_instance_map(args.json, args.output)
        print(f"objects: {result.object_count}")
        print(f"label map: {result.label_map_path}")
        print(f"preview: {result.overlay_path}")
        print(f"metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
