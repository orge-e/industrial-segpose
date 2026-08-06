"""Command-line entry point for planar calibration."""

from __future__ import annotations

import argparse

from .calibration.tool import run_calibration_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate planar camera calibration and error reports")
    parser.add_argument("--points", required=True, help="JSON file containing pixel_points and local_points_mm")
    parser.add_argument("--output", required=True, help="Output directory")
    arguments = parser.parse_args(argv)
    outputs = run_calibration_job(arguments.points, arguments.output)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
