"""Industrial SegPose command-line interface."""

import argparse
import logging
from pathlib import Path
from time import perf_counter
from .config import load_config, validate_config
from .io.image_reader import discover_images, read_image
from .io.result_writer import ResultWriter
from .pipeline import SegPosePipeline
from .types import ImageResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-segpose")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="Run instance segmentation and pose measurement")
    infer.add_argument("--input", required=True); infer.add_argument("--config", required=True); infer.add_argument("--output", default="outputs")
    infer.add_argument("--backend", choices=["threshold", "adaptive_threshold", "color_range", "watershed"])
    infer.add_argument("--recursive", action="store_true")
    for name in ("debug", "masks", "json", "csv"):
        infer.add_argument(f"--save-{name}", action=argparse.BooleanOptionalAction, default=None)
    return parser


def run_infer(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.backend: config["segmentation"]["backend"] = args.backend
    for name in ("debug", "masks", "json", "csv"):
        value = getattr(args, f"save_{name}")
        if value is not None: config["output"][f"save_{name}"] = value
    validate_config(config)
    paths = discover_images(args.input, args.recursive)
    if not paths: raise ValueError(f"No supported images found under: {args.input}")
    pipeline, writer = SegPosePipeline(config), ResultWriter(args.output, config)
    results: list[ImageResult] = []; started = perf_counter()
    input_root = Path(args.input)
    for index, path in enumerate(paths, 1):
        try:
            image_name = path.name if input_root.is_file() else path.relative_to(input_root).as_posix()
            image = read_image(path); result = pipeline.process(image, image_name)
            writer.write_image_result(pipeline.last_work_image, result, pipeline.last_segmentation.label_map, pipeline.last_segmentation.debug_images)
            logging.info("[%d/%d] %s: %d objects (%.1f ms)", index, len(paths), path.name, result.object_count, result.processing_time_ms)
        except Exception as exc:
            result = ImageResult(path.name, 0, 0, 0, [], 0.0, config["segmentation"]["backend"], False, str(exc))
            logging.error("[%d/%d] %s failed: %s", index, len(paths), path.name, exc)
        results.append(result)
    elapsed = (perf_counter() - started) * 1000.0; writer.finalize(results, elapsed)
    logging.info("Done: %d succeeded, %d failed, %.1f ms; output: %s", sum(r.success for r in results), sum(not r.success for r in results), elapsed, writer.run_dir)
    return 1 if any(not r.success for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try: return run_infer(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logging.error("%s", exc); return 2
