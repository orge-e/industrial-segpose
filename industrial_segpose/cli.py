"""Industrial SegPose command-line interface."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from .config import load_config, validate_config
from .io.image_reader import discover_images, read_image, write_image
from .io.result_writer import ResultWriter
from .detectors.fluorescent import FluorescentTextileDetector, FluorescentTextileParameters
from .pipeline import SegPosePipeline
from .template_matching import (
    WhiteTextileParameters,
    draw_multi_template_matches,
    save_white_textile_diagnostics,
    segment_white_textile,
)
from .types import ImageResult
from .evaluation.benchmark import EvaluationThresholds, SyntheticStressConfig, run_synthetic_benchmark
from .evaluation.real_dataset import run_real_dataset_validation
from .evaluation.runner import ALGORITHM_NAMES, REAL_ALGORITHM_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-segpose")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="Run instance segmentation and pose measurement")
    infer.add_argument("--input", required=True); infer.add_argument("--config", required=True); infer.add_argument("--output", default="outputs")
    infer.add_argument("--backend", choices=["threshold", "adaptive_threshold", "color_range", "watershed"])
    infer.add_argument("--recursive", action="store_true")
    for name in ("debug", "masks", "json", "csv"):
        infer.add_argument(f"--save-{name}", action=argparse.BooleanOptionalAction, default=None)
    white = sub.add_parser("white-diagnose", help="Enhance white textile cut seams and save all intermediate images")
    white.add_argument("--input", required=True)
    white.add_argument("--output", default="outputs/white_textile_diagnostics")
    white.add_argument("--recursive", action="store_true")
    white.add_argument("--max-edge", type=int, default=1600)
    white.add_argument("--seam-percentile", type=float, default=88.0)
    fluorescent = sub.add_parser(
        "fluorescent-diagnose",
        help="Detect fluorescent detached textiles and save masks, pick points and results",
    )
    fluorescent.add_argument("--input", required=True)
    fluorescent.add_argument("--output", default="outputs/fluorescent_diagnostics")
    fluorescent.add_argument("--recursive", action="store_true")
    fluorescent.add_argument("--max-edge", type=int, default=1920)
    fluorescent.add_argument("--hue-min", type=int, default=20)
    fluorescent.add_argument("--hue-max", type=int, default=100)
    fluorescent.add_argument("--min-saturation", type=int, default=62)
    fluorescent.add_argument("--min-area-ratio", type=float, default=0.003)
    fluorescent.add_argument("--max-area-ratio", type=float, default=0.12)
    benchmark = sub.add_parser(
        "benchmark",
        help="Run deterministic synthetic stress tests against the persistent template library",
    )
    benchmark.add_argument("--templates", default="templates")
    benchmark.add_argument("--output", default="outputs/synthetic_benchmark")
    benchmark.add_argument("--scenes", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--width", type=int, default=2560)
    benchmark.add_argument("--height", type=int, default=1600)
    benchmark.add_argument("--min-objects", type=int, default=1)
    benchmark.add_argument("--max-objects", type=int, default=4)
    benchmark.add_argument("--profile", choices=["clean", "balanced", "harsh"], default="balanced")
    benchmark.add_argument("--allow-overlap", action="store_true")
    benchmark.add_argument("--center-tolerance", type=float, default=60.0)
    benchmark.add_argument("--angle-tolerance", type=float, default=8.0)
    benchmark.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    real = sub.add_parser(
        "validate-real",
        help="Evaluate reviewed real images and render multi-algorithm visual comparisons",
    )
    real.add_argument("--dataset", required=True, help="Directory containing images/ and label_maps/")
    real.add_argument("--output", default="outputs/real_validation")
    real.add_argument(
        "--algorithms", nargs="+", choices=REAL_ALGORITHM_NAMES, default=list(REAL_ALGORITHM_NAMES)
    )
    real.add_argument("--minimum-iou", type=float, default=0.30)
    real.add_argument("--max-visual-images", type=int, default=12)
    real.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Fixed valid detection area in original-image pixels",
    )
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


def run_white_diagnose(args: argparse.Namespace) -> int:
    paths = discover_images(args.input, args.recursive)
    if not paths:
        raise ValueError(f"No supported images found under: {args.input}")
    parameters = WhiteTextileParameters(
        max_processing_edge=args.max_edge,
        seam_percentile=args.seam_percentile,
    )
    parameters.validate()
    output_root = Path(args.output)
    for index, path in enumerate(paths, 1):
        image = read_image(path)
        result = segment_white_textile(image, parameters)
        destination = output_root / f"{index:03d}_{path.stem}"
        save_white_textile_diagnostics(result, destination, image)
        logging.info(
            "[%d/%d] %s: candidates=%d coverage=%.3f quality=%.3f output=%s",
            index, len(paths), path.name, result.candidate_count,
            result.coverage, result.quality_score, destination,
        )
    return 0


def run_fluorescent_diagnose(args: argparse.Namespace) -> int:
    paths = discover_images(args.input, args.recursive)
    if not paths:
        raise ValueError(f"No supported images found under: {args.input}")
    parameters = FluorescentTextileParameters(
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        min_saturation=args.min_saturation,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )
    detector = FluorescentTextileDetector(parameters)
    detector.max_processing_edge = args.max_edge
    detector.max_processing_pixels = args.max_edge * args.max_edge
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for index, path in enumerate(paths, 1):
        image = read_image(path)
        started = perf_counter()
        result = detector.match(image)
        elapsed_ms = (perf_counter() - started) * 1000.0
        destination = output_root / f"{index:03d}_{path.stem}"
        destination.mkdir(parents=True, exist_ok=True)
        write_image(destination / "01_original.png", image)
        write_image(destination / "02_annotated.png", draw_multi_template_matches(image, result))
        for name, debug_image in (result.debug_images or {}).items():
            write_image(destination / f"debug_{name}.png", debug_image)
        payload = {"source": str(path), "elapsed_ms": elapsed_ms, **result.to_dict()}
        (destination / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary.append({"source": str(path), "count": result.object_count, "elapsed_ms": elapsed_ms})
        pickable_count = sum(item.auto_pick_allowed for item in result.objects)
        logging.info(
            "[%d/%d] %s: %d detected, %d pickable (%.1f ms)",
            index, len(paths), path.name, result.object_count, pickable_count, elapsed_ms,
        )
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    config = SyntheticStressConfig(
        width=args.width,
        height=args.height,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        profile=args.profile,
        allow_overlap=args.allow_overlap,
    )
    thresholds = EvaluationThresholds(
        center_tolerance_px=args.center_tolerance,
        angle_tolerance_deg=args.angle_tolerance,
    )
    output, summary = run_synthetic_benchmark(
        args.templates,
        args.output,
        scene_count=args.scenes,
        seed=args.seed,
        config=config,
        thresholds=thresholds,
        save_images=args.save_images,
    )
    logging.info(
        "Benchmark done: scenes=%d GT=%d predictions=%d precision=%.3f recall=%.3f F1=%.3f output=%s",
        summary.scene_count,
        summary.ground_truth_count,
        summary.prediction_count,
        summary.precision,
        summary.recall,
        summary.f1,
        output,
    )
    return 0


def run_real_validation(args: argparse.Namespace) -> int:
    result = run_real_dataset_validation(
        args.dataset,
        args.output,
        algorithms=args.algorithms,
        minimum_iou=args.minimum_iou,
        max_visual_images=args.max_visual_images,
        roi=tuple(args.roi) if args.roi else None,
    )
    for name, summary in result.summaries.items():
        logging.info(
            "%s: IoU=%.3f precision=%.3f recall=%.3f F1=%.3f exact-count=%.3f time=%.1f ms",
            name,
            summary.mean_iou,
            summary.precision,
            summary.recall,
            summary.f1,
            summary.exact_count_accuracy,
            summary.mean_processing_time_ms,
        )
    logging.info("Real validation output: %s", result.run_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "white-diagnose":
            return run_white_diagnose(args)
        if args.command == "fluorescent-diagnose":
            return run_fluorescent_diagnose(args)
        if args.command == "benchmark":
            return run_benchmark(args)
        if args.command == "validate-real":
            return run_real_validation(args)
        return run_infer(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logging.error("%s", exc); return 2


if __name__ == "__main__":
    raise SystemExit(main())
