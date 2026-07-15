"""Evaluate count, center, angle, mask overlap, and latency on labeled scenes."""

import argparse
import json
from pathlib import Path
import numpy as np
from industrial_segpose.config import load_config
from industrial_segpose.io.image_reader import read_image
from industrial_segpose.pipeline import SegPosePipeline


def angle_error(a, b):
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--config", required=True); args = parser.parse_args()
    pipeline = SegPosePipeline(load_config(args.config)); counts = []; centers = []; angles = []; ious = []; dices = []; times = []
    for truth_path in sorted(args.data.glob("*/ground_truth.json")):
        truth = json.loads(truth_path.read_text()); image = read_image(truth_path.parent / truth["image_name"]); result = pipeline.process(image, truth["image_name"]); times.append(result.processing_time_ms); counts.append(result.object_count == truth["object_count"])
        predicted = sorted(result.objects, key=lambda o: (o.center_y, o.center_x)); actual = sorted(truth["objects"], key=lambda o: (o["center_y"], o["center_x"]))
        for pred, gt in zip(predicted, actual):
            centers.append(float(np.hypot(pred.center_x - gt["center_x"], pred.center_y - gt["center_y"]))); angles.append(angle_error(pred.angle_deg, gt["angle_deg"]))
            gt_mask = read_image(truth_path.parent / "masks" / gt["mask_filename"])[:, :, 0] > 0; pred_mask = pred._mask > 0
            intersection = np.count_nonzero(gt_mask & pred_mask); union = np.count_nonzero(gt_mask | pred_mask); ious.append(intersection / max(union, 1)); dices.append(2 * intersection / max(np.count_nonzero(gt_mask) + np.count_nonzero(pred_mask), 1))
    metrics = {"images": len(times), "count_accuracy": float(np.mean(counts)), "center_mean_error_px": float(np.mean(centers)) if centers else None, "angle_mean_error_deg": float(np.mean(angles)) if angles else None, "mean_iou": float(np.mean(ious)) if ious else None, "mean_dice": float(np.mean(dices)) if dices else None, "mean_time_ms": float(np.mean(times)), "p50_time_ms": float(np.percentile(times, 50)), "p95_time_ms": float(np.percentile(times, 95))}
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
