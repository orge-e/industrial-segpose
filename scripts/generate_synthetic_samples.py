"""Generate deterministic synthetic images, instance masks, and ground truth."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def rotated_box(center, size, angle):
    return np.round(cv2.boxPoints((center, size, angle))).astype(np.int32)


def add_object(image, label_map, object_id, center, size, angle, color=(220, 220, 220), ellipse=False):
    mask = np.zeros(image.shape[:2], np.uint8)
    if ellipse:
        cv2.ellipse(mask, tuple(map(int, center)), (int(size[0] / 2), int(size[1] / 2)), angle, 0, 360, 255, -1)
    else:
        cv2.fillConvexPoly(mask, rotated_box(center, size, angle), 255)
    image[mask > 0] = color; label_map[mask > 0] = object_id
    return {"object_id": object_id, "center_x": center[0], "center_y": center[1], "angle_deg": angle % 180, "mask_filename": f"object_{object_id:03d}.png"}, mask


def generate(output: Path) -> None:
    rng = np.random.default_rng(42); output.mkdir(parents=True, exist_ok=True)
    scenes = {
        "separated": [((110, 90), (100, 34), 30, False), ((310, 90), (90, 40), 75, False), ((210, 230), (120, 45), 135, False)],
        "ellipses": [((120, 130), (110, 55), 20, True), ((320, 170), (130, 60), 110, True)],
        "border": [((20, 100), (90, 35), 0, False), ((270, 190), (120, 35), 45, False)],
        "noisy": [((130, 100), (120, 40), 10, False), ((330, 210), (100, 35), 150, False)],
        "uneven_light": [((150, 100), (120, 40), 60, False), ((330, 220), (120, 40), 120, False)],
        "blank": [],
    }
    for name, specs in scenes.items():
        h, w = 320, 480
        if name == "uneven_light":
            gradient = np.tile(np.linspace(0, 65, w, dtype=np.uint8), (h, 1)); image = np.dstack([gradient] * 3)
        else: image = np.zeros((h, w, 3), np.uint8)
        labels = np.zeros((h, w), np.uint16); objects = []
        scene_dir = output / name; masks_dir = scene_dir / "masks"; masks_dir.mkdir(parents=True, exist_ok=True)
        for object_id, (center, size, angle, ellipse) in enumerate(specs, 1):
            obj, mask = add_object(image, labels, object_id, center, size, angle, ellipse=ellipse); objects.append(obj)
            cv2.imencode(".png", mask)[1].tofile(masks_dir / obj["mask_filename"])
        if name == "noisy":
            noise = rng.normal(0, 18, image.shape).astype(np.int16); image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            for _ in range(35):
                x, y = rng.integers(0, w), rng.integers(0, h); cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)
        cv2.imencode(".png", image)[1].tofile(scene_dir / f"{name}.png")
        cv2.imencode(".png", labels)[1].tofile(scene_dir / "label_map.png")
        truth = {"image_name": f"{name}.png", "width": w, "height": h, "object_count": len(objects), "objects": objects}
        (scene_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Generated {len(scenes)} scenes under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("samples/generated")); generate(parser.parse_args().output)
