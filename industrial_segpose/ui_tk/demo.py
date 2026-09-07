"""Tk 稳定版界面：生成可复现的纺织件演示素材。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..io.image_reader import write_image
from ..template_matching import MatchParameters, TemplateLibrary, TemplateModel
from ..template_matching.matcher import _rotation_geometry


@dataclass(frozen=True)
class TemplateDemoCase:
    detection_path: Path
    expected_counts: dict[str, int]
    template_ids: tuple[str, ...]


def _textile_model(name: str, variant: str) -> TemplateModel:
    height, width = 96, 160
    background = (216, 216, 216)
    image = np.full((height, width, 3), background, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    if variant == "A":
        points = np.array(
            [[7, 42], [17, 25], [48, 22], [70, 34], [88, 38], [103, 29],
             [112, 9], [132, 12], [124, 29], [149, 35], [157, 49], [146, 62],
             [118, 58], [104, 66], [96, 87], [77, 87], [70, 64], [54, 56],
             [42, 83], [19, 82], [29, 56], [11, 55]],
            np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        image[mask > 0] = (188, 163, 216)
        curve = np.array([[25, 42], [48, 38], [70, 48], [92, 57], [119, 45], [141, 43]], np.int32)
        cv2.polylines(image, [curve], False, (58, 48, 174), 4, cv2.LINE_AA)
        cv2.circle(image, (28, 37), 3, (48, 42, 155), -1, cv2.LINE_AA)
    else:
        points = np.array(
            [[9, 28], [45, 22], [55, 7], [78, 6], [82, 24], [125, 27],
             [143, 42], [130, 61], [91, 57], [80, 86], [57, 87], [50, 59],
             [15, 55]],
            np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        image[mask > 0] = (174, 186, 224)
        cv2.line(image, (29, 38), (119, 43), (71, 79, 170), 4, cv2.LINE_AA)
        cv2.circle(image, (67, 42), 8, (220, 220, 220), 3, cv2.LINE_AA)
        cv2.circle(image, (112, 41), 4, (62, 68, 154), -1, cv2.LINE_AA)
    image[mask == 0] = background
    return TemplateModel(name=name, image=image, mask=mask, source_image="built-in-template-demo")


def _paste_rotated(scene: np.ndarray, model: TemplateModel, center: tuple[int, int], angle: float) -> None:
    height, width = model.image.shape[:2]
    matrix, rotated_width, rotated_height = _rotation_geometry(width, height, angle)
    rotated = cv2.warpAffine(
        model.image,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(216, 216, 216),
    )
    rotated_mask = cv2.warpAffine(
        model.mask,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    left = int(round(center[0] - rotated_width / 2))
    top = int(round(center[1] - rotated_height / 2))
    region = scene[top : top + rotated_height, left : left + rotated_width]
    region[rotated_mask > 0] = rotated[rotated_mask > 0]


def _demo_scene(models: tuple[TemplateModel, TemplateModel]) -> np.ndarray:
    height, width = 440, 720
    x_gradient = np.linspace(0, 7, width, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(0, 5, height, dtype=np.float32)[:, None, None]
    scene = np.full((height, width, 3), 211, np.float32) + x_gradient + y_gradient
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    for center, angle in [((125, 115), -24), ((360, 105), 16), ((595, 125), 44)]:
        _paste_rotated(scene, models[0], center, angle)
    for center, angle in [((220, 325), -36), ((510, 320), 28)]:
        _paste_rotated(scene, models[1], center, angle)
    return scene


def prepare_template_demo(library: TemplateLibrary, output_root: str | Path) -> TemplateDemoCase:
    """Create/reuse two demo templates and a five-object detection scene."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    models = (
        _textile_model("DEMO_Textile_A", "A"),
        _textile_model("DEMO_Textile_B", "B"),
    )
    parameters = (
        MatchParameters(score_threshold=0.76, angle_min=-48, angle_max=48, angle_step=4,
                        scale_min=1.0, scale_max=1.0, feature_mode="textile_chroma", use_edges=False),
        MatchParameters(score_threshold=0.72, angle_min=-48, angle_max=48, angle_step=4,
                        scale_min=1.0, scale_max=1.0, feature_mode="edges", use_edges=True),
    )
    existing = {entry.name: entry for entry in library.entries}
    template_ids: list[str] = []
    for model, params in zip(models, parameters):
        entry = existing.get(model.name)
        if entry is None:
            entry = library.add_model(model, params)
        else:
            loaded = next(item for item in library.load_entries() if item.entry.template_id == entry.template_id)
            if not loaded.valid or loaded.model.source_image != "built-in-template-demo":
                raise ValueError(f"模板库中已存在同名的非演示模板：{model.name}")
            library.update_parameters(entry.template_id, params)
            library.set_enabled(entry.template_id, True)
        template_ids.append(entry.template_id)
    detection_path = root / "textile_detection_demo.png"
    write_image(detection_path, _demo_scene(models))
    return TemplateDemoCase(detection_path, {models[0].name: 3, models[1].name: 2}, tuple(template_ids))
