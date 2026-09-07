import cv2
import numpy as np

from industrial_segpose.detectors.fluorescent import (
    FluorescentTextileDetector,
    FluorescentTextileParameters,
)


def _scene() -> np.ndarray:
    image = np.full((420, 720, 3), (178, 178, 178), np.uint8)
    colour = (35, 235, 205)  # bright yellow-green in BGR
    first = cv2.boxPoints(((205, 205), (190, 110), 24)).astype(np.int32)
    second = cv2.boxPoints(((510, 210), (175, 125), -31)).astype(np.int32)
    cv2.fillPoly(image, [first], colour)
    cv2.fillPoly(image, [second], colour)
    cv2.circle(image, (205, 205), 22, (178, 178, 178), -1)
    return image


def test_fluorescent_detector_counts_pose_and_pick_points():
    detector = FluorescentTextileDetector(
        FluorescentTextileParameters(min_area_ratio=0.01, minimum_pick_radius_px=6)
    )
    result = detector.match(_scene())

    assert result.object_count == 2
    assert result.counts_by_template == {"Fluorescent_Textile": 2}
    assert result.debug_images and "fluorescent_mask" in result.debug_images
    assert all(item.pick_point_x is not None for item in result.objects)
    assert all(item.pick_point_y is not None for item in result.objects)
    assert all(item.safe_radius_px > 6 for item in result.objects)
    assert all(item.auto_pick_allowed for item in result.objects)
    assert all(0 <= item.angle_deg < 360 for item in result.objects)


def test_fluorescent_detector_rejects_neutral_background_and_small_noise():
    image = np.full((240, 320, 3), 180, np.uint8)
    cv2.circle(image, (20, 20), 2, (20, 240, 210), -1)
    result = FluorescentTextileDetector().match(image)
    assert result.object_count == 0


def test_fluorescent_result_serializes_pick_information():
    result = FluorescentTextileDetector(
        FluorescentTextileParameters(min_area_ratio=0.01)
    ).match(_scene())
    payload = result.to_dict()
    first = payload["objects"][0]
    assert first["pick_point_x"] is not None
    assert first["pick_point_y"] is not None
    assert first["safe_radius_px"] > 0
    assert first["orientation_confidence"] >= 0


def test_close_workpieces_are_not_joined_by_noise_cleanup():
    image = np.full((360, 640, 3), (175, 175, 175), np.uint8)
    colour = (35, 235, 205)
    cv2.rectangle(image, (90, 100), (300, 270), colour, -1)
    cv2.rectangle(image, (304, 100), (514, 270), colour, -1)
    result = FluorescentTextileDetector(
        FluorescentTextileParameters(
            min_area_ratio=0.01,
            max_area_ratio=0.40,
            minimum_pick_radius_px=5,
            occlusion_overlap_ratio=0.01,
        )
    ).match(image)
    assert result.object_count == 2


def test_thin_fluorescent_outline_is_not_counted_as_a_workpiece():
    image = np.full((360, 640, 3), (175, 175, 175), np.uint8)
    colour = (35, 235, 205)
    cv2.rectangle(image, (80, 70), (270, 285), colour, 5)
    cv2.rectangle(image, (360, 70), (550, 285), colour, -1)
    result = FluorescentTextileDetector(
        FluorescentTextileParameters(
            min_area_ratio=0.001, max_area_ratio=0.40, minimum_pick_radius_px=5
        )
    ).match(image)
    assert result.object_count == 1


def test_hand_overlap_keeps_frame_count_but_blocks_auto_pick():
    image = np.full((360, 640, 3), (175, 175, 175), np.uint8)
    colour = (35, 235, 205)
    cv2.rectangle(image, (160, 90), (480, 280), colour, -1)
    # Skin-like BGR colour overlaps the right edge of the workpiece.
    cv2.circle(image, (470, 185), 55, (105, 150, 205), -1)
    result = FluorescentTextileDetector(
        FluorescentTextileParameters(
            min_area_ratio=0.01,
            max_area_ratio=0.40,
            minimum_pick_radius_px=5,
            occlusion_overlap_ratio=0.01,
        )
    ).match(image)
    assert result.object_count == 1
    assert result.objects[0].classification_status == "occluded"
    assert not result.objects[0].auto_pick_allowed


def test_dense_connected_layout_is_partitioned_into_instances():
    image = np.full((520, 760, 3), (170, 170, 170), np.uint8)
    colour = (35, 235, 205)
    rows, columns = 3, 4
    for row in range(rows):
        for column in range(columns):
            x0 = 90 + column * 145
            y0 = 70 + row * 135
            cv2.rectangle(image, (x0, y0), (x0 + 105, y0 + 82), colour, -1)
            if column + 1 < columns:
                cv2.rectangle(image, (x0 + 104, y0 + 38), (x0 + 146, y0 + 44), colour, -1)
            if row + 1 < rows:
                cv2.rectangle(image, (x0 + 50, y0 + 81), (x0 + 56, y0 + 136), colour, -1)

    detector = FluorescentTextileDetector(
        FluorescentTextileParameters(
            min_area_ratio=0.002,
            max_area_ratio=0.10,
            minimum_pick_radius_px=5,
            dense_min_rows=3,
            dense_max_rows=5,
            dense_min_columns=3,
            dense_max_columns=6,
        )
    )
    result = detector.match(image)

    assert result.object_count == rows * columns
    assert all(item.classification_status == "dense_layout" for item in result.objects)
    assert "fluorescent_dense_layout" in result.debug_images
