from k230_runtime.overlay import DetectionOverlay


class FakeFrame:
    def width(self):
        return 640

    def height(self):
        return 480

    def copy(self, roi=None):
        return ("crop", roi)


class FakeImage:
    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.append(("clear",))

    def draw_image(self, frame, x, y, **kwargs):
        self.calls.append(("draw_image", frame, x, y, kwargs))

    def draw_line(self, *args, **kwargs):
        self.calls.append(("draw_line", args, kwargs))

    def draw_rectangle(self, *args, **kwargs):
        self.calls.append(("draw_rectangle", args, kwargs))

    def draw_cross(self, *args, **kwargs):
        self.calls.append(("draw_cross", args, kwargs))

    def draw_arrow(self, *args, **kwargs):
        self.calls.append(("draw_arrow", args, kwargs))

    def draw_string_advanced(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))


def test_detection_overlay_redraws_live_frame_at_default_zoom():
    image = FakeImage()
    frame = FakeFrame()

    DetectionOverlay().draw(image, [], frame=frame, zoom=1.0, draw_toolbar=False)

    assert image.calls[0] == ("clear",)
    assert ("draw_image", frame, 0, 0, {}) in image.calls


def test_detection_overlay_uses_center_crop_when_zoomed():
    image = FakeImage()
    frame = FakeFrame()

    DetectionOverlay().draw(image, [], frame=frame, zoom=2.0, draw_toolbar=False)

    draw_calls = [call for call in image.calls if call[0] == "draw_image"]
    assert draw_calls == [("draw_image", ("crop", (160, 120, 320, 240)), 0, 0, {"x_scale": 2.0, "y_scale": 2.0})]


def test_detection_overlay_shows_template_pose_and_current_frame_count():
    image = FakeImage()
    detection = {
        "track_id": 7,
        "template_id": "template_12345678",
        "template_name": "Pink_Textile",
        "status": "ready",
        "bbox": [100, 120, 180, 90],
        "image_center": [190.0, 165.0],
        "angle_deg": 32.5,
        "confidence": 0.91,
        "primary_pick_point": {"x": 205.0, "y": 170.0, "safe_radius_px": 12.0, "score": 1.0},
        "safe_radius_px": 12.0,
        "quality_flags": 0,
        "auto_pick_allowed": True,
    }

    DetectionOverlay().draw(
        image, [detection], total_count=4, counts={"Pink_Textile": 4},
        fps=12.0, frame=FakeFrame(), draw_toolbar=False,
    )

    texts = [call[1][3] for call in image.calls if call[0] == "text"]
    assert any("TYPE:Pink_Textile" in value for value in texts)
    assert any("C:(190,165) A:32.5 S:0.91" in value for value in texts)
    assert any("COUNT:1  FPS:12.0" in value for value in texts)
    assert any("TYPES: Pink_Textile:1" in value for value in texts)
    assert not any("LINE TOTAL" in value or "LINE:" in value for value in texts)
    assert "DETECTION RESULT" in texts
    assert any("TYPE: Pink_Textile" in value for value in texts)
    assert any("ID:7 OK  X:190 Y:165" in value for value in texts)
    assert any("ANGLE:32.5 P:180? SCORE:0.91" in value for value in texts)
    assert any("PICK:205,170 R:12.0 F:0" in value for value in texts)
    assert not any("12345678" in value for value in texts)
    assert not any(call[0] == "draw_line" for call in image.calls)
    type_calls = [call for call in image.calls if call[0] == "text" and "TYPE:" in call[1][3]]
    assert type_calls
    assert all(call[2]["color"] == (255, 255, 255, 255) for call in type_calls)


def test_detection_overlay_hides_internal_template_id_when_name_is_empty():
    detection = {
        "track_id": 2,
        "template_id": "template_deadbeef",
        "template_name": "",
        "status": "ready",
        "bbox": [10, 80, 100, 60],
        "image_center": [60.0, 110.0],
        "angle_deg": 0.0,
        "confidence": 0.8,
    }
    image = FakeImage()

    DetectionOverlay().draw(image, [detection], frame=FakeFrame(), draw_toolbar=False)

    texts = [call[1][3] for call in image.calls if call[0] == "text"]
    assert any("TYPE:Unnamed_Template" in value for value in texts)
    assert not any("deadbeef" in value for value in texts)


def test_detection_overlay_decodes_utf8_template_name_bytes():
    detection = {
        "track_id": 3,
        "template_id": "internal_uuid",
        "template_name": "粉色鞋面".encode("utf-8"),
        "status": "ready",
        "bbox": [10, 80, 100, 60],
        "image_center": [60.0, 110.0],
        "angle_deg": 0.0,
        "confidence": 0.8,
    }
    image = FakeImage()

    DetectionOverlay().draw(image, [detection], frame=FakeFrame(), draw_toolbar=False)

    texts = [call[1][3] for call in image.calls if call[0] == "text"]
    assert any("TYPE:粉色鞋面" in value for value in texts)
