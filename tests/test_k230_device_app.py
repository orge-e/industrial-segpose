from k230_runtime.device_app import DeviceVisionApp


class Camera:
    overlay_image = object()

    def read(self):
        return "frame"

    def show_overlay(self):
        return None


class Frame:
    def width(self):
        return 640

    def height(self):
        return 480

    def copy(self):
        return self


class Clock:
    def millis(self):
        return 100


class Detector:
    def detect(self, frame, timestamp_ms):
        raise RuntimeError("axis failure")


class Tracker:
    total_count = 0
    counts = {}

    def update(self, detections):
        return detections


class TouchUI:
    page = "detect"
    message = ""

    def poll(self, template_count=0):
        return None

    def render(self, image, frame, detections, tracker, fps, library, frozen_frame=None):
        assert detections == []


class Reporter:
    def __init__(self):
        self.calls = []

    def report(self, error, stage, context):
        self.calls.append((str(error), stage, context))
        return "/logs/error_000001.log"


class Transport:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(message)


class Library:
    templates = []


def test_detection_exception_is_reported_without_stopping_live_view():
    reporter = Reporter()
    app = DeviceVisionApp(
        {"heartbeat_interval_ms": 1000}, Camera(), Detector(), Tracker(), object(),
        Transport(), Clock(), touch_ui=TouchUI(), template_library=Library(),
        error_reporter=reporter,
    )

    assert app.process_once() is True
    assert reporter.calls[0][0:2] == ("axis failure", "detect")
    assert reporter.calls[0][2]["page"] == "detect"


class PreviewTouchUI:
    page = "confirm"
    message = ""
    auto_roi = None
    template_preview = None

    def __init__(self):
        self.event = "capture_template_preview"

    def poll(self, template_count=0):
        return self.event


class PreviewBuilder:
    def __init__(self):
        self.roi = None

    def preview(self, frame, roi):
        self.roi = roi
        return {"valid": True, "message": "preview ready"}


class FreezingCamera(Camera):
    def __init__(self):
        self.freeze_count = 0
        self.resume_count = 0

    def freeze_display(self):
        self.freeze_count += 1

    def resume_display(self):
        self.resume_count += 1


def test_template_capture_generates_full_frame_preview_before_manual_roi():
    builder = PreviewBuilder()
    ui = PreviewTouchUI()
    camera = FreezingCamera()
    app = DeviceVisionApp(
        {}, camera, Detector(), Tracker(), object(), Transport(), Clock(),
        touch_ui=ui, template_library=Library(), template_builder=builder,
    )

    app._handle_touch(Frame())

    assert builder.roi == (0, 0, 640, 480)
    assert ui.auto_roi == (0, 0, 640, 480)
    assert ui.template_preview["valid"] is True
    assert ui.message == "preview ready"
    assert ui.build_frame_id == 1
    assert camera.freeze_count == 1

    ui.event = "recapture_template"
    app._handle_touch(Frame())
    assert camera.resume_count == 1
    assert app.frozen_frame is None


class CountingCamera(Camera):
    def __init__(self):
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return Frame()


class FrozenTouchUI(TouchUI):
    page = "confirm"


def test_frozen_template_editing_does_not_request_new_camera_frames():
    camera = CountingCamera()
    app = DeviceVisionApp(
        {"heartbeat_interval_ms": 1000}, camera, Detector(), Tracker(), object(),
        Transport(), Clock(), touch_ui=FrozenTouchUI(), template_library=Library(),
    )
    frozen = Frame()
    app.frozen_frame = frozen

    assert app.process_once() is True

    assert camera.read_count == 0


def test_pipeline_camera_unbinds_and_rebinds_hardware_video_layer_without_stopping_sensor():
    from k230_runtime.canmv_adapter import CanMVPipelineCamera

    class Sensor:
        def __init__(self):
            self.stops = 0
            self.runs = 0

        def stop(self):
            self.stops += 1

        def run(self):
            self.runs += 1

        def bind_info(self, **kwargs):
            return {"src": (1, 2, 3), "rect": (0, 0, 640, 480), "pix_format": 4}

    class Pipeline:
        sensor = Sensor()

    class Display:
        LAYER_VIDEO1 = 1
        unbinds = 0
        binds = 0

        @classmethod
        def unbind_layer(cls, layer):
            cls.unbinds += 1
            return True

        @classmethod
        def bind_layer(cls, **kwargs):
            cls.binds += 1

    camera = CanMVPipelineCamera()
    camera.pipeline = Pipeline()
    camera.display_api = Display
    camera.video_bind_info = camera.pipeline.sensor.bind_info()

    camera.freeze_display()
    camera.freeze_display()
    camera.resume_display()
    camera.resume_display()

    assert Display.unbinds == 1
    assert Display.binds == 1
    assert camera.pipeline.sensor.stops == 0
    assert camera.pipeline.sensor.runs == 0


def test_heartbeat_exposes_current_template_center_angle_and_counts():
    from shared_protocol import decode_message

    transport = Transport()
    app = DeviceVisionApp(
        {"device_id": "test", "heartbeat_interval_ms": 1000}, Camera(), Detector(),
        Tracker(), object(), transport, Clock(), touch_ui=TouchUI(),
        template_library=Library(),
    )
    detection = {
        "track_id": 3,
        "template_id": "template_abc",
        "template_name": "Pink",
        "status": "ready",
        "image_center": [123.0, 234.0],
        "angle_deg": 27.5,
        "confidence": 0.92,
    }

    app._emit_due_heartbeat(100, [detection])

    message = decode_message(transport.messages[0])
    counters = message["counters"]
    assert counters["current_detection_count"] == 1
    assert counters["current_counts_by_template"] == {"Pink": 1}
    assert counters["current_objects"][0]["template_id"] == "template_abc"
    assert counters["current_objects"][0]["image_center"] == [123.0, 234.0]
    assert counters["current_objects"][0]["angle_deg"] == 27.5
