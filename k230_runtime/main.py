"""K230 manual-run entry point.  Actuator output remains disabled."""

from k230_runtime.canmv_adapter import CanMVClock, CanMVPipelineCamera, ConsoleTransport
from k230_runtime.calibration import K230PlanarCalibration
from k230_runtime.capture import SessionCaptureManager
from k230_runtime.app_ui import IndustrialTouchUI
from k230_runtime.config import load_config
from k230_runtime.detector import BlobTemplateDetector
from k230_runtime.error_reporter import ErrorReporter
from k230_runtime.image_quality import K230ImageQualityGate
from k230_runtime.device_app import DeviceVisionApp
from k230_runtime.overlay import DetectionOverlay
from k230_runtime.template_library import K230TemplateLibrary
from k230_runtime.tracking import CentroidCounter


def main():
    config = load_config("/sdcard/industrial_vision/device_config.json")
    camera_config = config.get("camera", {})
    display_config = config.get("display", {})
    tracking_config = config.get("tracking", {})
    library = K230TemplateLibrary(config["template_library"]).load(verify_assets=False)
    detector = BlobTemplateDetector(library)
    tracker = CentroidCounter(**tracking_config)
    camera = CanMVPipelineCamera(
        width=camera_config.get("width", 640),
        height=camera_config.get("height", 480),
        algorithm_width=camera_config.get("algorithm_width", 320),
        algorithm_height=camera_config.get("algorithm_height", 240),
        to_ide=display_config.get("to_ide", True),
    )
    overlay = DetectionOverlay(
        width=display_config.get("width", 640),
        height=display_config.get("height", 480),
        line_axis=tracking_config.get("line_axis", "x"),
        line_position=tracking_config.get("line_position", 520),
    )
    print("industrial vision dry-run starting", config["device_id"], len(library.templates))
    ui_config = config.get("ui", {})
    touch_ui = IndustrialTouchUI(
        overlay,
        product_name=ui_config.get("product_name", "FlexPose Vision"),
        subtitle=ui_config.get("subtitle", "柔性工件定位系统"),
        template_authoring_enabled=config.get("template_authoring", {}).get("enabled", False),
    )
    template_builder = None
    if config.get("template_authoring", {}).get("enabled", False):
        from k230_runtime.template_builder import OnDeviceTemplateBuilder
        template_builder = OnDeviceTemplateBuilder("/sdcard/industrial_vision/templates")
    capture_config = config.get("capture", {})
    capture_manager = SessionCaptureManager(
        capture_config.get("root", "/sdcard/industrial_vision/captures"),
        capture_config.get("jpeg_quality", 95),
    )
    error_reporter = ErrorReporter(config.get("error_reports", {}).get("root", "/sdcard/industrial_vision/logs"))
    quality_gate = K230ImageQualityGate(config.get("quality_gate", {}))
    calibration = None
    calibration_config = config.get("calibration", {})
    if calibration_config.get("enabled", False):
        try:
            calibration = K230PlanarCalibration.load(calibration_config["path"])
        except Exception as error:
            # Calibration is optional during the current visual-development
            # stage.  A missing or damaged file must not prevent the detection
            # application from starting.
            error_reporter.report(
                error,
                "calibration_load",
                {"path": calibration_config.get("path", "")},
            )
            print("calibration disabled after load failure", error)
    DeviceVisionApp(
        config, camera, detector, tracker, overlay, ConsoleTransport(), CanMVClock(),
        touch_ui=touch_ui, template_builder=template_builder, template_library=library,
        capture_manager=capture_manager, error_reporter=error_reporter,
        quality_gate=quality_gate,
        calibration=calibration,
    ).run()


if __name__ == "__main__":
    main()
