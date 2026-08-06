from k230_runtime.app_ui import IndustrialTouchUI


class Overlay:
    pass


def test_home_routes_to_two_primary_functions():
    ui = IndustrialTouchUI(Overlay())
    assert ui.process_tap(100, 200, 2) == "open_templates"
    assert ui.page == "templates"
    ui.page = "home"
    assert ui.process_tap(450, 200, 2) == "open_detect"
    assert ui.page == "detect"


def test_template_wizard_captures_and_previews_before_manual_roi():
    ui = IndustrialTouchUI(Overlay())
    ui.page = "templates"
    assert ui.process_tap(80, 430, 2) == "new_template"
    assert ui.page == "name"
    assert ui.process_tap(50, 160) == "name_changed"
    assert ui.template_name == "1"
    assert ui.process_tap(500, 410) == "open_template_capture"
    assert ui.page == "capture"
    assert ui.process_tap(300, 430) == "capture_template_preview"
    assert ui.page == "confirm"
    ui.auto_roi = (0, 0, 640, 480)
    ui.template_preview = {"valid": True}
    assert ui.selected_roi() == (0, 0, 640, 480)
    assert ui.process_tap(180, 430) == "preview_original"
    assert ui.preview_view == "original"
    assert ui.process_tap(300, 430) == "preview_mask"
    assert ui.preview_view == "mask"
    assert ui.process_tap(450, 430) == "manual_roi"
    assert ui.page == "roi"
    assert ui.process_tap(100, 100) == "roi_first"
    assert ui.process_tap(400, 350) == "roi_ready"
    assert ui.page == "confirm"
    assert ui.selected_roi() == (100, 100, 300, 250)
    ui.template_preview = {"valid": True}
    assert ui.process_tap(550, 430) == "save_template"


def test_template_confirmation_can_return_to_live_recapture():
    ui = IndustrialTouchUI(Overlay())
    ui.page = "confirm"
    ui.auto_roi = (0, 0, 640, 480)
    ui.template_preview = {"valid": True}
    ui.build_frame_id = 4

    assert ui.process_tap(100, 430) == "recapture_template"
    assert ui.page == "capture"
    assert ui.template_preview is None
    assert ui.build_frame_id is None


def test_detection_controls_and_home_button():
    ui = IndustrialTouchUI(Overlay())
    ui.page = "detect"
    assert ui.process_tap(150, 450) == "zoom_in"
    assert ui.zoom == 1.25
    assert ui.process_tap(250, 450) == "zoom_out"
    assert ui.zoom == 1.0
    assert ui.process_tap(430, 450) == "capture_frame"
    assert ui.process_tap(570, 450) == "reset_tracking"
    assert ui.process_tap(50, 450) == "home"
    assert ui.page == "home"


def test_product_name_is_configurable():
    ui = IndustrialTouchUI(Overlay(), product_name="FlexPose Vision", subtitle="定位系统")
    assert ui.product_name == "FlexPose Vision"
    assert ui.subtitle == "定位系统"


def test_production_ui_keeps_template_library_read_only():
    ui = IndustrialTouchUI(Overlay(), template_authoring_enabled=False)
    ui.page = "templates"

    assert ui.process_tap(80, 430, 2) == "authoring_disabled"
    assert ui.page == "templates"
    assert ui.process_tap(420, 430, 2) == "authoring_disabled"


def test_device_default_preserves_debug_template_authoring():
    from k230_runtime.config import load_config

    assert load_config()["template_authoring"]["enabled"] is True


def test_invalid_template_preview_cannot_be_saved():
    ui = IndustrialTouchUI(Overlay())
    ui.page = "confirm"
    ui.template_preview = {"valid": False, "message": "background flood"}

    assert ui.process_tap(550, 430, 1) == "preview_invalid"
    assert ui.page == "confirm"
