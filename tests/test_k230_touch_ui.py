from k230_runtime.touch_ui import TouchUI


def test_touch_toolbar_zoom_and_fit():
    ui = TouchUI()
    assert ui.process_tap(20, 450) == "zoom_in"
    assert ui.zoom == 1.25
    assert ui.process_tap(150, 450) == "zoom_out"
    assert ui.zoom == 1.0
    ui.zoom = 2.0
    assert ui.process_tap(250, 450) == "fit"
    assert ui.zoom == 1.0


def test_touch_template_roi_flow():
    ui = TouchUI()
    assert ui.process_tap(400, 450) == "new_template"
    assert ui.process_tap(100, 80) == "roi_first"
    assert ui.process_tap(420, 350) == "roi_complete"
    assert ui.selected_roi() == (100, 80, 320, 270)


def test_touch_rejects_small_roi():
    ui = TouchUI()
    ui.process_tap(400, 450)
    ui.process_tap(100, 100)
    ui.process_tap(110, 110)
    assert ui.selected_roi() is None
