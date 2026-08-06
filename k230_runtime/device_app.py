"""Integrated on-device detection loop with overlay and dry-run output."""

try:
    import gc
except ImportError:  # pragma: no cover
    gc = None

from shared_protocol import encode_message, make_heartbeat, make_pick_target


class DeviceVisionApp:
    def __init__(self, config, camera, detector, tracker, overlay, transport, clock, touch_ui=None, template_builder=None, template_library=None, capture_manager=None, error_reporter=None):
        self.config = config
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.overlay = overlay
        self.transport = transport
        self.clock = clock
        self.touch_ui = touch_ui
        self.template_builder = template_builder
        self.template_library = template_library
        self.capture_manager = capture_manager
        self.error_reporter = error_reporter
        self.frames = 0
        self.rejected = 0
        self.last_heartbeat = -1
        self.last_frame_ms = None
        self.fps = 0.0
        self.frozen_frame = None
        self.template_sequence = 1
        self.template_capture_sequence = 0
        self.last_error_stage = ""
        self.last_error_signature = ""
        self.last_error_frame = -1000

    def _freeze_camera_display(self):
        method = getattr(self.camera, "freeze_display", None)
        if callable(method):
            method()

    def _resume_camera_display(self):
        method = getattr(self.camera, "resume_display", None)
        if callable(method):
            method()

    def _report_error(self, error, stage):
        self.last_error_stage = str(stage)
        signature = "%s:%s:%s" % (stage, type(error).__name__, error)
        context = {
            "frame": self.frames,
            "page": getattr(self.touch_ui, "page", "none"),
            "templates": len(self.template_library.templates) if self.template_library is not None else 0,
        }
        should_write = signature != self.last_error_signature or self.frames - self.last_error_frame >= 60
        path = self.error_reporter.report(error, stage, context) if self.error_reporter is not None and should_write else None
        if should_write:
            self.last_error_signature = signature
            self.last_error_frame = self.frames
        print("vision error", stage, error, path or "")
        if self.touch_ui is not None:
            self.touch_ui.message = "ERROR %s #%s" % (stage, path.split("_")[-1].split(".")[0] if path else "-")

    def _handle_touch(self, frame):
        if self.touch_ui is None:
            return
        try:
            event = self.touch_ui.poll(len(self.template_library.templates) if self.template_library is not None else 0)
        except TypeError:
            event = self.touch_ui.poll()
        if event is None:
            return
        action = event[0] if isinstance(event, tuple) else event
        if action == "new_template":
            self._resume_camera_display()
            self.frozen_frame = None
        elif action == "open_template_capture":
            self._resume_camera_display()
            self.frozen_frame = None
        elif action == "capture_template_preview":
            self.frozen_frame = frame.copy()
            self._freeze_camera_display()
            self.template_capture_sequence += 1
            self.touch_ui.build_frame_id = self.template_capture_sequence
            roi = (0, 0, int(frame.width()), int(frame.height()))
            self.touch_ui.auto_roi = roi
            if self.template_builder is None:
                self.touch_ui.message = "TEMPLATE BUILDER NOT CONFIGURED"
            else:
                try:
                    preview = self.template_builder.preview(self.frozen_frame, roi)
                    self.touch_ui.template_preview = preview
                    self.touch_ui.message = preview.get("message", "CHECK SEGMENTATION PREVIEW")
                    self.touch_ui.page = "confirm"
                except Exception as error:
                    self.touch_ui.template_preview = None
                    self.touch_ui.message = "PREVIEW FAILED: " + str(error)
                    self._report_error(error, "template_preview")
        elif action == "open_detect":
            self._resume_camera_display()
            self.frozen_frame = None
        elif action == "home":
            self._resume_camera_display()
            self.frozen_frame = None
        elif action == "toggle_template" and self.template_library is not None and self.template_library.templates:
            index = min(self.touch_ui.selected_template, len(self.template_library.templates) - 1)
            template = self.template_library.templates[index]
            self.template_library.set_enabled(template["template_id"], not template.get("enabled", True))
        elif action == "remove_template" and self.template_library is not None and self.template_library.templates:
            index = min(self.touch_ui.selected_template, len(self.template_library.templates) - 1)
            self.template_library.remove(self.template_library.templates[index]["template_id"])
            self.touch_ui.selected_template = max(0, min(index, len(self.template_library.templates) - 1))
        elif action == "reset_count":
            self.tracker.total_count = 0
            self.tracker.counts = {}
        elif action == "capture_frame":
            if self.capture_manager is None:
                self.touch_ui.message = "拍摄功能未配置"
            else:
                try:
                    path = self.capture_manager.capture(frame)
                    self.touch_ui.message = "已保存: " + path.split("/")[-1]
                except Exception as error:
                    self.touch_ui.message = "保存失败: " + str(error)
        elif action in ("cancel_roi", "cancel_name", "cancel_capture", "recapture_template"):
            self._resume_camera_display()
            self.frozen_frame = None
            self.touch_ui.template_preview = None
            if hasattr(self.touch_ui, "auto_roi"):
                self.touch_ui.auto_roi = None
            if hasattr(self.touch_ui, "build_frame_id"):
                self.touch_ui.build_frame_id = None
        elif action in ("retry_roi", "manual_roi"):
            self.touch_ui.template_preview = None
            if hasattr(self.touch_ui, "auto_roi"):
                self.touch_ui.auto_roi = None
        elif action in ("roi_ready", "roi_complete") and self.template_builder is not None:
            roi = self.touch_ui.selected_roi()
            if roi is None:
                self.touch_ui.message = "ROI过小，请重新选择"
                return
            try:
                source_frame = self.frozen_frame if self.frozen_frame is not None else frame
                preview = self.template_builder.preview(source_frame, roi)
                self.touch_ui.template_preview = preview
                self.touch_ui.message = preview.get("message", "请检查模板预览")
                if hasattr(self.touch_ui, "page"):
                    self.touch_ui.page = "confirm"
            except Exception as error:
                self.touch_ui.template_preview = None
                self.touch_ui.message = "预览失败: " + str(error)
                self._report_error(error, "template_preview")
        elif action == "save_template" and self.template_builder is not None:
            roi = self.touch_ui.selected_roi()
            if roi is None:
                if hasattr(self.touch_ui, "page"):
                    self.touch_ui.page = "roi"
                else:
                    self.touch_ui.mode = "select_roi"
                self.touch_ui.roi_points = []
                self.touch_ui.message = "ROI过小，请重新选择"
                return
            try:
                name = self.touch_ui.template_name if getattr(self.touch_ui, "template_name", "") else "Template_%03d" % self.template_sequence
                source_frame = self.frozen_frame if self.frozen_frame is not None else frame
                preview = getattr(self.touch_ui, "template_preview", None)
                if preview is not None and not preview.get("valid", False):
                    self.touch_ui.message = preview.get("message", "模板预览无效")
                    return
                threshold = preview.get("threshold") if preview is not None else None
                mode = preview.get("mode") if preview is not None else None
                self.template_builder.create(
                    source_frame, roi, name, threshold=threshold,
                    segmentation_mode=mode, confirmed_preview=preview,
                )
                self.template_sequence += 1
                if self.template_library is not None:
                    self.template_library.load(verify_assets=False)
                if hasattr(self.touch_ui, "page"):
                    self.touch_ui.page = "templates"
                else:
                    self.touch_ui.mode = "detect"
                self.touch_ui.roi_points = []
                if hasattr(self.touch_ui, "auto_roi"):
                    self.touch_ui.auto_roi = None
                self.touch_ui.message = "TEMPLATE SAVED: " + name
                self.touch_ui.template_preview = None
                if hasattr(self.touch_ui, "build_frame_id"):
                    self.touch_ui.build_frame_id = None
                self._resume_camera_display()
                self.frozen_frame = None
            except Exception as error:
                if hasattr(self.touch_ui, "page"):
                    self.touch_ui.page = "confirm"
                else:
                    self.touch_ui.mode = "detect"
                self.touch_ui.message = "BUILD FAILED: " + str(error)

    def _write(self, message):
        self.transport.write(encode_message(message))

    def _emit_due_heartbeat(self, now_ms, detections=None):
        interval = int(self.config.get("heartbeat_interval_ms", 1000))
        if self.last_heartbeat < 0 or now_ms - self.last_heartbeat >= interval:
            detections = list(detections or [])
            current_counts = {}
            current_objects = []
            for detection in detections[:8]:
                name = str(detection.get("template_name", "object"))
                current_counts[name] = current_counts.get(name, 0) + 1
                current_objects.append({
                    "track_id": int(detection.get("track_id", 0)),
                    "template_id": str(detection.get("template_id", "")),
                    "template_name": name,
                    "status": str(detection.get("status", "ready")),
                    "image_center": list(detection.get("image_center", (0.0, 0.0))),
                    "angle_deg": float(detection.get("angle_deg", 0.0)),
                    "confidence": float(detection.get("confidence", 0.0)),
                })
            self._write(make_heartbeat(
                self.config.get("device_id", "k230"), now_ms, "running",
                {
                    "frames": self.frames,
                    "current_detection_count": len(detections),
                    "current_counts_by_template": current_counts,
                    "current_objects": current_objects,
                    "line_total": self.tracker.total_count,
                    "counts_by_template": dict(self.tracker.counts),
                    "rejected": self.rejected,
                },
            ))
            self.last_heartbeat = now_ms

    def _emit_counted_targets(self, detections, now_ms):
        threshold = float(self.config.get("minimum_confidence", 0.60))
        for detection in detections:
            if not detection.get("emit", False):
                continue
            if detection.get("status") == "ambiguous" or detection.get("confidence", 0.0) < threshold:
                self.rejected += 1
                continue
            self._write(make_pick_target(
                detection["track_id"], detection["template_id"], detection["template_name"],
                detection["confidence"], detection["image_center"], detection["pick_point"],
                detection["angle_deg"], now_ms, status="ready",
            ))

    def _template_frame_is_frozen(self):
        return (
            self.frozen_frame is not None
            and getattr(self.touch_ui, "page", "") in ("confirm", "roi")
        )

    def process_once(self):
        # Once a template frame is captured, stop requesting new snapshots.
        # Every preview, ROI correction and save operation therefore sees the
        # exact same image buffer until the operator saves, cancels or retakes.
        frame = self.frozen_frame if self._template_frame_is_frozen() else self.camera.read()
        if frame is None:
            return False
        now_ms = int(self.clock.millis())
        try:
            self._handle_touch(frame)
        except Exception as error:
            self._report_error(error, "touch")
        detection_enabled = not hasattr(self.touch_ui, "page") or self.touch_ui.page == "detect"
        detections = []
        if detection_enabled:
            try:
                detections = self.tracker.update(self.detector.detect(frame, now_ms))
            except Exception as error:
                self._report_error(error, "detect")
        self.frames += 1
        if self.last_frame_ms is not None and now_ms > self.last_frame_ms:
            instant = 1000.0 / (now_ms - self.last_frame_ms)
            self.fps = instant if self.fps <= 0 else self.fps * 0.8 + instant * 0.2
        self.last_frame_ms = now_ms
        try:
            if self.touch_ui is not None and hasattr(self.touch_ui, "render"):
                self.touch_ui.render(
                    self.camera.overlay_image, frame, detections, self.tracker, self.fps,
                    self.template_library, frozen_frame=self.frozen_frame,
                )
            else:
                self.overlay.draw(
                    self.camera.overlay_image, detections,
                    total_count=self.tracker.total_count, counts=self.tracker.counts, fps=self.fps,
                    ui=self.touch_ui, frame=self.frozen_frame if self.frozen_frame is not None else frame,
                )
        except Exception as error:
            self._report_error(error, "render")
            try:
                self.camera.overlay_image.clear()
                self.camera.overlay_image.draw_image(frame, 0, 0)
            except Exception as fallback_error:
                self._report_error(fallback_error, "render_fallback")
        self.camera.show_overlay()
        self._emit_counted_targets(detections, now_ms)
        self._emit_due_heartbeat(now_ms, detections)
        if gc is not None and self.frames % 5 == 0:
            gc.collect()
        return True

    def run(self, maximum_frames=None):
        self.camera.open()
        self.transport.open()
        if self.touch_ui is not None:
            self.touch_ui.open()
        processed = 0
        try:
            while maximum_frames is None or processed < maximum_frames:
                try:
                    if not self.process_once():
                        break
                except Exception as error:
                    self._report_error(error, "frame_loop")
                processed += 1
        finally:
            self.transport.close()
            self.camera.close()
        return processed
