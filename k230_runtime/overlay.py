"""Large, legible on-device annotations for the 640x480 display."""

import math


class DetectionOverlay:
    def __init__(self, width=640, height=480, line_axis="x", line_position=520):
        self.width = int(width)
        self.height = int(height)
        self.line_axis = line_axis
        self.line_position = int(line_position)

    @staticmethod
    def _color(detection):
        if detection.get("status") == "ambiguous":
            return (255, 210, 0, 255)
        return (0, 190, 255, 255)

    @staticmethod
    def _text(image, x, y, text, color, size=22):
        if hasattr(image, "draw_string_advanced"):
            image.draw_string_advanced(x, y, size, text, color=color)
        else:
            image.draw_string(x, y, text, color=color, scale=2)

    @staticmethod
    def _template_name(detection):
        name = detection.get("template_name") or detection.get("name")
        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8")
            except (UnicodeError, AttributeError):
                name = ""
        name = str(name or "").strip()
        template_id = str(detection.get("template_id", "")).strip()
        if not name or name == template_id:
            return "Unnamed_Template"
        return name

    @staticmethod
    def _current_counts(detections):
        counts = {}
        for detection in detections:
            name = DetectionOverlay._template_name(detection)
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _result_panel(self, image, detections):
        # Keep the complete pose result in a fixed panel. Text attached to a
        # moving bounding box can be clipped at image borders or hidden by the
        # toolbar, so it is only supplementary.
        panel_x, panel_y, panel_w = 348, 55, 284
        visible = list(detections[:2])
        row_height = 94
        panel_h = 42 + max(1, len(visible)) * row_height
        image.draw_rectangle(panel_x, panel_y, panel_w, panel_h, color=(7, 15, 27, 225), fill=True)
        image.draw_rectangle(panel_x, panel_y, panel_w, panel_h, color=(90, 180, 245, 255), thickness=2)
        self._text(image, panel_x + 10, panel_y + 7, "DETECTION RESULT", (130, 220, 255, 255), size=18)
        if not visible:
            self._text(image, panel_x + 10, panel_y + 40, "NO TARGET", (190, 205, 220, 255), size=17)
            return
        for index, detection in enumerate(visible):
            row_y = panel_y + 34 + index * row_height
            status = "AMB" if detection.get("status") == "ambiguous" else "OK"
            name = self._template_name(detection)[:18]
            center = detection.get("image_center", (0.0, 0.0))
            self._text(
                image, panel_x + 9, row_y,
                "TYPE: %s" % name,
                (255, 255, 255, 255), size=18,
            )
            self._text(
                image, panel_x + 9, row_y + 21,
                "ID:%s %s  X:%.0f Y:%.0f" % (
                    detection.get("track_id", 0), status, center[0], center[1],
                ),
                (235, 240, 248, 255), size=16,
            )
            self._text(
                image, panel_x + 9, row_y + 42,
                "ANGLE:%.1f P:%s SCORE:%.2f" % (
                    detection.get("angle_deg", 0.0),
                    "360" if detection.get("angle_direction_reliable", False) else "180?",
                    detection.get("confidence", 0.0),
                ),
                (235, 240, 248, 255), size=16,
            )
            pick = detection.get("primary_pick_point") or {}
            pick_x = pick.get("x", detection.get("pick_point", (0.0, 0.0))[0])
            pick_y = pick.get("y", detection.get("pick_point", (0.0, 0.0))[1])
            self._text(
                image, panel_x + 9, row_y + 63,
                "PICK:%.0f,%.0f R:%.1f F:%d" % (
                    pick_x, pick_y, detection.get("safe_radius_px", 0.0),
                    int(detection.get("quality_flags", 0)),
                ),
                (160, 255, 185, 255) if detection.get("auto_pick_allowed", True) else (255, 175, 90, 255),
                size=15,
            )

    def _toolbar(self, image, ui):
        if ui is None:
            return
        y = ui.TOOLBAR_Y
        labels = [(0, 110, "+ ZOOM"), (110, 220, "- ZOOM"), (220, 330, "FIT"), (330, 500, "NEW TEMPLATE"), (500, 640, "RESET")]
        for left, right, label in labels:
            image.draw_rectangle(left, y, right - left, self.height - y, color=(25, 40, 60, 230), thickness=1, fill=True)
            image.draw_rectangle(left, y, right - left, self.height - y, color=(120, 180, 255, 255), thickness=2)
            self._text(image, left + 7, y + 13, label, (255, 255, 255, 255))
        self._text(image, 10, 38, "ZOOM:%.2f  %s" % (ui.zoom, ui.message), (255, 255, 255, 255))
        if ui.roi_points:
            x, point_y = ui.roi_points[0]
            image.draw_cross(x, point_y, color=(255, 220, 0, 255), size=16, thickness=4)
        roi = ui.selected_roi()
        if roi is not None:
            image.draw_rectangle(*roi, color=(255, 220, 0, 255), thickness=4)

    def _draw_zoomed_frame(self, image, frame, zoom):
        if frame is None:
            return 0.0, 0.0
        if zoom <= 1.001:
            # The OSD canvas is cleared at the start of every detection frame.
            # Draw the live image back even at 1x; otherwise entering the
            # detection page leaves a completely black canvas.
            image.draw_image(frame, 0, 0)
            return 0.0, 0.0
        source_width = max(1, int(self.width / zoom))
        source_height = max(1, int(self.height / zoom))
        left = max(0, int((frame.width() - source_width) / 2))
        top = max(0, int((frame.height() - source_height) / 2))
        crop = frame.copy(roi=(left, top, source_width, source_height))
        image.draw_image(crop, 0, 0, x_scale=zoom, y_scale=zoom)
        return float(left), float(top)

    def draw(self, image, detections, total_count=0, counts=None, fps=0.0, ui=None, frame=None, zoom=None, draw_toolbar=True):
        image.clear()
        zoom = float(zoom if zoom is not None else (ui.zoom if ui is not None else 1.0))
        view_left, view_top = self._draw_zoomed_frame(image, frame, zoom)
        for detection in detections:
            color = self._color(detection)
            raw_x, raw_y, raw_width, raw_height = detection["bbox"]
            x = int((raw_x - view_left) * zoom)
            y = int((raw_y - view_top) * zoom)
            width = int(raw_width * zoom)
            height = int(raw_height * zoom)
            cx = int((detection["image_center"][0] - view_left) * zoom)
            cy = int((detection["image_center"][1] - view_top) * zoom)
            angle = math.radians(float(detection["angle_deg"]))
            arrow_length = max(35, int(max(width, height) * 0.35))
            ex = int(cx + math.cos(angle) * arrow_length)
            ey = int(cy + math.sin(angle) * arrow_length)
            image.draw_rectangle(x, y, width, height, color=color, thickness=4)
            image.draw_cross(cx, cy, color=(255, 60, 60, 255), size=14, thickness=4)
            image.draw_arrow(cx, cy, ex, ey, color=color, thickness=4)
            pick = detection.get("primary_pick_point") or {}
            if pick:
                pick_x = int((float(pick.get("x", detection["image_center"][0])) - view_left) * zoom)
                pick_y = int((float(pick.get("y", detection["image_center"][1])) - view_top) * zoom)
                image.draw_cross(pick_x, pick_y, color=(50, 255, 120, 255), size=12, thickness=3)
            status = "AMB" if detection.get("status") == "ambiguous" else "OK"
            template_name = self._template_name(detection)[:18]
            line1 = "TYPE:%s" % template_name
            line2 = "#%s %s C:(%.0f,%.0f) A:%.1f S:%.2f" % (
                detection.get("track_id", 0), status,
                detection.get("image_center", (0, 0))[0],
                detection.get("image_center", (0, 0))[1],
                detection.get("angle_deg", 0.0), detection.get("confidence", 0.0),
            )
            label_x = max(2, min(x, self.width - 360))
            label_y = max(52, min(y - 48, self.height - 92))
            if hasattr(image, "draw_rectangle"):
                image.draw_rectangle(label_x - 2, label_y - 2, 360, 45, color=(8, 16, 26, 210), fill=True)
            # White text is intentionally independent of the outline colour.
            # Some K230 display pipelines lose coloured glyphs over the live
            # video layer even though white text remains visible.
            self._text(image, label_x, label_y, line1, (255, 255, 255, 255), size=18)
            self._text(image, label_x, label_y + 21, line2, (255, 255, 255, 255), size=16)

        current_counts = self._current_counts(detections)
        summary = "COUNT:%d  FPS:%.1f" % (len(detections), float(fps))
        if hasattr(image, "draw_rectangle"):
            image.draw_rectangle(0, 0, self.width, 50, color=(8, 16, 28, 215), fill=True)
        self._text(image, 10, 5, summary, (255, 255, 255, 255), size=20)
        current_text = "TYPES: " + (", ".join("%s:%d" % item for item in current_counts.items()) or "NONE")
        self._text(image, 10, 27, current_text[:76], (190, 235, 255, 255), size=15)
        self._result_panel(image, detections)
        if draw_toolbar:
            self._toolbar(image, ui)
        return image
