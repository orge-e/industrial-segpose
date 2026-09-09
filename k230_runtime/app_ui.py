"""Touch-first two-function UI for the 640x480 K230 screen."""


class IndustrialTouchUI:
    WIDTH = 640
    HEIGHT = 480

    def __init__(self, detection_overlay, touch=None, product_name="FlexPose Vision", subtitle="柔性工件定位系统", template_authoring_enabled=True):
        self.overlay = detection_overlay
        self.touch = touch
        self.product_name = str(product_name)
        self.subtitle = str(subtitle)
        self.template_authoring_enabled = bool(template_authoring_enabled)
        self.page = "home"
        self.message = "系统就绪"
        self.zoom = 1.0
        self.template_name = ""
        self.roi_points = []
        self.selected_template = 0
        self.template_preview = None
        self.preview_view = "overlay"
        self.auto_roi = None
        self.build_frame_id = None
        self.quality_status = "QUALITY:WAIT"
        self._pressed = False

    def open(self):
        if self.touch is None:
            from machine import TOUCH
            self.touch = TOUCH(0)

    @staticmethod
    def _inside(x, y, rect):
        left, top, width, height = rect
        return left <= x < left + width and top <= y < top + height

    def _keyboard_key(self, x, y):
        rows = ("1234567890", "QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM_-")
        starts = (45, 45, 72, 72)
        widths = (54, 54, 55, 55)
        tops = (145, 200, 255, 310)
        for row, left, key_width, top in zip(rows, starts, widths, tops):
            if top <= y < top + 46 and left <= x < left + key_width * len(row):
                index = int((x - left) / key_width)
                if index < len(row):
                    return row[index]
        return None

    def process_tap(self, x, y, template_count=0):
        x, y = int(x), int(y)
        if self.page == "home":
            if self._inside(x, y, (35, 125, 270, 260)):
                self.page = "templates"
                return "open_templates"
            if self._inside(x, y, (335, 125, 270, 260)):
                self.page = "detect"
                return "open_detect"
        elif self.page == "templates":
            if y < 60 and x < 110:
                self.page = "home"
                return "home"
            if self._inside(x, y, (25, 400, 150, 60)):
                if not self.template_authoring_enabled:
                    self.message = "模板由电脑端生成，请导入模板包"
                    return "authoring_disabled"
                self.page = "name"
                self.template_name = ""
                return "new_template"
            if self._inside(x, y, (190, 400, 150, 60)):
                return "toggle_template"
            if self._inside(x, y, (355, 400, 150, 60)):
                if not self.template_authoring_enabled:
                    self.message = "生产模式不允许删除模板"
                    return "authoring_disabled"
                return "remove_template"
            if 85 <= y < 370 and template_count:
                self.selected_template = min(template_count - 1, max(0, int((y - 85) / 58)))
                return "select_template"
        elif self.page == "name":
            if y < 60 and x < 110:
                self.page = "templates"
                return "cancel_name"
            key = self._keyboard_key(x, y)
            if key and len(self.template_name) < 20:
                self.template_name += key
                return "name_changed"
            if self._inside(x, y, (70, 380, 140, 60)):
                self.template_name = self.template_name[:-1]
                return "name_changed"
            if self._inside(x, y, (430, 380, 140, 60)) and self.template_name:
                self.page = "capture"
                self.roi_points = []
                self.message = "调整工件位置后点击拍摄"
                self.auto_roi = None
                self.template_preview = None
                self.preview_view = "overlay"
                self.build_frame_id = None
                return "open_template_capture"
        elif self.page == "capture":
            if y < 60 and x < 110:
                self.page = "name"
                return "cancel_capture"
            if self._inside(x, y, (180, 390, 280, 75)):
                self.page = "confirm"
                self.message = "正在冻结图像并自动分割"
                return "capture_template_preview"
        elif self.page == "roi":
            if y >= 425:
                if x < 180:
                    self.page = "templates"
                    self.roi_points = []
                    return "cancel_roi"
                if x > 460:
                    self.roi_points = []
                    return "retry_roi"
            else:
                self.roi_points.append((x, y))
                if len(self.roi_points) == 1:
                    self.message = "请点击另一个对角"
                    return "roi_first"
                self.page = "confirm"
                self.message = "检查黄色ROI后保存"
                return "roi_ready"
        elif self.page == "confirm":
            if y >= 400 and x < 105:
                self.page = "capture"
                self.roi_points = []
                self.auto_roi = None
                self.template_preview = None
                self.preview_view = "overlay"
                self.build_frame_id = None
                self.message = "请重新调整工件后拍摄"
                return "recapture_template"
            if y >= 400 and x < 205:
                self.preview_view = "original"
                self.message = "原图：确认冻结帧与工件姿态"
                return "preview_original"
            if y >= 400 and x < 320:
                self.preview_view = "mask"
                self.message = "分割：白色区域必须只覆盖目标工件"
                return "preview_mask"
            if y >= 400 and x < 420:
                self.preview_view = "overlay"
                self.message = "叠加：绿色区域就是将要保存的工件Mask"
                return "preview_overlay"
            if y >= 400 and x < 520:
                self.page = "roi"
                self.roi_points = []
                self.auto_roi = None
                self.template_preview = None
                return "manual_roi"
            if y >= 400:
                if self.template_preview is None:
                    self.message = "请先完成分割预览"
                    return "preview_invalid"
                if not self.template_preview.get("valid", False):
                    self.message = self.template_preview.get("message", "模板预览无效")
                    return "preview_invalid"
                return "save_template"
        elif self.page == "detect":
            if y >= 425:
                if x < 90:
                    self.page = "home"
                    return "home"
                if x < 180:
                    self.zoom = min(3.0, self.zoom + 0.25)
                    return "zoom_in"
                if x < 270:
                    self.zoom = max(1.0, self.zoom - 0.25)
                    return "zoom_out"
                if x < 360:
                    self.zoom = 1.0
                    return "fit"
                if x < 500:
                    return "capture_frame"
                return "reset_tracking"
        return None

    def poll(self, template_count=0):
        points = self.touch.read(1) if self.touch is not None else []
        if not points:
            self._pressed = False
            return None
        point = points[0]
        if getattr(point, "event", 0) not in (2, 3) or self._pressed:
            return None
        self._pressed = True
        return self.process_tap(point.x, point.y, template_count)

    def selected_roi(self):
        if len(self.roi_points) != 2:
            return self.auto_roi
        (x1, y1), (x2, y2) = self.roi_points
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), min(max(y1, y2), 424)
        if right - left < 24 or bottom - top < 24:
            return None
        return left, top, right - left, bottom - top

    @staticmethod
    def _text(image, x, y, size, text, color=(255, 255, 255, 255)):
        image.draw_string_advanced(x, y, size, text, color=color)

    def _button(self, image, rect, text, color=(25, 86, 180, 255), size=22):
        x, y, width, height = rect
        image.draw_rectangle(x, y, width, height, color=color, thickness=1, fill=True)
        image.draw_rectangle(x, y, width, height, color=(130, 185, 255, 255), thickness=2)
        self._text(image, x + 12, y + int((height - size) / 2), size, text)

    def _header(self, image, title, back=True):
        image.draw_rectangle(0, 0, self.WIDTH, 64, color=(15, 31, 58, 255), fill=True)
        if back:
            self._button(image, (10, 10, 90, 44), "< 返回", color=(37, 55, 82, 255), size=18)
        self._text(image, 125 if back else 24, 16, 28, title)

    def _render_home(self, image, library):
        image.clear()
        image.draw_rectangle(0, 0, self.WIDTH, self.HEIGHT, color=(10, 20, 38, 255), fill=True)
        self._header(image, self.product_name, back=False)
        self._text(image, 24, 76, 18, self.subtitle + " · 请选择功能", (170, 195, 225, 255))
        template_title = "1  建立模板" if self.template_authoring_enabled else "1  模板库"
        template_hint = "拍摄 / ROI / 自动提取" if self.template_authoring_enabled else "电脑端建立 / 模块端加载"
        self._button(image, (35, 125, 270, 260), template_title, color=(25, 93, 185, 255), size=30)
        self._button(image, (335, 125, 270, 260), "2  图像检测", color=(14, 130, 112, 255), size=30)
        self._text(image, 55, 315, 18, template_hint)
        self._text(image, 360, 315, 18, "识别 / 定位 / 角度 / 计数")
        self._text(image, 24, 430, 18, "模板数量: %d    安全模式: 不控制吸盘" % len(library.templates), (130, 205, 245, 255))

    def _render_templates(self, image, library):
        image.clear()
        image.draw_rectangle(0, 0, self.WIDTH, self.HEIGHT, color=(238, 243, 250, 255), fill=True)
        self._header(image, "模板库")
        for index, template in enumerate(library.templates[:5]):
            y = 85 + index * 58
            selected = index == self.selected_template
            color = (202, 225, 255, 255) if selected else (255, 255, 255, 255)
            image.draw_rectangle(25, y, 590, 48, color=color, fill=True)
            image.draw_rectangle(25, y, 590, 48, color=(95, 125, 160, 255), thickness=2)
            state = "启用" if template.get("enabled", True) else "停用"
            self._text(image, 42, y + 11, 20, "%d. %s" % (index + 1, template.get("name", "Template")), (20, 35, 55, 255))
            self._text(image, 520, y + 11, 18, state, (10, 120, 90, 255) if state == "启用" else (150, 70, 60, 255))
        if self.message:
            self._text(image, 25, 370, 16, self.message[:50], (35, 75, 120, 255))
        self._button(image, (25, 400, 150, 60), "+ 新建" if self.template_authoring_enabled else "电脑端模板", color=(25, 93, 185, 255) if self.template_authoring_enabled else (100, 112, 130, 255), size=18)
        self._button(image, (190, 400, 150, 60), "启用/停用", color=(64, 102, 160, 255), size=18)
        self._button(image, (355, 400, 150, 60), "移出模板库" if self.template_authoring_enabled else "只读模式", color=(155, 62, 62, 255) if self.template_authoring_enabled else (100, 112, 130, 255), size=18)

    def _render_name(self, image):
        image.clear()
        image.draw_rectangle(0, 0, self.WIDTH, self.HEIGHT, color=(235, 241, 249, 255), fill=True)
        self._header(image, "新模板：输入名称")
        image.draw_rectangle(50, 78, 540, 52, color=(255, 255, 255, 255), fill=True)
        image.draw_rectangle(50, 78, 540, 52, color=(60, 100, 150, 255), thickness=2)
        self._text(image, 65, 90, 24, self.template_name or "请输入英文或数字", (20, 35, 55, 255))
        rows = ("1234567890", "QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM_-")
        starts = (45, 45, 72, 72)
        widths = (54, 54, 55, 55)
        tops = (145, 200, 255, 310)
        for row, left, key_width, top in zip(rows, starts, widths, tops):
            for index, key in enumerate(row):
                self._button(image, (left + index * key_width, top, key_width - 4, 46), key, color=(52, 78, 112, 255), size=18)
        self._button(image, (70, 380, 140, 60), "退格", color=(110, 80, 70, 255))
        self._button(image, (430, 380, 140, 60), "下一步", color=(16, 130, 100, 255))

    def _render_capture(self, image, frame):
        image.clear()
        image.draw_image(frame, 0, 0)
        image.draw_rectangle(0, 0, self.WIDTH, 70, color=(10, 20, 35, 225), fill=True)
        self._button(image, (10, 10, 90, 44), "< 返回", color=(37, 55, 82, 255), size=18)
        self._text(image, 125, 12, 25, "实时取景：调整工件位置")
        self._text(image, 125, 40, 17, "拍摄后画面将冻结，后续建模只使用该帧", (180, 220, 255, 255))
        image.draw_rectangle(40, 90, 560, 275, color=(80, 210, 255, 255), thickness=3)
        self._button(image, (180, 390, 280, 75), "拍摄建模图像", color=(16, 135, 95, 255), size=26)

    def _render_roi(self, image, frame, confirm=False):
        image.clear()
        image.draw_image(frame, 0, 0)
        preview = self.template_preview if confirm else None
        mask = preview.get("mask") if preview is not None else None
        if self.preview_view == "mask" and mask is not None:
            image.clear()
            panel_w, panel_h = self.WIDTH, 335
            scale = min(float(panel_w) / max(mask.width(), 1), float(panel_h) / max(mask.height(), 1))
            draw_w = int(mask.width() * scale)
            draw_h = int(mask.height() * scale)
            image.draw_image(mask, (panel_w - draw_w) // 2, 60 + (panel_h - draw_h) // 2, x_scale=scale, y_scale=scale)
        elif self.preview_view == "overlay" and mask is not None:
            mask_rect = preview.get("mask_rect") or preview.get("blob_rect")
            overlay_layer = preview.get("mask_overlay")
            if mask_rect:
                try:
                    if overlay_layer is not None:
                        image.draw_image(
                            overlay_layer, int(mask_rect[0]), int(mask_rect[1]),
                            mask=mask, alpha=150,
                        )
                    else:
                        # Firmware fallback: the exact selected pixels remain
                        # visible even if an RGB565 overlay cannot be allocated.
                        image.draw_image(
                            mask, int(mask_rect[0]), int(mask_rect[1]),
                            mask=mask, alpha=130,
                        )
                except Exception:
                    pass
        image.draw_rectangle(0, 0, self.WIDTH, 60, color=(10, 20, 35, 220), fill=True)
        frame_label = "图像已冻结"
        if self.build_frame_id is not None:
            frame_label += "  BUILD FRAME:%06d" % int(self.build_frame_id)
        self._text(image, 16, 7, 20, frame_label, (100, 235, 255, 255))
        self._text(image, 16, 31, 17, self.message)
        if self.roi_points:
            image.draw_cross(self.roi_points[0][0], self.roi_points[0][1], color=(255, 220, 0, 255), size=16, thickness=4)
        roi = self.selected_roi()
        if roi:
            image.draw_rectangle(*roi, color=(255, 220, 0, 255), thickness=4)
        if confirm:
            if preview is not None:
                blob_rect = preview.get("blob_rect")
                if blob_rect and self.preview_view == "overlay":
                    image.draw_rectangle(*blob_rect, color=(0, 255, 80, 255), thickness=4)
                quality = preview.get("quality", "invalid")
                color = (70, 255, 130, 255) if quality == "good" else ((255, 215, 70, 255) if quality == "warning" else (255, 90, 90, 255))
                self._text(image, 16, 65, 18, "预览: %s  覆盖率: %.1f%%" % (quality, preview.get("coverage", 0.0) * 100.0), color)
                self._text(image, 16, 88, 16, preview.get("message", ""), color)
                self._text(image, 16, 110, 15, "模式: %s  LAB: %s" % (preview.get("mode", "-"), preview.get("threshold", [])), (230, 240, 255, 255))
                self._text(
                    image, 16, 132, 15,
                    "评分: %.2f  密度: %.2f  边界:%d  连通块:%d" % (
                        preview.get("score", 0.0), preview.get("density", 0.0),
                        preview.get("edge_touches", 0), preview.get("component_count", 0),
                    ),
                    (230, 240, 255, 255),
                )
            self._button(image, (0, 400, 105, 80), "重拍", color=(70, 90, 120, 255), size=17)
            self._button(image, (105, 400, 100, 80), "原图", color=(45, 90, 145, 255), size=17)
            self._button(image, (205, 400, 115, 80), "Mask", color=(45, 110, 105, 255), size=17)
            self._button(image, (320, 400, 100, 80), "叠加", color=(25, 125, 95, 255), size=17)
            self._button(image, (420, 400, 100, 80), "ROI", color=(100, 80, 60, 255), size=17)
            can_save = self.template_preview is not None and self.template_preview.get("valid", False)
            self._button(image, (520, 400, 120, 80), "保存" if can_save else "不合格", color=(16, 135, 95, 255) if can_save else (130, 65, 65, 255), size=17)
        else:
            self._button(image, (10, 425, 170, 50), "取消", color=(90, 70, 70, 255), size=18)
            self._button(image, (460, 425, 170, 50), "重选", color=(70, 90, 120, 255), size=18)

    def _render_detect(self, image, frame, detections, tracker, fps):
        self.overlay.draw(
            image, detections, tracker.total_count, tracker.counts, fps,
            frame=frame, zoom=self.zoom, draw_toolbar=False,
        )
        buttons = (
            (0, 90, "<主页"), (90, 180, "+放大"), (180, 270, "-缩小"),
            (270, 360, "适应"), (360, 500, "拍摄原图"), (500, 640, "刷新识别"),
        )
        for left, right, label in buttons:
            self._button(image, (left, 425, right - left, 55), label, color=(20, 46, 78, 235), size=18)
        self._text(image, 420, 8, 18, "缩放 %.2fx" % self.zoom)
        quality_color = (90, 255, 150, 255) if self.quality_status == "QUALITY:OK" else (255, 190, 80, 255)
        self._text(image, 12, 52, 16, self.quality_status, quality_color)
        if self.message:
            self._text(image, 12, 392, 17, self.message, (255, 225, 80, 255))

    def render(self, image, frame, detections, tracker, fps, library, frozen_frame=None):
        if self.page == "home":
            self._render_home(image, library)
        elif self.page == "templates":
            self._render_templates(image, library)
        elif self.page == "name":
            self._render_name(image)
        elif self.page == "capture":
            self._render_capture(image, frame)
        elif self.page in ("roi", "confirm"):
            source_frame = frozen_frame if frozen_frame is not None else frame
            self._render_roi(image, source_frame, confirm=self.page == "confirm")
        else:
            self._render_detect(image, frame, detections, tracker, fps)
        return image
