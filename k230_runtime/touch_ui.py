"""Touch input state and large on-screen controls for the K230 LCD."""


class TouchUI:
    TOOLBAR_Y = 430

    def __init__(self, width=640, height=480, touch=None):
        self.width = int(width)
        self.height = int(height)
        self.touch = touch
        self.zoom = 1.0
        self.mode = "detect"
        self.roi_points = []
        self.message = "READY"
        self._pressed = False

    def open(self):
        if self.touch is None:
            from machine import TOUCH
            self.touch = TOUCH(0)

    def process_tap(self, x, y):
        x, y = int(x), int(y)
        if y >= self.TOOLBAR_Y:
            if x < 110:
                self.zoom = min(3.0, self.zoom + 0.25)
                return "zoom_in"
            if x < 220:
                self.zoom = max(1.0, self.zoom - 0.25)
                return "zoom_out"
            if x < 330:
                self.zoom = 1.0
                return "fit"
            if x < 500:
                self.mode = "select_roi"
                self.roi_points = []
                self.message = "SELECT ROI: 2 CORNERS"
                return "new_template"
            self.message = "COUNT RESET"
            return "reset_count"
        if self.mode == "select_roi":
            self.roi_points.append((x, y))
            if len(self.roi_points) == 1:
                self.message = "SELECT OPPOSITE CORNER"
                return "roi_first"
            self.mode = "building"
            self.message = "BUILDING TEMPLATE"
            return "roi_complete"
        return "tap"

    def poll(self):
        points = self.touch.read(1) if self.touch is not None else []
        if not points:
            self._pressed = False
            return None
        point = points[0]
        # Yahboom reports 2/3 while touching. Trigger only once per press.
        if getattr(point, "event", 0) not in (2, 3):
            return None
        if self._pressed:
            return None
        self._pressed = True
        action = self.process_tap(point.x, point.y)
        return action, int(point.x), int(point.y)

    def selected_roi(self):
        if len(self.roi_points) != 2:
            return None
        (x1, y1), (x2, y2) = self.roi_points
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), min(max(y1, y2), self.TOOLBAR_Y - 1)
        if right - left < 24 or bottom - top < 24:
            return None
        return left, top, right - left, bottom - top
