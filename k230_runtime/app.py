"""Production loop independent from the concrete camera and detector APIs."""

from shared_protocol import encode_message, make_heartbeat, make_pick_target


class K230Runtime:
    def __init__(self, config, camera, detector, transport, clock):
        self.config = config
        self.camera = camera
        self.detector = detector
        self.transport = transport
        self.clock = clock
        self.running = False
        self.frames = 0
        self.targets = 0
        self.rejected = 0
        self.last_heartbeat_ms = -1

    def _emit(self, message):
        self.transport.write(encode_message(message))

    def _heartbeat_if_due(self, now_ms):
        interval = int(self.config.get("heartbeat_interval_ms", 1000))
        if self.last_heartbeat_ms < 0 or now_ms - self.last_heartbeat_ms >= interval:
            self._emit(
                make_heartbeat(
                    self.config.get("device_id", "k230"),
                    now_ms,
                    "running" if self.running else "ready",
                    {"frames": self.frames, "targets": self.targets, "rejected": self.rejected},
                )
            )
            self.last_heartbeat_ms = now_ms

    def process_once(self):
        frame = self.camera.read()
        if frame is None:
            return False
        self.frames += 1
        now_ms = int(self.clock.millis())
        detections = self.detector.detect(frame, now_ms)
        threshold = float(self.config.get("minimum_confidence", 0.60))
        emit_ambiguous = bool(self.config.get("emit_ambiguous", False))
        for detection in detections:
            if not detection.get("emit", True):
                continue
            confidence = float(detection.get("confidence", 0.0))
            status = detection.get("status", "ready")
            if confidence < threshold or (status == "ambiguous" and not emit_ambiguous):
                self.rejected += 1
                continue
            message = make_pick_target(
                detection["track_id"],
                detection["template_id"],
                detection["template_name"],
                confidence,
                detection["image_center"],
                detection.get("pick_point", detection["image_center"]),
                detection["angle_deg"],
                now_ms,
                detection.get("world_point_mm"),
                detection.get("arrival_time_ms"),
                status,
            )
            self._emit(message)
            self.targets += 1
        self._heartbeat_if_due(now_ms)
        return True

    def run(self, maximum_frames=None):
        self.camera.open()
        self.transport.open()
        self.running = True
        processed = 0
        try:
            while maximum_frames is None or processed < maximum_frames:
                if not self.process_once():
                    break
                processed += 1
        finally:
            self.running = False
            self.camera.close()
            self.transport.close()
        return processed
