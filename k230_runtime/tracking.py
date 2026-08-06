"""Small centroid tracker and conveyor counting line for MicroPython."""

import math


class CentroidCounter:
    def __init__(self, max_distance=90, max_missed=6, line_axis="x", line_position=520, direction=1):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.line_axis = line_axis
        self.line_position = float(line_position)
        self.direction = 1 if int(direction) >= 0 else -1
        self.next_id = 1
        self.tracks = {}
        self.counts = {}
        self.total_count = 0

    def _coordinate(self, detection):
        return detection["image_center"][0 if self.line_axis == "x" else 1]

    def _distance(self, track, detection):
        x, y = detection["image_center"]
        return math.sqrt((x - track["x"]) ** 2 + (y - track["y"]) ** 2)

    def _new_track(self, detection):
        track_id = self.next_id
        self.next_id += 1
        x, y = detection["image_center"]
        self.tracks[track_id] = {
            "x": float(x), "y": float(y), "missed": 0, "counted": False,
            "previous_coordinate": float(self._coordinate(detection)),
            "template_id": detection["template_id"],
        }
        return track_id

    def _crossed(self, previous, current):
        if self.direction > 0:
            return previous < self.line_position <= current
        return previous > self.line_position >= current

    def update(self, detections):
        unmatched = set(self.tracks.keys())
        for detection in detections:
            best_id = None
            best_distance = self.max_distance
            for track_id in list(unmatched):
                track = self.tracks[track_id]
                if track["template_id"] != detection["template_id"]:
                    continue
                distance = self._distance(track, detection)
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id is None:
                best_id = self._new_track(detection)
            else:
                unmatched.remove(best_id)
            track = self.tracks[best_id]
            current = float(self._coordinate(detection))
            counted_now = False
            if not track["counted"] and self._crossed(track["previous_coordinate"], current):
                track["counted"] = True
                counted_now = True
                name = detection.get("template_name", detection["template_id"])
                self.counts[name] = self.counts.get(name, 0) + 1
                self.total_count += 1
            track["previous_coordinate"] = current
            track["x"], track["y"] = map(float, detection["image_center"])
            track["missed"] = 0
            detection["track_id"] = best_id
            detection["counted"] = track["counted"]
            detection["emit"] = counted_now

        for track_id in list(unmatched):
            self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.max_missed:
                del self.tracks[track_id]
        return detections


class TrackedDetector:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker

    def detect(self, frame, timestamp_ms=0):
        return self.tracker.update(self.detector.detect(frame, timestamp_ms))
