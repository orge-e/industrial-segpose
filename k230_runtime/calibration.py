"""MicroPython-friendly planar coordinate conversion."""

try:
    import ujson as json
except ImportError:
    import json


class K230PlanarCalibration:
    def __init__(self, homography, tool_offset_mm=(0.0, 0.0), calibration_id="default"):
        if len(homography) != 3 or any(len(row) != 3 for row in homography):
            raise ValueError("homography must be 3x3")
        self.homography = [[float(value) for value in row] for row in homography]
        self.tool_offset_mm = [float(tool_offset_mm[0]), float(tool_offset_mm[1])]
        self.calibration_id = str(calibration_id)

    @classmethod
    def load(cls, path):
        with open(path, "r") as stream:
            payload = json.loads(stream.read())
        return cls(
            payload["homography"], payload.get("tool_offset_mm", (0.0, 0.0)),
            payload.get("calibration_id", "default"),
        )

    def pixel_to_global(self, point, axis_snapshot_mm=(0.0, 0.0)):
        x, y = float(point[0]), float(point[1])
        matrix = self.homography
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if abs(denominator) < 1e-9:
            raise ValueError("point projects to infinity")
        local_x = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator
        local_y = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator
        return [
            local_x + float(axis_snapshot_mm[0]) + self.tool_offset_mm[0],
            local_y + float(axis_snapshot_mm[1]) + self.tool_offset_mm[1],
        ]
