from k230_runtime.image_quality import K230ImageQualityGate


class Statistics:
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def l_mean(self):
        return self.mean

    def l_stdev(self):
        return self.stdev


class Frame:
    def __init__(self, mean, stdev, grid=None):
        self.mean = mean
        self.stdev = stdev
        self.grid = list(grid or [mean] * 9)
        self.roi_calls = 0

    def width(self):
        return 320

    def height(self):
        return 240

    def get_statistics(self, roi=None):
        if roi is None:
            return Statistics(self.mean, self.stdev)
        value = self.grid[min(self.roi_calls, len(self.grid) - 1)]
        self.roi_calls += 1
        return Statistics(value, self.stdev)


def test_quality_gate_accepts_well_exposed_textured_frame():
    gate = K230ImageQualityGate({"maximum_retries": 2})

    report = gate.evaluate(Frame(55, 12, [50, 51, 52, 53, 54, 55, 56, 57, 58]))

    assert report["passed"] is True
    assert report["decision"] == "accept"
    assert report["issues"] == []
    assert report["grid_spread"] == 8


def test_quality_gate_retries_then_alarms_and_recovers():
    gate = K230ImageQualityGate({"maximum_retries": 2})
    dark = Frame(5, 2)

    assert gate.evaluate(dark)["decision"] == "retry"
    assert gate.evaluate(Frame(5, 2))["decision"] == "retry"
    alarm = gate.evaluate(Frame(5, 2))
    assert alarm["decision"] == "alarm"
    assert "underexposed" in alarm["issues"]

    recovered = gate.evaluate(Frame(50, 10))
    assert recovered["decision"] == "accept"
    assert recovered["retry_index"] == 0


def test_quality_gate_detects_large_illumination_gradient():
    gate = K230ImageQualityGate({"max_grid_spread": 25})
    report = gate.evaluate(Frame(55, 10, [20, 25, 30, 45, 50, 55, 65, 70, 80]))
    assert report["passed"] is False
    assert "uneven_illumination" in report["issues"]
