"""Low-allocation image-quality gate for CanMV/MicroPython frames."""


def _metric(statistics, name, default=None):
    method = getattr(statistics, name, None)
    if not callable(method):
        return default
    try:
        return float(method())
    except Exception:
        return default


class K230ImageQualityGate:
    """Evaluate exposure, local contrast and large illumination gradients.

    The implementation only requests image statistics and small ROI statistics;
    it does not allocate a second full-resolution image on the board.
    """

    def __init__(self, config=None):
        settings = dict(config or {})
        self.enabled = bool(settings.get("enabled", True))
        self.reject_bad_frames = bool(settings.get("reject_bad_frames", True))
        self.min_l_mean = float(settings.get("min_l_mean", 12.0))
        self.max_l_mean = float(settings.get("max_l_mean", 92.0))
        self.min_l_stdev = float(settings.get("min_l_stdev", 5.0))
        self.max_grid_spread = float(settings.get("max_grid_spread", 42.0))
        self.maximum_retries = max(0, int(settings.get("maximum_retries", 3)))
        self.bad_streak = 0

    @staticmethod
    def _statistics(frame, roi=None):
        method = getattr(frame, "get_statistics", None)
        if not callable(method):
            return None
        try:
            return method(roi=roi) if roi is not None else method()
        except TypeError:
            return method()
        except Exception:
            return None

    def evaluate(self, frame):
        if not self.enabled:
            self.bad_streak = 0
            return {
                "passed": True, "score": 1.0, "issues": [], "decision": "accept",
                "retry_index": 0, "maximum_retries": self.maximum_retries,
            }
        statistics = self._statistics(frame)
        if statistics is None:
            self.bad_streak += 1
            return self._report(None, None, None, ["statistics_unavailable"])
        mean = _metric(statistics, "l_mean")
        stdev = _metric(statistics, "l_stdev")
        grid_means = []
        try:
            width, height = int(frame.width()), int(frame.height())
            cell_width, cell_height = max(8, width // 3), max(8, height // 3)
            for row in range(3):
                for column in range(3):
                    x, y = column * cell_width, row * cell_height
                    roi_width = cell_width if column < 2 else width - x
                    roi_height = cell_height if row < 2 else height - y
                    value = _metric(self._statistics(frame, (x, y, roi_width, roi_height)), "l_mean")
                    if value is not None:
                        grid_means.append(value)
        except Exception:
            grid_means = []
        spread = max(grid_means) - min(grid_means) if len(grid_means) >= 4 else None
        issues = []
        if mean is None:
            issues.append("brightness_unavailable")
        elif mean < self.min_l_mean:
            issues.append("underexposed")
        elif mean > self.max_l_mean:
            issues.append("overexposed")
        if stdev is not None and stdev < self.min_l_stdev:
            issues.append("low_contrast_or_blur")
        if spread is not None and spread > self.max_grid_spread:
            issues.append("uneven_illumination")
        return self._report(mean, stdev, spread, issues)

    def _report(self, mean, stdev, spread, issues):
        passed = not issues or not self.reject_bad_frames
        if issues:
            self.bad_streak += 1
        else:
            self.bad_streak = 0
        decision = "accept"
        if issues and self.reject_bad_frames:
            decision = "retry" if self.bad_streak <= self.maximum_retries else "alarm"
        penalties = []
        if mean is not None:
            penalties.append(max(0.0, (self.min_l_mean - mean) / max(self.min_l_mean, 1.0)))
            penalties.append(max(0.0, (mean - self.max_l_mean) / max(100.0 - self.max_l_mean, 1.0)))
        if stdev is not None:
            penalties.append(max(0.0, (self.min_l_stdev - stdev) / max(self.min_l_stdev, 1.0)))
        if spread is not None:
            penalties.append(max(0.0, (spread - self.max_grid_spread) / max(self.max_grid_spread, 1.0)))
        score = max(0.0, min(1.0, 1.0 - max(penalties or [1.0 if issues else 0.0])))
        return {
            "passed": passed,
            "score": score,
            "l_mean": mean,
            "l_stdev": stdev,
            "grid_spread": spread,
            "issues": issues,
            "decision": decision,
            "retry_index": self.bad_streak if issues else 0,
            "maximum_retries": self.maximum_retries,
        }
