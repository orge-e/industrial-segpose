"""Small persistent exception reporter compatible with CanMV MicroPython."""

try:
    import uos as os
except ImportError:
    import os

try:
    import sys
except ImportError:  # pragma: no cover
    sys = None


def _ensure_dir(path):
    parts = str(path).replace("\\", "/").split("/")
    current = "/" if str(path).startswith("/") else ""
    for part in parts:
        if not part:
            continue
        current = current + part if current in ("", "/") else current + "/" + part
        try:
            os.stat(current)
        except OSError:
            os.mkdir(current)


def _read_index(path):
    try:
        with open(path, "r") as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return 0


def _write_index(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(str(int(value)))
    try:
        os.remove(path)
    except OSError:
        pass
    os.rename(temporary, path)


class ErrorReporter:
    """Write numbered reports without relying on an RTC clock."""

    def __init__(self, root="/sdcard/industrial_vision/logs"):
        self.root = str(root).replace("\\", "/").rstrip("/")

    @staticmethod
    def _write_exception(stream, error):
        printer = getattr(sys, "print_exception", None) if sys is not None else None
        if callable(printer):
            printer(error, stream)
            return
        try:
            import traceback
            traceback.print_exception(type(error), error, error.__traceback__, file=stream)
        except Exception:
            stream.write("%s: %s\n" % (type(error).__name__, error))

    def report(self, error, stage="unknown", context=None):
        try:
            _ensure_dir(self.root)
            index_path = self.root + "/error_index.txt"
            index = _read_index(index_path) + 1
            _write_index(index_path, index)
            report_path = self.root + "/error_%06d.log" % index
            latest_path = self.root + "/latest_error.log"
            for path in (report_path, latest_path):
                with open(path, "w") as stream:
                    stream.write("report_id: %06d\n" % index)
                    stream.write("stage: %s\n" % stage)
                    stream.write("error_type: %s\n" % type(error).__name__)
                    stream.write("error: %s\n" % error)
                    for key, value in sorted((context or {}).items()):
                        stream.write("%s: %s\n" % (key, value))
                    stream.write("traceback:\n")
                    self._write_exception(stream, error)
            return report_path
        except Exception as reporting_error:
            print("error reporter failed", reporting_error)
            return None

    def event(self, category, fields=None, maximum_bytes=262144):
        """Append a compact RTC-independent event and rotate at a size limit."""
        try:
            _ensure_dir(self.root)
            index_path = self.root + "/event_index.txt"
            index = _read_index(index_path) + 1
            _write_index(index_path, index)
            path = self.root + "/runtime_events.log"
            try:
                size = os.stat(path)[6]
            except OSError:
                size = 0
            if size >= int(maximum_bytes):
                previous = path + ".old"
                try:
                    os.remove(previous)
                except OSError:
                    pass
                os.rename(path, previous)
            values = ["event_id=%06d" % index, "category=%s" % category]
            for key, value in sorted((fields or {}).items()):
                values.append("%s=%s" % (key, value))
            with open(path, "a") as stream:
                stream.write("|".join(values) + "\n")
            return path
        except Exception as reporting_error:
            print("event logger failed", reporting_error)
            return None
