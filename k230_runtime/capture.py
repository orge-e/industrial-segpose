"""RTC-independent image capture for the standalone K230 application."""

try:
    import uos as os
except ImportError:
    import os


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


def _read_number(path, default=0):
    try:
        with open(path, "r") as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return int(default)


def _write_number_atomic(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(str(int(value)))
    try:
        os.remove(path)
    except OSError:
        pass
    os.rename(temporary, path)


class SessionCaptureManager:
    """Save frames as ``session_NNNN/IMG_NNNNNN.jpg`` without using RTC time."""

    def __init__(self, root="/sdcard/industrial_vision/captures", quality=95):
        self.root = str(root).replace("\\", "/").rstrip("/")
        self.quality = int(quality)
        self.session_id = None
        self.sequence = 0

    def _open_session(self):
        if self.session_id is not None:
            return
        _ensure_dir(self.root)
        index_path = self.root + "/session_index.txt"
        self.session_id = _read_number(index_path, 0) + 1
        _write_number_atomic(index_path, self.session_id)
        _ensure_dir(self.root + "/session_%04d" % self.session_id)

    def capture(self, frame):
        self._open_session()
        self.sequence += 1
        directory = self.root + "/session_%04d" % self.session_id
        filename = "IMG_%06d.jpg" % self.sequence
        path = directory + "/" + filename
        try:
            frame.save(path, quality=self.quality)
        except TypeError:
            frame.save(path)
        return path


# Keep imports from early v0.9 development bundles working.
DateCaptureManager = SessionCaptureManager
