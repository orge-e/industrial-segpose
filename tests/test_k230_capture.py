from pathlib import Path

from k230_runtime.capture import SessionCaptureManager


class FakeFrame:
    def __init__(self):
        self.saved = []

    def save(self, path, quality=0):
        self.saved.append((path, quality))
        Path(path).write_bytes(b"jpeg")


def test_capture_uses_persistent_session_and_unique_filename(tmp_path):
    frame = FakeFrame()
    manager = SessionCaptureManager(tmp_path, quality=92)

    first = manager.capture(frame)
    second = manager.capture(frame)

    assert first.endswith("session_0001/IMG_000001.jpg")
    assert second.endswith("session_0001/IMG_000002.jpg")
    assert Path(first).read_bytes() == b"jpeg"
    assert frame.saved[0][1] == 92

    restarted = SessionCaptureManager(tmp_path)
    assert restarted.capture(frame).endswith("session_0002/IMG_000001.jpg")
