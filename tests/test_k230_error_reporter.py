from pathlib import Path

from k230_runtime.error_reporter import ErrorReporter


def test_error_reporter_writes_numbered_and_latest_traceback(tmp_path):
    reporter = ErrorReporter(tmp_path)
    try:
        raise ValueError("broken detector")
    except ValueError as error:
        first = reporter.report(error, "detect", {"frame": 12, "page": "detect"})

    text = Path(first).read_text(encoding="utf-8")
    assert first.endswith("error_000001.log")
    assert "stage: detect" in text
    assert "error: broken detector" in text
    assert "frame: 12" in text
    assert "ValueError" in text
    assert (tmp_path / "latest_error.log").read_text(encoding="utf-8") == text

    second = reporter.report(RuntimeError("render failed"), "render")
    assert second.endswith("error_000002.log")


def test_event_logger_appends_compact_records_and_rotates(tmp_path):
    reporter = ErrorReporter(tmp_path)
    path = reporter.event("image_quality", {"decision": "retry", "frame": 8}, maximum_bytes=80)
    reporter.event("image_quality", {"decision": "alarm", "issues": "underexposed"}, maximum_bytes=80)

    combined = Path(path).read_text(encoding="utf-8")
    previous = tmp_path / "runtime_events.log.old"
    if previous.exists():
        combined += previous.read_text(encoding="utf-8")
    assert "category=image_quality" in combined
    assert "decision=retry" in combined
    assert "decision=alarm" in combined
