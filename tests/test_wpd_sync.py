from pathlib import Path
from types import SimpleNamespace

import industrial_segpose.wpd_sync as wpd_sync


class FixedDateTime:
    @classmethod
    def now(cls):
        return cls()

    def strftime(self, _format):
        return "sync_20260806_120000"


def test_wpd_sync_uses_local_cache_and_returns_capture_folder(tmp_path, monkeypatch):
    def fake_run(command, **_kwargs):
        destination = Path(command[command.index("-Destination") + 1])
        (destination / "captures").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wpd_sync.sys, "platform", "win32")
    monkeypatch.setattr(wpd_sync, "datetime", FixedDateTime)
    monkeypatch.setattr(wpd_sync.subprocess, "run", fake_run)

    result = wpd_sync.sync_canmv_captures(tmp_path)

    assert result == tmp_path / "sync_20260806_120000" / "captures"
    assert result.is_dir()
