"""Windows WPD/MTP bridge for CanMV capture folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys


def _sync_canmv_folder(
    cache_root: str | Path,
    source_folder: str,
    device_name: str = "CanMV",
) -> Path:
    """Copy one application folder from a connected CanMV WPD device.

    CanMV is exposed by Windows as a WPD/MTP portable device rather than a
    mounted drive. Shell copy is therefore required before OpenCV can read the
    files using normal filesystem paths.
    """
    if sys.platform != "win32":
        raise OSError("CanMV WPD synchronization is only available on Windows")
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "sync_canmv_captures.ps1"
    if not script.is_file():
        raise FileNotFoundError(f"缺少WPD同步脚本：{script}")
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / datetime.now().strftime("sync_%Y%m%d_%H%M%S")
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(script), "-Destination", str(destination),
        "-DeviceName", device_name, "-SourceFolder", source_folder,
    ]
    startupinfo = None
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=90, check=False, startupinfo=startupinfo,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "WPD同步失败").strip()
        raise RuntimeError(detail)
    synchronized = destination / source_folder
    if not synchronized.is_dir():
        raise RuntimeError(f"CanMV同步完成但未生成{source_folder}目录")
    return synchronized


def sync_canmv_captures(cache_root: str | Path, device_name: str = "CanMV") -> Path:
    """Copy `/sdcard/industrial_vision/captures` from a connected CanMV."""
    return _sync_canmv_folder(cache_root, "captures", device_name)


def sync_canmv_diagnostics(cache_root: str | Path, device_name: str = "CanMV") -> Path:
    """Copy the latest on-device template segmentation diagnostics."""
    root = _sync_canmv_folder(cache_root, "diagnostics", device_name)
    latest = root / "latest_template"
    if not latest.is_dir():
        raise RuntimeError("K230中尚未生成模板分割诊断结果")
    return latest
