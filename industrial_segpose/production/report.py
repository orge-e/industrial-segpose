"""生产运行：记录实时采集和检测流程的结构化诊断报告。"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import platform
import threading
import traceback
from time import perf_counter
from typing import Any


class RuntimeReport:
    """Append-only JSONL diagnostics plus an easy-to-read session summary."""

    HIGH_FREQUENCY_STAGES = {
        "camera_capture", "frame_header", "frame_payload", "frame_decoded",
        "detection", "display",
    }

    def __init__(
        self,
        output_root: str | Path,
        session_name: str = "live",
        event_sample_interval: int = 30,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory = Path(output_root) / f"{session_name}_{timestamp}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.started_at = datetime.now()
        self.started_clock = perf_counter()
        self.lock = threading.Lock()
        self.event_sample_interval = max(1, int(event_sample_interval))
        self.counts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.duration_totals: dict[str, float] = defaultdict(float)
        self.duration_minimums: dict[str, float] = {}
        self.duration_maximums: dict[str, float] = {}
        self.last_error: dict[str, Any] | None = None
        self.closed = False
        self.record(
            "session",
            "started",
            details={
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        )

    def record(
        self,
        stage: str,
        status: str = "ok",
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_ms": round((perf_counter() - self.started_clock) * 1000.0, 3),
            "stage": str(stage),
            "status": str(status),
        }
        if duration_ms is not None:
            event["duration_ms"] = round(float(duration_ms), 3)
        if details:
            event["details"] = details
        with self.lock:
            if self.closed:
                return
            self.counts[str(stage)] += 1
            stage_count = self.counts[str(stage)]
            if status not in {"ok", "started", "stopped", "waiting"}:
                self.failures[str(stage)] += 1
            if duration_ms is not None:
                value = float(duration_ms)
                self.duration_totals[str(stage)] += value
                self.duration_minimums[str(stage)] = min(
                    value, self.duration_minimums.get(str(stage), value)
                )
                self.duration_maximums[str(stage)] = max(
                    value, self.duration_maximums.get(str(stage), value)
                )
            sampled = stage in self.HIGH_FREQUENCY_STAGES and status == "ok"
            if not sampled or stage_count == 1 or stage_count % self.event_sample_interval == 0:
                with self.events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_error(self, stage: str, error: BaseException, details: dict[str, Any] | None = None) -> None:
        payload = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            **(details or {}),
        }
        self.last_error = {"stage": stage, **payload}
        self.record(stage, "error", details=payload)

    def close(self, reason: str = "normal") -> Path:
        with self.lock:
            if self.closed:
                return self.directory
            ended_at = datetime.now()
            duration_seconds = max((ended_at - self.started_at).total_seconds(), 1e-9)
            stage_statistics = {}
            for stage, total in self.duration_totals.items():
                count = self.counts.get(stage, 0)
                if count:
                    stage_statistics[stage] = {
                        "count": count,
                        "average_ms": round(total / count, 3),
                        "maximum_ms": round(self.duration_maximums[stage], 3),
                        "minimum_ms": round(self.duration_minimums[stage], 3),
                    }
            received = self.counts.get("frame_decoded", 0)
            summary = {
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "duration_seconds": round(duration_seconds, 3),
                "stop_reason": reason,
                "decoded_frames": received,
                "average_decoded_fps": round(received / duration_seconds, 3),
                "stage_counts": dict(self.counts),
                "stage_failures": dict(self.failures),
                "stage_statistics": stage_statistics,
                "last_error": self.last_error,
                "fault_hint": self._fault_hint(),
            }
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.closed = True
        return self.directory

    def _fault_hint(self) -> str:
        if self.last_error:
            stage = self.last_error.get("stage", "unknown")
            hints = {
                "device_connect": "检查工业相机连接、视频源地址、驱动和供电。",
                "frame_header": "视频流不同步或连接中断，检查相机协议与网络稳定性。",
                "frame_payload": "图像帧未完整到达，检查网络带宽和图像尺寸。",
                "frame_decode": "收到的数据不是有效图像，检查相机编码与传输格式。",
                "network_wait": "电脑正在等待网络相机，检查相机地址、端口和防火墙。",
                "camera_capture": "相机读取失败，检查摄像头初始化、媒体缓冲区和供电。",
                "detection": "图像已到达但算法失败，检查模板库、ROI、图像尺寸和异常堆栈。",
                "display": "检测完成但界面刷新失败，检查UI线程和图像内存占用。",
            }
            return hints.get(stage, f"最后错误位于 {stage} 阶段，请查看 events.jsonl 中的 traceback。")
        if self.counts.get("device_connect", 0) == 0:
            return "没有设备连接记录，请检查工业相机或视频源是否已启动。"
        if self.counts.get("frame_decoded", 0) == 0:
            return "设备已连接但没有成功解码图像，优先检查帧发送与JPEG压缩。"
        return "各阶段未记录异常。"
