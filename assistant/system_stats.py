from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import psutil


_CPU_COUNT = max(1, os.cpu_count() or 1)


@dataclass
class ProcessStat:
    pid: int
    name: str
    cpu_seconds: float
    memory_bytes: int
def _process_snapshot(proc: psutil.Process) -> ProcessStat | None:
    try:
        cpu_times = proc.cpu_times()
        cpu = float(cpu_times.user + cpu_times.system)
        mem_info = proc.memory_info()
        memory = int(getattr(mem_info, "private", getattr(mem_info, "rss", 0)))
        return ProcessStat(pid=proc.pid, name=proc.name(), cpu_seconds=cpu, memory_bytes=memory)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


class ProcessStatsSampler:
    def __init__(self, assistant_pid: int | None = None, sample_rate: float = 0.1):
        self.assistant_pid = int(assistant_pid or os.getpid())
        self.overlay_pid: int | None = None
        self.sample_rate = sample_rate
        self._last_sample_ts: float = 0.0
        self._prev_sample_ts: float = 0.0
        self._last_cpu_seconds: dict[int, float] = {}
        self._last_cpu_percent: dict[int, float] = {}
        self._last_stats: dict[int, ProcessStat] = {}

    def _resolve_overlay_pid(self) -> int | None:
        best_pid = None
        best_score = float("-inf")

        try:
            assistant = psutil.Process(self.assistant_pid)
            candidates = assistant.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        for child in candidates:
            if child.pid == self.assistant_pid:
                continue
            try:
                cmd = " ".join(child.cmdline()).lower()
                name = child.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            score = 0.0
            if "spawn_main" in cmd:
                score += 100.0
            if "overlay" in cmd:
                score += 50.0
            if name.startswith("python"):
                score += 10.0
            if score > best_score:
                best_score = score
                best_pid = child.pid
        return best_pid

    def sample(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._last_sample_ts < self.sample_rate and self._last_stats:
            return self._format_cached()

        self._last_sample_ts = now
        if self.overlay_pid is None:
            self.overlay_pid = self._resolve_overlay_pid()
        elif self._process_snapshot_by_pid(self.overlay_pid) is None:
            self.overlay_pid = self._resolve_overlay_pid()

        stats: dict[int, ProcessStat] = {}
        for pid in [self.assistant_pid, self.overlay_pid]:
            if pid is None:
                continue
            snap = self._process_snapshot_by_pid(pid)
            if snap is not None:
                stats[pid] = snap

        elapsed = max(1e-3, now - getattr(self, "_prev_sample_ts", now))
        cpu_percent: dict[int, float] = {}
        for pid, snap in stats.items():
            prev_cpu = self._last_cpu_seconds.get(pid)
            if prev_cpu is None:
                cpu_percent[pid] = 0.0
            else:
                cpu_percent[pid] = max(0.0, ((snap.cpu_seconds - prev_cpu) / elapsed) * 100.0 / _CPU_COUNT)
            self._last_cpu_seconds[pid] = snap.cpu_seconds
            self._last_stats[pid] = snap
            self._last_cpu_percent[pid] = cpu_percent[pid]

        self._prev_sample_ts = now
        return self._format_cached()

    def _process_snapshot_by_pid(self, pid: int) -> ProcessStat | None:
        try:
            return _process_snapshot(psutil.Process(pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    def _format_cached(self) -> dict[str, Any]:
        assistant = self._last_stats.get(self.assistant_pid)
        overlay = self._last_stats.get(self.overlay_pid or -1)
        return {
            "assistant": assistant,
            "overlay": overlay,
            "assistant_cpu": self._last_cpu_percent.get(self.assistant_pid, 0.0),
            "overlay_cpu": self._last_cpu_percent.get(self.overlay_pid or -1, 0.0),
        }
