# lezea_mlops/monitoring/resource_logger.py
"""
Per-part resource logging for LeZeA.

Tracks CPU time (hours), memory GB-hours, approximate GPU-hours, and IO bytes
for any "part" (builder/tasker/algorithm/network/layer) identified by name.

Design notes
------------
- CPU: uses psutil.Process.cpu_percent(None) deltas. 100% == one full CPU core.
        cpu_hours += (cpu_percent/100) * elapsed_seconds / 3600
- Memory: instantaneous RSS is converted to GB-hours: (rss_bytes / 1e9) * elapsed_hours
- GPU: if pynvml present, attributes device utilization to processes by their
       used GPU memory share. gpu_hours += (attributed_util/100) * elapsed_hours
- IO: read_bytes + write_bytes delta since last sample

If psutil or pynvml are unavailable, the corresponding figures gracefully fall back to 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

# Optional NVML (GPU)
try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_NVML = False


@dataclass
class _PartState:
    name: str
    pid: int
    started: bool = False
    last_ts: Optional[float] = None
    # psutil rolling state
    _cpu_init: bool = False
    _last_io: Optional[Tuple[int, int]] = None  # (read_bytes, write_bytes)

    # accumulators
    cpu_hours: float = 0.0
    mem_gb_hours: float = 0.0
    gpu_hours: float = 0.0
    io_bytes: int = 0


class ResourceLogger:
    """
    ResourceLogger tracks resources per logical "part" (e.g., 'tasker#0').
    Typical usage:
        rl = ResourceLogger()
        rl.register_part("tasker#0")      # pid defaults to current process
        rl.start("tasker#0")
        # ... training ...
        snap = rl.sample("tasker#0")      # returns delta since last sample
        rl.stop("tasker#0")
        totals = rl.get_totals()
    """

    def __init__(self) -> None:
        self._parts: Dict[str, _PartState] = {}

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass

    # ---------------------- public API ----------------------

    def register_part(self, name: str, pid: Optional[int] = None) -> None:
        """Register a new part; pid defaults to current process."""
        if name in self._parts:
            return
        if pid is None:
            pid = psutil.Process().pid if _HAS_PSUTIL else 0
        self._parts[name] = _PartState(name=name, pid=int(pid))

    def start(self, name: str) -> None:
        s = self._require_part(name)
        if s.started:
            return
        s.started = True
        s.last_ts = time.time()
        if _HAS_PSUTIL:
            try:
                p = psutil.Process(s.pid)
                # initialize cpu_percent rolling window
                p.cpu_percent(None)
                s._cpu_init = True
                io = p.io_counters() if hasattr(p, "io_counters") else None
                if io:
                    s._last_io = (getattr(io, "read_bytes", 0), getattr(io, "write_bytes", 0))
            except Exception:
                pass

    def stop(self, name: str) -> None:
        s = self._require_part(name)
        if not s.started:
            return
        # final sample to flush
        self.sample(name)
        s.started = False
        s.last_ts = None

    def sample(self, name: str) -> Dict[str, float]:
        """
        Collect a delta sample since the last sample for this part.

        Returns keys:
        - cpu_h
        - mem_gb_h
        - gpu_h
        - io_bytes
        - elapsed_s
        """
        s = self._require_part(name)
        now = time.time()
        if not s.started:
            # If not started, nothing to add but return totals delta 0.
            return {"cpu_h": 0.0, "mem_gb_h": 0.0, "gpu_h": 0.0, "io_bytes": 0.0, "elapsed_s": 0.0}

        prev_ts = s.last_ts or now
        elapsed_s = max(0.0, now - prev_ts)
        elapsed_h = elapsed_s / 3600.0
        s.last_ts = now

        cpu_h_delta = 0.0
        mem_gb_h_delta = 0.0
        gpu_h_delta = 0.0
        io_delta = 0

        if _HAS_PSUTIL:
            try:
                p = psutil.Process(s.pid)
                # CPU percent (since last call)
                if s._cpu_init:
                    cpu_pct = float(p.cpu_percent(None))  # 100 == one full core
                else:
                    p.cpu_percent(None)
                    cpu_pct = 0.0
                    s._cpu_init = True
                cpu_h_delta = (cpu_pct / 100.0) * elapsed_h

                # Memory GB-hours (RSS)
                rss = float(p.memory_info().rss)
                mem_gb = rss / 1e9
                mem_gb_h_delta = mem_gb * elapsed_h

                # IO bytes delta
                io = p.io_counters() if hasattr(p, "io_counters") else None
                if io:
                    cur = (getattr(io, "read_bytes", 0), getattr(io, "write_bytes", 0))
                    if s._last_io:
                        io_delta = max(0, (cur[0] - s._last_io[0]) + (cur[1] - s._last_io[1]))
                    s._last_io = cur
            except Exception:
                pass

        # GPU hours approximation via NVML (optional)
        if _HAS_NVML:
            try:
                gpu_h_delta = self._attrib_gpu_hours_to_pid(s.pid, elapsed_h)
            except Exception:
                pass

        # accumulate
        s.cpu_hours += cpu_h_delta
        s.mem_gb_hours += mem_gb_h_delta
        s.gpu_hours += gpu_h_delta
        s.io_bytes += int(io_delta)

        return {
            "cpu_h": cpu_h_delta,
            "mem_gb_h": mem_gb_h_delta,
            "gpu_h": gpu_h_delta,
            "io_bytes": float(io_delta),
            "elapsed_s": elapsed_s,
        }

    def sample_all(self) -> Dict[str, Dict[str, float]]:
        """Sample all started parts and return a dict of deltas."""
        return {name: self.sample(name) for name, st in self._parts.items() if st.started}

    def get_totals(self) -> Dict[str, Dict[str, float]]:
        """Return cumulative totals for all parts."""
        out: Dict[str, Dict[str, float]] = {}
        for name, s in self._parts.items():
            out[name] = {
                "cpu_h": s.cpu_hours,
                "mem_gb_h": s.mem_gb_hours,
                "gpu_h": s.gpu_hours,
                "io_bytes": float(s.io_bytes),
            }
        return out

    # ---------------------- internals ----------------------

    def _attrib_gpu_hours_to_pid(self, pid: int, elapsed_h: float) -> float:
        """Attribute device GPU utilization to a pid proportional to used memory."""
        if not _HAS_NVML:
            return 0.0
        total_util = 0.0
        total_attr = 0.0
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu  # 0..100
                total_util += float(util)
                # running processes with mem usage
                mem_sum = 0
                target_mem = 0
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)  # type: ignore[attr-defined]
                except Exception:
                    # fall back to v2/v1 or graphics processes
                    try:
                        procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(h)  # type: ignore[attr-defined]
                    except Exception:
                        procs = []
                for pr in procs:
                    used = float(getattr(pr, "usedGpuMemory", 0.0))
                    mem_sum += max(0.0, used)
                    if int(getattr(pr, "pid", -1)) == pid:
                        target_mem = max(0.0, used)
                if mem_sum > 0 and util > 0:
                    frac = target_mem / mem_sum
                    total_attr += (util * frac)
            # convert to hours
            gpu_h = (total_attr / 100.0) * elapsed_h
            return float(gpu_h)
        except Exception:
            return 0.0

    def _require_part(self, name: str) -> _PartState:
        if name not in self._parts:
            self.register_part(name)  # auto-register on first touch
        return self._parts[name]
