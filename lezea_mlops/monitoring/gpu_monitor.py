# lezea_mlops/monitoring/gpu_monitor.py
# GPU Monitor for LeZeA MLOps — hardened & tracker-aligned

from __future__ import annotations

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque
import json
import os

# Optional deps
GPU_LIBRARIES: List[str] = []

try:
    import GPUtil  # type: ignore
    GPU_LIBRARIES.append("GPUtil")
except Exception:
    GPUtil = None  # type: ignore

try:
    import pynvml  # type: ignore
    GPU_LIBRARIES.append("pynvml")
except Exception:
    pynvml = None  # type: ignore

try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        GPU_LIBRARIES.append("torch")
except Exception:
    torch = None  # type: ignore

try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

try:
    from prometheus_client import start_http_server, Gauge, CollectorRegistry  # type: ignore
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False


class GPUMonitor:
    """Comprehensive GPU + system monitor with safe fallbacks."""

    # Alert thresholds (can be tweaked at runtime if needed)
    THRESH_GPU_MEM_CRIT = 90.0
    THRESH_GPU_TEMP_HIGH = 80.0
    THRESH_GPU_UNDERUTIL = 20.0
    THRESH_CPU_HIGH = 90.0
    THRESH_SYS_MEM_HIGH = 90.0

    def __init__(self, sampling_interval: float = 1.0, history_size: int = 1000):
        """
        Args:
            sampling_interval: seconds between samples
            history_size: number of samples to retain in memory
        """
        self.sampling_interval = float(sampling_interval)
        self.history_size = int(history_size)

        # GPU detection
        self.gpu_library: Optional[str] = None
        self.device_count = 0
        self.devices_info: List[Dict[str, Any]] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.experiment_id: Optional[None | str] = None

        # Data buffers
        self._lock = threading.Lock()
        self.gpu_history: deque = deque(maxlen=self.history_size)
        self.system_history: deque = deque(maxlen=self.history_size)
        self.alerts: List[Dict[str, Any]] = []

        # Prometheus
        self._prom_registry: Optional[CollectorRegistry] = None
        self._prom_started = False
        # Gauges will be defined lazily when Prometheus starts
        self.g_gpu_util = None
        self.g_gpu_mem_used = None
        self.g_gpu_mem_pct = None
        self.g_gpu_temp = None
        self.g_sys_cpu = None
        self.g_sys_mem = None
        # Data-usage gauges (Grafana-friendly)
        self.g_du_usage_rate = None
        self.g_du_gini = None

        self._detect_gpus()

        print("🎮 GPU Monitor initialized")
        print(f"   GPUs detected: {self.device_count}")
        print(f"   Library: {self.gpu_library or 'none'}")
        print(f"   Sampling: {self.sampling_interval}s")

    # ---------------------------
    # Detection
    # ---------------------------
    def _detect_gpus(self) -> None:
        """Detect available GPUs and choose the best library."""
        # Try GPUtil first
        if "GPUtil" in GPU_LIBRARIES:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_library = "GPUtil"
                    self.device_count = len(gpus)
                    self.devices_info = [
                        {
                            "id": g.id,
                            "name": g.name,
                            "memory_total_mb": getattr(g, "memoryTotal", None),
                            "driver_version": getattr(g, "driver", None),
                        }
                        for g in gpus
                    ]
                    return
            except Exception as e:
                print(f"⚠️ GPUtil detection failed: {e}")

        # Then try NVML
        if "pynvml" in GPU_LIBRARIES:
            try:
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                if n > 0:
                    self.gpu_library = "pynvml"
                    self.device_count = n
                    self.devices_info = []
                    # driver once
                    try:
                        driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
                    except Exception:
                        driver = None
                    for i in range(n):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        self.devices_info.append(
                            {
                                "id": i,
                                "name": name,
                                "memory_total_mb": mem.total // (1024 ** 2),
                                "driver_version": driver,
                            }
                        )
                    # leave NVML initialized for runtime queries
                    return
            except Exception as e:
                print(f"⚠️ NVML detection failed: {e}")
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        # Then torch CUDA
        if "torch" in GPU_LIBRARIES:
            try:
                n = torch.cuda.device_count()
                if n > 0:
                    self.gpu_library = "torch"
                    self.device_count = n
                    self.devices_info = []
                    for i in range(n):
                        p = torch.cuda.get_device_properties(i)
                        self.devices_info.append(
                            {
                                "id": i,
                                "name": p.name,
                                "memory_total_mb": p.total_memory // (1024 ** 2),
                                "compute_capability": f"{p.major}.{p.minor}",
                            }
                        )
                    return
            except Exception as e:
                print(f"⚠️ PyTorch CUDA detection failed: {e}")

        # No GPUs or libs
        self.gpu_library = None
        self.device_count = 0
        self.devices_info = []
        print("ℹ️ No GPUs detected or GPU libs unavailable")

    # ---------------------------
    # Control
    # ---------------------------
    def start_monitoring(self, experiment_id: str | None = None, *, prometheus_port: Optional[int] = None) -> None:
        if self.is_monitoring:
            print("⚠️ GPU monitoring already active")
            return

        # Even if no GPUs, we still monitor system metrics
        self.experiment_id = experiment_id
        self.is_monitoring = True

        if PROM_AVAILABLE and prometheus_port is not None and not self._prom_started:
            self._start_prometheus(prometheus_port)

        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True, name="GPUMonitor")
        self.monitor_thread.start()
        print(f"📈 Started monitoring (GPUs: {self.device_count}, prom: {self._prom_started})")

    def stop_monitoring(self) -> None:
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        # NVML shutdown if we were using it
        if self.gpu_library == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("📊 Stopped GPU monitoring")

    # Simple health hook (not used by tracker health, but handy)
    def ping(self) -> bool:
        return True  # lightweight; monitoring is optional

    # ---------------------------
    # Prometheus (optional)
    # ---------------------------
    def _start_prometheus(self, port: int) -> None:
        try:
            self._prom_registry = CollectorRegistry()
            # Define gauges
            self.g_gpu_util = Gauge("lezea_gpu_utilization_percent", "GPU utilization", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_mem_used = Gauge("lezea_gpu_memory_used_mb", "GPU memory used (MB)", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_mem_pct = Gauge("lezea_gpu_memory_percent", "GPU memory percent", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_temp = Gauge("lezea_gpu_temperature_c", "GPU temperature (C)", ["gpu_id"], registry=self._prom_registry)
            self.g_sys_cpu = Gauge("lezea_system_cpu_percent", "System CPU percent", registry=self._prom_registry)
            self.g_sys_mem = Gauge("lezea_system_memory_percent", "System memory percent", registry=self._prom_registry)

            # Data-usage gauges (per split)
            self.g_du_usage_rate = Gauge(
                "lezea_data_usage_rate",
                "Unique coverage rate of a split (unique_seen/total)",
                ["split"],
                registry=self._prom_registry,
            )
            self.g_du_gini = Gauge(
                "lezea_data_usage_gini",
                "Gini coefficient of per-sample exposure distribution (split)",
                ["split"],
                registry=self._prom_registry,
            )

            start_http_server(port, registry=self._prom_registry)
            self._prom_started = True
            print(f"🛰️ Prometheus exporter started on :{port}")
        except Exception as e:
            print(f"⚠️ Prometheus init failed: {e}")
            self._prom_started = False
            self._prom_registry = None

    def _update_prometheus(self, gpu_metrics: List[Dict[str, Any]], sys_metrics: Dict[str, Any]) -> None:
        if not self._prom_started:
            return
        try:
            for m in gpu_metrics:
                gpu_id = str(m.get("device_id", -1))
                if "utilization_percent" in m and self.g_gpu_util:
                    self.g_gpu_util.labels(gpu_id).set(m["utilization_percent"])
                if "memory_used_mb" in m and self.g_gpu_mem_used:
                    self.g_gpu_mem_used.labels(gpu_id).set(m["memory_used_mb"])
                if "memory_percent" in m and self.g_gpu_mem_pct:
                    self.g_gpu_mem_pct.labels(gpu_id).set(m["memory_percent"])
                if "temperature_c" in m and isinstance(m["temperature_c"], (int, float)) and self.g_gpu_temp:
                    self.g_gpu_temp.labels(gpu_id).set(m["temperature_c"])
            if "cpu_percent" in sys_metrics and self.g_sys_cpu:
                self.g_sys_cpu.set(sys_metrics["cpu_percent"])
            if "memory_percent" in sys_metrics and self.g_sys_mem:
                self.g_sys_mem.set(sys_metrics["memory_percent"])
        except Exception:
            pass

    def update_data_usage_metrics(self, split: str, usage_rate: Optional[float], gini: Optional[float]) -> None:
        """
        Lightweight hook used by the tracker to push per-split data-usage stats
        into the same Prometheus exporter as GPU/system metrics.
        """
        if not self._prom_started:
            return
        try:
            if isinstance(usage_rate, (int, float)) and self.g_du_usage_rate:
                self.g_du_usage_rate.labels(str(split)).set(float(usage_rate))
            if isinstance(gini, (int, float)) and self.g_du_gini:
                self.g_du_gini.labels(str(split)).set(float(gini))
        except Exception:
            # Never break training loop because of metrics push
            pass

    # ---------------------------
    # Loop & collectors
    # ---------------------------
    def _monitoring_loop(self) -> None:
        while self.is_monitoring:
            t0 = time.time()
            try:
                gpu_metrics = self._collect_gpu_metrics()
                sys_metrics = self._collect_system_metrics()
                ts = datetime.now()

                with self._lock:
                    if gpu_metrics:
                        self.gpu_history.append({"timestamp": ts, "metrics": gpu_metrics})
                    if sys_metrics:
                        self.system_history.append({"timestamp": ts, "metrics": sys_metrics})
                    self._analyze_performance(gpu_metrics, sys_metrics)

                self._update_prometheus(gpu_metrics, sys_metrics)

            except Exception as e:
                print(f"❌ Error in monitor loop: {e}")

            # Sleep exactly to interval (account for work time)
            elapsed = time.time() - t0
            delay = max(0.0, self.sampling_interval - elapsed)
            time.sleep(delay)

    def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        if not self.gpu_library or self.device_count == 0:
            return []
        try:
            if self.gpu_library == "GPUtil":
                return self._collect_gputil_metrics()
            elif self.gpu_library == "pynvml":
                return self._collect_pynvml_metrics()
            elif self.gpu_library == "torch":
                return self._collect_torch_metrics()
        except Exception as e:
            print(f"❌ GPU metrics error: {e}")
        return []

    def _collect_gputil_metrics(self) -> List[Dict[str, Any]]:
        gpus = GPUtil.getGPUs()
        out: List[Dict[str, Any]] = []
        for g in gpus:
            mem_pct = (g.memoryUsed / g.memoryTotal) * 100 if getattr(g, "memoryTotal", 0) else 0.0
            out.append(
                {
                    "device_id": g.id,
                    "name": g.name,
                    "utilization_percent": round(getattr(g, "load", 0.0) * 100, 1),
                    "memory_used_mb": round(getattr(g, "memoryUsed", 0.0), 1),
                    "memory_total_mb": round(getattr(g, "memoryTotal", 0.0), 1),
                    "memory_free_mb": round(getattr(g, "memoryFree", 0.0), 1),
                    "memory_percent": round(mem_pct, 1),
                    "temperature_c": getattr(g, "temperature", None),
                }
            )
        return out

    def _collect_pynvml_metrics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(self.device_count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) // 1000
                except Exception:
                    power_w = None
                try:
                    gclk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
                    mclk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    gclk = mclk = None

                mem_pct = (mem.used / mem.total) * 100 if mem.total else 0.0
                out.append(
                    {
                        "device_id": i,
                        "name": pynvml.nvmlDeviceGetName(h).decode("utf-8"),
                        "utilization_percent": float(util.gpu),
                        "memory_utilization_percent": float(util.memory),
                        "memory_used_mb": round(mem.used / (1024 ** 2), 1),
                        "memory_total_mb": round(mem.total / (1024 ** 2), 1),
                        "memory_free_mb": round(mem.free / (1024 ** 2), 1),
                        "memory_percent": round(mem_pct, 1),
                        "temperature_c": temp,
                        "power_usage_w": power_w,
                        "graphics_clock_mhz": gclk,
                        "memory_clock_mhz": mclk,
                    }
                )
            except Exception as e:
                print(f"⚠️ NVML metrics failed for GPU {i}: {e}")
        return out

    def _collect_torch_metrics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(self.device_count):
            try:
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                mem_used = reserved or allocated
                mem_pct = (mem_used / total) * 100 if total else 0.0
                row = {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": round(allocated / (1024 ** 2), 1),
                    "memory_reserved_mb": round(reserved / (1024 ** 2), 1),
                    "memory_total_mb": round(total / (1024 ** 2), 1),
                    "memory_used_mb": round(mem_used / (1024 ** 2), 1),
                    "memory_percent": round(mem_pct, 1),
                }
                out.append(row)
            except Exception as e:
                print(f"⚠️ Torch metrics failed for GPU {i}: {e}")
        return out

    def _collect_system_metrics(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {}
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)  # non-blocking
            cpu_count = psutil.cpu_count()
            mem = psutil.virtual_memory()
            try:
                load1, load5, load15 = psutil.getloadavg()
            except Exception:
                load1 = load5 = load15 = 0.0
            # disk I/O (best effort; cumulative since boot)
            io_read_bytes = io_write_bytes = 0
            try:
                dio = psutil.disk_io_counters()
                if dio:
                    io_read_bytes = int(getattr(dio, "read_bytes", 0))
                    io_write_bytes = int(getattr(dio, "write_bytes", 0))
            except Exception:
                pass
            return {
                "cpu_percent": float(cpu_percent),
                "cpu_count": int(cpu_count) if cpu_count else None,
                "memory_percent": float(mem.percent),
                "memory_used_gb": round(mem.used / (1024 ** 3), 2),
                "memory_total_gb": round(mem.total / (1024 ** 3), 2),
                "load_avg_1m": float(load1),
                "load_avg_5m": float(load5),
                "load_avg_15m": float(load15),
                "io_read_bytes": io_read_bytes,
                "io_write_bytes": io_write_bytes,
            }
        except Exception as e:
            print(f"⚠️ System metrics failed: {e}")
            return {}

    # ---------------------------
    # Analysis & alerts
    # ---------------------------
    def _analyze_performance(self, gpus: List[Dict[str, Any]], system: Dict[str, Any]) -> None:
        try:
            now = datetime.now()
            # GPU alerts
            for m in gpus:
                did = m.get("device_id")
                mem_pct = float(m.get("memory_percent", 0.0))
                if mem_pct > self.THRESH_GPU_MEM_CRIT:
                    self._push_alert(now, "gpu_memory_high", f"GPU {did} memory critical: {mem_pct:.1f}%", device_id=did, value=mem_pct)
                temp = m.get("temperature_c")
                if isinstance(temp, (int, float)) and temp > self.THRESH_GPU_TEMP_HIGH:
                    self._push_alert(now, "gpu_temperature_high", f"GPU {did} temp high: {temp}°C", device_id=did, value=float(temp))
                util = float(m.get("utilization_percent", 0.0))
                if util < self.THRESH_GPU_UNDERUTIL and mem_pct > 50.0:
                    self._push_alert(
                        now,
                        "gpu_underutilized",
                        f"GPU {did} underutilized: {util:.1f}% (mem {mem_pct:.1f}%)",
                        device_id=did,
                        utilization=util,
                        memory=mem_pct,
                    )

            # System alerts
            cpu_pct = float(system.get("cpu_percent", 0.0))
            if cpu_pct > self.THRESH_CPU_HIGH:
                self._push_alert(now, "cpu_high", f"CPU usage high: {cpu_pct:.1f}%", value=cpu_pct)
            sys_mem_pct = float(system.get("memory_percent", 0.0))
            if sys_mem_pct > self.THRESH_SYS_MEM_HIGH:
                self._push_alert(now, "memory_high", f"System memory high: {sys_mem_pct:.1f}%", value=sys_mem_pct)

            # Trim alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
        except Exception:
            pass

    def _push_alert(self, ts: datetime, typ: str, message: str, **kwargs: Any) -> None:
        self.alerts.append({"timestamp": ts, "type": typ, "message": message, **kwargs})

    # ---------------------------
    # Public getters
    # ---------------------------
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Return the latest snapshot. Includes a convenient top-level summary:
          - cpu_percent (system)
          - memory_mb (sum of GPU used MB)
          - gpu_util_percent (avg across devices if available)
          - gpu_count (alias for device_count)
          - gpu_memory_mb (alias of memory_mb)
          - util (alias of gpu_util_percent)
          - ram_gb (system total RAM in GB)
          - io_read_bytes / io_write_bytes (cumulative, best-effort)
        """
        try:
            g = self._collect_gpu_metrics()
            s = self._collect_system_metrics()

            # Summary for tracker
            total_gpu_mem_mb = 0.0
            util_vals: List[float] = []
            for m in g:
                total_gpu_mem_mb += float(m.get("memory_used_mb", 0.0))
                if "utilization_percent" in m:
                    util_vals.append(float(m["utilization_percent"]))
            avg_util = sum(util_vals) / len(util_vals) if util_vals else 0.0

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "gpu_devices": g,
                "system": s,
                "device_count": self.device_count,
                # Tracker-friendly rollups:
                "cpu_percent": float(s.get("cpu_percent", 0.0)),
                "memory_mb": round(total_gpu_mem_mb, 2),
                "gpu_util_percent": round(avg_util, 2),
            }

            # Aliases for tracker/cost_model compatibility
            snapshot["gpu_count"] = snapshot["device_count"]
            snapshot["util"] = snapshot["gpu_util_percent"]
            snapshot["gpu_memory_mb"] = snapshot["memory_mb"]
            if "memory_total_gb" in s:
                snapshot["ram_gb"] = float(s["memory_total_gb"])
            if "io_read_bytes" in s:
                snapshot["io_read_bytes"] = int(s["io_read_bytes"])
            if "io_write_bytes" in s:
                snapshot["io_write_bytes"] = int(s["io_write_bytes"])

            return snapshot
        except Exception as e:
            print(f"❌ Snapshot failed: {e}")
            return {}

    def get_usage_history(self, minutes: int = 10) -> Dict[str, Any]:
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            with self._lock:
                rg = [e for e in self.gpu_history if e["timestamp"] > cutoff]
                rs = [e for e in self.system_history if e["timestamp"] > cutoff]
            return {
                "time_range_minutes": minutes,
                "gpu_history": [{"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in rg],
                "system_history": [{"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in rs],
                "sample_count": len(rg),
            }
        except Exception as e:
            print(f"❌ History failed: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        try:
            with self._lock:
                if not self.gpu_history:
                    sys_sum = self.system_history[-1]["metrics"] if self.system_history else {}
                    return {"error": "No GPU samples yet", "system": sys_sum}

                gpu_stats: Dict[int, Dict[str, List[float]]] = {}
                for entry in self.gpu_history:
                    for m in entry["metrics"]:
                        did = int(m.get("device_id", -1))
                        bucket = gpu_stats.setdefault(did, {"util": [], "mem": [], "temp": []})
                        if "utilization_percent" in m:
                            bucket["util"].append(float(m["utilization_percent"]))
                        if "memory_percent" in m:
                            bucket["mem"].append(float(m["memory_percent"]))
                        if "temperature_c" in m and isinstance(m["temperature_c"], (int, float)):
                            bucket["temp"].append(float(m["temperature_c"]))

                devices: Dict[str, Any] = {}
                for did, b in gpu_stats.items():
                    dev: Dict[str, Any] = {}
                    if b["util"]:
                        dev["avg_utilization"] = round(sum(b["util"]) / len(b["util"]), 1)
                        dev["max_utilization"] = round(max(b["util"]), 1)
                    if b["mem"]:
                        dev["avg_memory_usage"] = round(sum(b["mem"]) / len(b["mem"]), 1)
                        dev["max_memory_usage"] = round(max(b["mem"]), 1)
                    if b["temp"]:
                        dev["avg_temperature"] = round(sum(b["temp"]) / len(b["temp"]), 1)
                        dev["max_temperature"] = round(max(b["temp"]), 1)
                    devices[str(did)] = dev

                duration_min = len(self.gpu_history) * self.sampling_interval / 60.0
                recent_alerts = [a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(minutes=30)]
                return {
                    "monitoring_duration_minutes": round(duration_min, 2),
                    "sample_count": len(self.gpu_history),
                    "devices": devices,
                    "recent_alerts": len(recent_alerts),
                    "alert_types": sorted(set(a["type"] for a in recent_alerts)),
                }
        except Exception as e:
            print(f"❌ Summary failed: {e}")
            return {"error": str(e)}

    def get_optimization_recommendations(self) -> List[str]:
        recs: List[str] = []
        try:
            snap = self.get_current_usage()
            gpus = snap.get("gpu_devices", [])
            sysm = snap.get("system", {})

            if not gpus:
                return ["No GPU data available; utilization looks CPU-bound or system has no CUDA devices."]

            for m in gpus:
                did = m.get("device_id")
                util = float(m.get("utilization_percent", 0.0))
                mp = float(m.get("memory_percent", 0.0))
                temp = m.get("temperature_c", None)

                if util < 30 and mp > 60:
                    recs.append(
                        f"GPU {did}: low util ({util:.1f}%) but high memory ({mp:.1f}%). "
                        "Consider larger batch size, fused kernels, or overlapping data transfer/compute."
                    )
                if mp > 95:
                    recs.append(f"GPU {did}: memory critical ({mp:.1f}%). Enable gradient checkpointing or reduce batch size.")
                elif mp > 85:
                    recs.append(f"GPU {did}: memory high ({mp:.1f}%). Consider mixed precision or smaller activations.")
                if isinstance(temp, (int, float)) and temp > 85:
                    recs.append(f"GPU {did}: hot ({temp}°C). Improve cooling or reduce sustained load.")

            if len(gpus) > 1:
                utils = [float(m.get("utilization_percent", 0.0)) for m in gpus]
                if utils:
                    mean_u = sum(utils) / len(utils)
                    var = sum((u - mean_u) ** 2 for u in utils) / len(utils)
                    std = var ** 0.5
                    if std > 20:
                        recs.append("Uneven multi-GPU utilization. Check data sharding, DDP setup, and I/O bottlenecks.")

            if float(sysm.get("cpu_percent", 0.0)) > 90.0:
                recs.append("High CPU usage. Increase dataloader workers, enable pinned memory, or prefetch/cache datasets.")
            if float(sysm.get("memory_percent", 0.0)) > 90.0:
                recs.append("System RAM high. Stream data or reduce in-memory caching.")

            if not recs:
                recs.append("GPU utilization looks healthy — no immediate tuning needed.")
            return recs
        except Exception as e:
            print(f"❌ Recommendations failed: {e}")
            return ["Unable to generate recommendations due to an error."]

    # ---------------------------
    # Export
    # ---------------------------
    def export_monitoring_data(self, filepath: str, include_history: bool = True) -> None:
        try:
            with self._lock:
                export = {
                    "export_timestamp": datetime.now().isoformat(),
                    "experiment_id": self.experiment_id,
                    "device_info": self.devices_info,
                    "monitoring_config": {
                        "sampling_interval": self.sampling_interval,
                        "history_size": self.history_size,
                        "gpu_library": self.gpu_library,
                    },
                    "performance_summary": self.get_performance_summary(),
                    "current_usage": self.get_current_usage(),
                    "recommendations": self.get_optimization_recommendations(),
                    "recent_alerts": [
                        {"timestamp": a["timestamp"].isoformat(), **{k: v for k, v in a.items() if k != "timestamp"}}
                        for a in self.alerts[-20:]
                    ],
                }
                if include_history:
                    export["gpu_history"] = [
                        {"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in self.gpu_history
                    ]
                    export["system_history"] = [
                        {"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in self.system_history
                    ]
            with open(filepath, "w") as f:
                json.dump(export, f, indent=2)
            print(f"📁 Exported monitoring data to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to export monitoring data: {e}")

    # ---------------------------
    # Cleanup / Context
    # ---------------------------
    def cleanup(self) -> None:
        self.stop_monitoring()
        with self._lock:
            self.gpu_history.clear()
            self.system_history.clear()
            self.alerts.clear()
        print("🧹 GPU monitor cleaned up")

    def __enter__(self) -> "GPUMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
