# Enhanced GPU Monitor for LeZeA MLOps with component-level attribution

from __future__ import annotations

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import deque, defaultdict
import json
import os
from dataclasses import dataclass, field
from enum import Enum
import uuid

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
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, CollectorRegistry  # type: ignore
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False


class ComponentType(Enum):
    """LeZeA component types for resource attribution"""
    BUILDER = "builder"
    TASKER = "tasker" 
    ALGORITHM = "algorithm"
    NETWORK = "network"
    LAYER = "layer"
    GLOBAL = "global"


@dataclass
class ComponentMetrics:
    """Resource metrics for a specific component"""
    component_id: str
    component_type: ComponentType
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_util_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    operations_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    cumulative_time_seconds: float = 0.0


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: datetime
    global_metrics: Dict[str, Any]
    component_metrics: Dict[str, ComponentMetrics]
    gpu_devices: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]


class GPUMonitor:
    """Enhanced GPU + system monitor with LeZeA component-level attribution."""

    # Alert thresholds (can be tweaked at runtime if needed)
    THRESH_GPU_MEM_CRIT = 90.0
    THRESH_GPU_TEMP_HIGH = 80.0
    THRESH_GPU_UNDERUTIL = 20.0
    THRESH_CPU_HIGH = 90.0
    THRESH_SYS_MEM_HIGH = 90.0
    THRESH_COMPONENT_MEM_HIGH = 1000.0  # MB per component
    THRESH_COMPONENT_IMBALANCE = 50.0  # % difference between components

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

        # NEW: Component-level tracking
        self.active_components: Dict[str, ComponentMetrics] = {}
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.component_start_times: Dict[str, datetime] = {}
        self.component_baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Resource attribution tracking
        self.resource_snapshots: deque = deque(maxlen=self.history_size)
        self.component_io_baselines: Dict[str, Tuple[int, int]] = {}  # (read, write) baseline
        
        # Component operation tracking
        self.component_operations: Dict[str, int] = defaultdict(int)
        self.component_peak_usage: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Prometheus
        self._prom_registry: Optional[CollectorRegistry] = None
        self._prom_started = False
        # Original gauges
        self.g_gpu_util = None
        self.g_gpu_mem_used = None
        self.g_gpu_mem_pct = None
        self.g_gpu_temp = None
        self.g_sys_cpu = None
        self.g_sys_mem = None
        # Data-usage gauges (Grafana-friendly)
        self.g_du_usage_rate = None
        self.g_du_gini = None
        # NEW: Component-level gauges
        self.g_comp_cpu = None
        self.g_comp_memory = None
        self.g_comp_gpu_memory = None
        self.g_comp_gpu_util = None
        self.g_comp_operations = None
        self.g_comp_active_count = None

        self._detect_gpus()

        print("🎮 Enhanced GPU Monitor initialized")
        print(f"   GPUs detected: {self.device_count}")
        print(f"   Library: {self.gpu_library or 'none'}")
        print(f"   Sampling: {self.sampling_interval}s")
        print(f"   Component tracking: enabled")

    # ---------------------------
    # Detection (unchanged)
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
    # NEW: Component management
    # ---------------------------
    def register_component(
        self, 
        component_id: str, 
        component_type: ComponentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a component for resource tracking."""
        with self._lock:
            if component_id in self.active_components:
                print(f"⚠️ Component {component_id} already registered")
                return
            
            # Capture baseline metrics
            current_system = self._collect_system_metrics()
            baseline = {
                "cpu_percent": current_system.get("cpu_percent", 0.0),
                "memory_used_gb": current_system.get("memory_used_gb", 0.0),
                "io_read_bytes": current_system.get("io_read_bytes", 0),
                "io_write_bytes": current_system.get("io_write_bytes", 0),
            }
            
            component = ComponentMetrics(
                component_id=component_id,
                component_type=component_type,
                start_time=datetime.now()
            )
            
            self.active_components[component_id] = component
            self.component_start_times[component_id] = datetime.now()
            self.component_baseline_metrics[component_id] = baseline
            self.component_io_baselines[component_id] = (
                baseline["io_read_bytes"], 
                baseline["io_write_bytes"]
            )
            
            print(f"📊 Registered {component_type.value} component: {component_id}")

    def unregister_component(self, component_id: str) -> Optional[ComponentMetrics]:
        """Unregister a component and return its final metrics."""
        with self._lock:
            if component_id not in self.active_components:
                return None
            
            # Final update
            self._update_component_metrics(component_id)
            
            # Archive final metrics
            final_metrics = self.active_components.pop(component_id)
            final_metrics.last_update = datetime.now()
            final_metrics.cumulative_time_seconds = (
                final_metrics.last_update - final_metrics.start_time
            ).total_seconds()
            
            # Cleanup
            self.component_start_times.pop(component_id, None)
            self.component_baseline_metrics.pop(component_id, None)
            self.component_io_baselines.pop(component_id, None)
            
            print(f"📊 Unregistered component: {component_id}")
            return final_metrics

    def _update_component_metrics(self, component_id: str) -> None:
        """Update metrics for a specific component."""
        if component_id not in self.active_components:
            return
        
        try:
            component = self.active_components[component_id]
            baseline = self.component_baseline_metrics.get(component_id, {})
            io_baseline = self.component_io_baselines.get(component_id, (0, 0))
            
            # Get current system state
            current_system = self._collect_system_metrics()
            current_gpu = self._collect_gpu_metrics()
            
            # Calculate component-specific attribution
            # Note: This is a simplified attribution model
            # In practice, you might use more sophisticated profiling
            
            total_components = len(self.active_components)
            if total_components == 0:
                return
            
            # Simple even distribution model (can be enhanced with actual profiling)
            cpu_delta = max(0, current_system.get("cpu_percent", 0) - baseline.get("cpu_percent", 0))
            memory_delta = max(0, current_system.get("memory_used_gb", 0) - baseline.get("memory_used_gb", 0))
            
            # Attribute proportionally to active components
            component.cpu_percent = cpu_delta / total_components
            component.memory_mb = (memory_delta * 1024) / total_components  # Convert GB to MB
            
            # GPU attribution (simplified)
            if current_gpu:
                total_gpu_memory = sum(float(g.get("memory_used_mb", 0)) for g in current_gpu)
                total_gpu_util = sum(float(g.get("utilization_percent", 0)) for g in current_gpu)
                
                component.gpu_memory_mb = total_gpu_memory / total_components
                component.gpu_util_percent = total_gpu_util / max(len(current_gpu), 1) / total_components
            
            # I/O attribution
            current_io_read = current_system.get("io_read_bytes", 0)
            current_io_write = current_system.get("io_write_bytes", 0)
            
            io_read_delta = max(0, current_io_read - io_baseline[0])
            io_write_delta = max(0, current_io_write - io_baseline[1])
            
            component.io_read_bytes = io_read_delta // total_components
            component.io_write_bytes = io_write_delta // total_components
            
            # Update operation count
            component.operations_count = self.component_operations.get(component_id, 0)
            component.last_update = datetime.now()
            
            # Track peak usage
            peaks = self.component_peak_usage[component_id]
            peaks["cpu_percent"] = max(peaks.get("cpu_percent", 0), component.cpu_percent)
            peaks["memory_mb"] = max(peaks.get("memory_mb", 0), component.memory_mb)
            peaks["gpu_memory_mb"] = max(peaks.get("gpu_memory_mb", 0), component.gpu_memory_mb)
            peaks["gpu_util_percent"] = max(peaks.get("gpu_util_percent", 0), component.gpu_util_percent)
            
            # Store in history
            self.component_history[component_id].append({
                "timestamp": datetime.now(),
                "metrics": {
                    "cpu_percent": component.cpu_percent,
                    "memory_mb": component.memory_mb,
                    "gpu_memory_mb": component.gpu_memory_mb,
                    "gpu_util_percent": component.gpu_util_percent,
                    "io_read_bytes": component.io_read_bytes,
                    "io_write_bytes": component.io_write_bytes,
                    "operations_count": component.operations_count
                }
            })
            
        except Exception as e:
            print(f"❌ Failed to update component {component_id} metrics: {e}")

    def record_component_operation(self, component_id: str, operation_count: int = 1) -> None:
        """Record operations performed by a component."""
        with self._lock:
            if component_id in self.active_components:
                self.component_operations[component_id] += operation_count

    def get_component_metrics(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get current metrics for a specific component."""
        with self._lock:
            if component_id not in self.active_components:
                return None
            
            component = self.active_components[component_id]
            peaks = self.component_peak_usage.get(component_id, {})
            history = list(self.component_history.get(component_id, []))
            
            return {
                "component_id": component_id,
                "component_type": component.component_type.value,
                "current_metrics": {
                    "cpu_percent": component.cpu_percent,
                    "memory_mb": component.memory_mb,
                    "gpu_memory_mb": component.gpu_memory_mb,
                    "gpu_util_percent": component.gpu_util_percent,
                    "io_read_bytes": component.io_read_bytes,
                    "io_write_bytes": component.io_write_bytes,
                    "operations_count": component.operations_count
                },
                "peak_usage": peaks,
                "start_time": component.start_time.isoformat(),
                "last_update": component.last_update.isoformat(),
                "uptime_seconds": (datetime.now() - component.start_time).total_seconds(),
                "history_samples": len(history)
            }

    def get_component_analysis(self) -> Dict[str, Any]:
        """Analyze resource usage across all components."""
        with self._lock:
            if not self.active_components:
                return {"error": "No active components"}
            
            analysis = {
                "total_components": len(self.active_components),
                "component_types": {},
                "resource_distribution": {
                    "cpu_percent": {},
                    "memory_mb": {},
                    "gpu_memory_mb": {},
                    "gpu_util_percent": {}
                },
                "top_consumers": {
                    "cpu": [],
                    "memory": [],
                    "gpu_memory": [],
                    "gpu_util": []
                },
                "imbalance_detected": False,
                "recommendations": []
            }
            
            # Group by component type
            type_stats = defaultdict(lambda: {"count": 0, "total_cpu": 0, "total_memory": 0})
            
            components_data = []
            for comp_id, component in self.active_components.items():
                comp_type = component.component_type.value
                type_stats[comp_type]["count"] += 1
                type_stats[comp_type]["total_cpu"] += component.cpu_percent
                type_stats[comp_type]["total_memory"] += component.memory_mb
                
                components_data.append({
                    "id": comp_id,
                    "type": comp_type,
                    "cpu": component.cpu_percent,
                    "memory": component.memory_mb,
                    "gpu_memory": component.gpu_memory_mb,
                    "gpu_util": component.gpu_util_percent
                })
            
            # Calculate type averages
            for comp_type, stats in type_stats.items():
                count = stats["count"]
                analysis["component_types"][comp_type] = {
                    "count": count,
                    "avg_cpu_percent": stats["total_cpu"] / count,
                    "avg_memory_mb": stats["total_memory"] / count
                }
            
            # Resource distribution
            for metric in ["cpu", "memory", "gpu_memory", "gpu_util"]:
                values = [comp[metric] for comp in components_data]
                if values:
                    analysis["resource_distribution"][f"{metric}_percent"]["min"] = min(values)
                    analysis["resource_distribution"][f"{metric}_percent"]["max"] = max(values)
                    analysis["resource_distribution"][f"{metric}_percent"]["avg"] = sum(values) / len(values)
                    analysis["resource_distribution"][f"{metric}_percent"]["std"] = (
                        sum((v - analysis["resource_distribution"][f"{metric}_percent"]["avg"]) ** 2 for v in values) / len(values)
                    ) ** 0.5
            
            # Top consumers
            for metric, key in [("cpu", "cpu"), ("memory", "memory"), ("gpu_memory", "gpu_memory"), ("gpu_util", "gpu_util")]:
                sorted_components = sorted(components_data, key=lambda x: x[key], reverse=True)
                analysis["top_consumers"][metric] = [
                    {"component_id": comp["id"], "type": comp["type"], "value": comp[key]}
                    for comp in sorted_components[:5]
                ]
            
            # Imbalance detection
            for metric in ["cpu", "memory", "gpu_memory", "gpu_util"]:
                values = [comp[metric] for comp in components_data if comp[metric] > 0]
                if len(values) > 1:
                    max_val = max(values)
                    min_val = min(values)
                    if max_val > 0 and (max_val - min_val) / max_val * 100 > self.THRESH_COMPONENT_IMBALANCE:
                        analysis["imbalance_detected"] = True
                        analysis["recommendations"].append(
                            f"High {metric} imbalance detected: {max_val:.1f} vs {min_val:.1f}. "
                            f"Consider load balancing or component optimization."
                        )
            
            # Resource usage recommendations
            for comp in components_data:
                if comp["memory"] > self.THRESH_COMPONENT_MEM_HIGH:
                    analysis["recommendations"].append(
                        f"Component {comp['id']} ({comp['type']}) using {comp['memory']:.1f}MB memory. "
                        f"Consider memory optimization."
                    )
            
            return analysis

    # ---------------------------
    # Control (enhanced)
    # ---------------------------
    def start_monitoring(self, experiment_id: str | None = None, *, prometheus_port: Optional[int] = None) -> None:
        if self.is_monitoring:
            print("⚠️ GPU monitoring already active")
            return

        self.experiment_id = experiment_id
        self.is_monitoring = True

        if PROM_AVAILABLE and prometheus_port is not None and not self._prom_started:
            self._start_prometheus(prometheus_port)

        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True, name="GPUMonitor")
        self.monitor_thread.start()
        print(f"📈 Started enhanced monitoring (GPUs: {self.device_count}, components: enabled, prom: {self._prom_started})")

    def stop_monitoring(self) -> None:
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        
        # Unregister all components
        with self._lock:
            active_components = list(self.active_components.keys())
        for comp_id in active_components:
            self.unregister_component(comp_id)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # NVML shutdown if we were using it
        if self.gpu_library == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("📊 Stopped enhanced GPU monitoring")

    # Simple health hook
    def ping(self) -> bool:
        return True

    # ---------------------------
    # Enhanced Prometheus
    # ---------------------------
    def _start_prometheus(self, port: int) -> None:
        try:
            self._prom_registry = CollectorRegistry()
            
            # Original gauges
            self.g_gpu_util = Gauge("lezea_gpu_utilization_percent", "GPU utilization", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_mem_used = Gauge("lezea_gpu_memory_used_mb", "GPU memory used (MB)", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_mem_pct = Gauge("lezea_gpu_memory_percent", "GPU memory percent", ["gpu_id"], registry=self._prom_registry)
            self.g_gpu_temp = Gauge("lezea_gpu_temperature_c", "GPU temperature (C)", ["gpu_id"], registry=self._prom_registry)
            self.g_sys_cpu = Gauge("lezea_system_cpu_percent", "System CPU percent", registry=self._prom_registry)
            self.g_sys_mem = Gauge("lezea_system_memory_percent", "System memory percent", registry=self._prom_registry)

            # Data-usage gauges
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

            # NEW: Component-level gauges
            self.g_comp_cpu = Gauge("lezea_component_cpu_percent", "Component CPU usage", ["component_id", "component_type"], registry=self._prom_registry)
            self.g_comp_memory = Gauge("lezea_component_memory_mb", "Component memory usage (MB)", ["component_id", "component_type"], registry=self._prom_registry)
            self.g_comp_gpu_memory = Gauge("lezea_component_gpu_memory_mb", "Component GPU memory (MB)", ["component_id", "component_type"], registry=self._prom_registry)
            self.g_comp_gpu_util = Gauge("lezea_component_gpu_util_percent", "Component GPU utilization", ["component_id", "component_type"], registry=self._prom_registry)
            self.g_comp_operations = Counter("lezea_component_operations_total", "Component operations count", ["component_id", "component_type"], registry=self._prom_registry)
            self.g_comp_active_count = Gauge("lezea_active_components_count", "Number of active components", ["component_type"], registry=self._prom_registry)

            start_http_server(port, registry=self._prom_registry)
            self._prom_started = True
            print(f"🛰️ Enhanced Prometheus exporter started on :{port}")
        except Exception as e:
            print(f"⚠️ Prometheus init failed: {e}")
            self._prom_started = False
            self._prom_registry = None

    def _update_prometheus(self, gpu_metrics: List[Dict[str, Any]], sys_metrics: Dict[str, Any]) -> None:
        if not self._prom_started:
            return
        try:
            # Original metrics
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
            
            # NEW: Component metrics
            with self._lock:
                # Update component gauges
                for comp_id, component in self.active_components.items():
                    comp_type = component.component_type.value
                    
                    if self.g_comp_cpu:
                        self.g_comp_cpu.labels(comp_id, comp_type).set(component.cpu_percent)
                    if self.g_comp_memory:
                        self.g_comp_memory.labels(comp_id, comp_type).set(component.memory_mb)
                    if self.g_comp_gpu_memory:
                        self.g_comp_gpu_memory.labels(comp_id, comp_type).set(component.gpu_memory_mb)
                    if self.g_comp_gpu_util:
                        self.g_comp_gpu_util.labels(comp_id, comp_type).set(component.gpu_util_percent)
                    if self.g_comp_operations:
                        # Counter increment (Prometheus handles the cumulative nature)
                        current_ops = self.component_operations.get(comp_id, 0)
                        self.g_comp_operations.labels(comp_id, comp_type)._value._value = current_ops
                
                # Count active components by type
                if self.g_comp_active_count:
                    type_counts = defaultdict(int)
                    for component in self.active_components.values():
                        type_counts[component.component_type.value] += 1
                    
                    for comp_type, count in type_counts.items():
                        self.g_comp_active_count.labels(comp_type).set(count)

        except Exception:
            pass

    def update_data_usage_metrics(self, split: str, usage_rate: Optional[float], gini: Optional[float]) -> None:
        """Lightweight hook used by the tracker to push per-split data-usage stats."""
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
    # Enhanced Loop & collectors
    # ---------------------------
    def _monitoring_loop(self) -> None:
        while self.is_monitoring:
            t0 = time.time()
            try:
                gpu_metrics = self._collect_gpu_metrics()
                sys_metrics = self._collect_system_metrics()
                ts = datetime.now()

                # Update all active components
                with self._lock:
                    for comp_id in list(self.active_components.keys()):
                        self._update_component_metrics(comp_id)

                # Store global snapshot
                snapshot = ResourceSnapshot(
                    timestamp=ts,
                    global_metrics={
                        "total_cpu_percent": sys_metrics.get("cpu_percent", 0),
                        "total_memory_gb": sys_metrics.get("memory_used_gb", 0),
                        "total_gpu_memory_mb": sum(float(g.get("memory_used_mb", 0)) for g in gpu_metrics),
                        "active_components": len(self.active_components)
                    },
                    component_metrics=dict(self.active_components),
                    gpu_devices=gpu_metrics,
                    system_metrics=sys_metrics
                )

                with self._lock:
                    if gpu_metrics:
                        self.gpu_history.append({"timestamp": ts, "metrics": gpu_metrics})
                    if sys_metrics:
                        self.system_history.append({"timestamp": ts, "metrics": sys_metrics})
                    
                    self.resource_snapshots.append(snapshot)
                    self._analyze_performance(gpu_metrics, sys_metrics)

                self._update_prometheus(gpu_metrics, sys_metrics)

            except Exception as e:
                print(f"❌ Error in enhanced monitor loop: {e}")

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
    # Enhanced Analysis & alerts
    # ---------------------------
    def _analyze_performance(self, gpus: List[Dict[str, Any]], system: Dict[str, Any]) -> None:
        try:
            now = datetime.now()
            
            # Original GPU alerts
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

            # Original system alerts
            cpu_pct = float(system.get("cpu_percent", 0.0))
            if cpu_pct > self.THRESH_CPU_HIGH:
                self._push_alert(now, "cpu_high", f"CPU usage high: {cpu_pct:.1f}%", value=cpu_pct)
            sys_mem_pct = float(system.get("memory_percent", 0.0))
            if sys_mem_pct > self.THRESH_SYS_MEM_HIGH:
                self._push_alert(now, "memory_high", f"System memory high: {sys_mem_pct:.1f}%", value=sys_mem_pct)

            # NEW: Component-level alerts
            with self._lock:
                for comp_id, component in self.active_components.items():
                    # High memory usage per component
                    if component.memory_mb > self.THRESH_COMPONENT_MEM_HIGH:
                        self._push_alert(
                            now, 
                            "component_memory_high", 
                            f"Component {comp_id} ({component.component_type.value}) memory high: {component.memory_mb:.1f}MB",
                            component_id=comp_id,
                            component_type=component.component_type.value,
                            value=component.memory_mb
                        )
                
                # Check for component imbalance
                if len(self.active_components) > 1:
                    cpu_values = [c.cpu_percent for c in self.active_components.values() if c.cpu_percent > 0]
                    if len(cpu_values) > 1:
                        max_cpu = max(cpu_values)
                        min_cpu = min(cpu_values)
                        if max_cpu > 0 and (max_cpu - min_cpu) / max_cpu * 100 > self.THRESH_COMPONENT_IMBALANCE:
                            self._push_alert(
                                now,
                                "component_imbalance",
                                f"High CPU imbalance: max={max_cpu:.1f}%, min={min_cpu:.1f}%",
                                max_cpu=max_cpu,
                                min_cpu=min_cpu,
                                imbalance_percent=(max_cpu - min_cpu) / max_cpu * 100
                            )

            # Trim alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
        except Exception:
            pass

    def _push_alert(self, ts: datetime, typ: str, message: str, **kwargs: Any) -> None:
        self.alerts.append({"timestamp": ts, "type": typ, "message": message, **kwargs})

    # ---------------------------
    # Enhanced Public getters
    # ---------------------------
    def get_current_usage(self) -> Dict[str, Any]:
        """Enhanced snapshot with component-level data."""
        try:
            g = self._collect_gpu_metrics()
            s = self._collect_system_metrics()

            # Original summary for tracker
            total_gpu_mem_mb = 0.0
            util_vals: List[float] = []
            for m in g:
                total_gpu_mem_mb += float(m.get("memory_used_mb", 0.0))
                if "utilization_percent" in m:
                    util_vals.append(float(m["utilization_percent"]))
            avg_util = sum(util_vals) / len(util_vals) if util_vals else 0.0

            # Component-level summary
            component_summary = {}
            with self._lock:
                for comp_id, component in self.active_components.items():
                    component_summary[comp_id] = {
                        "type": component.component_type.value,
                        "cpu_percent": component.cpu_percent,
                        "memory_mb": component.memory_mb,
                        "gpu_memory_mb": component.gpu_memory_mb,
                        "gpu_util_percent": component.gpu_util_percent,
                        "operations_count": component.operations_count
                    }

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "gpu_devices": g,
                "system": s,
                "device_count": self.device_count,
                # Tracker-friendly rollups:
                "cpu_percent": float(s.get("cpu_percent", 0.0)),
                "memory_mb": round(total_gpu_mem_mb, 2),
                "gpu_util_percent": round(avg_util, 2),
                # NEW: Component data
                "active_components_count": len(self.active_components),
                "component_summary": component_summary
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
            print(f"❌ Enhanced snapshot failed: {e}")
            return {}

    def get_component_history(self, component_id: str, minutes: int = 10) -> Dict[str, Any]:
        """Get historical data for a specific component."""
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            with self._lock:
                if component_id not in self.component_history:
                    return {"error": f"Component {component_id} not found"}
                
                history = [
                    entry for entry in self.component_history[component_id] 
                    if entry["timestamp"] > cutoff
                ]
                
                return {
                    "component_id": component_id,
                    "time_range_minutes": minutes,
                    "sample_count": len(history),
                    "history": [
                        {"timestamp": entry["timestamp"].isoformat(), "metrics": entry["metrics"]}
                        for entry in history
                    ]
                }
        except Exception as e:
            print(f"❌ Component history failed: {e}")
            return {"error": str(e)}

    def get_usage_history(self, minutes: int = 10) -> Dict[str, Any]:
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            with self._lock:
                rg = [e for e in self.gpu_history if e["timestamp"] > cutoff]
                rs = [e for e in self.system_history if e["timestamp"] > cutoff]
                rc = [s for s in self.resource_snapshots if s.timestamp > cutoff]
                
            return {
                "time_range_minutes": minutes,
                "gpu_history": [{"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in rg],
                "system_history": [{"timestamp": e["timestamp"].isoformat(), "metrics": e["metrics"]} for e in rs],
                "component_snapshots": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "global_metrics": s.global_metrics,
                        "component_count": len(s.component_metrics)
                    } for s in rc
                ],
                "sample_count": len(rg),
            }
        except Exception as e:
            print(f"❌ Enhanced history failed: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        try:
            with self._lock:
                if not self.gpu_history:
                    sys_sum = self.system_history[-1]["metrics"] if self.system_history else {}
                    return {"error": "No GPU samples yet", "system": sys_sum}

                # Original GPU device stats
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

                # NEW: Component performance summary
                component_stats = {}
                for comp_id, history in self.component_history.items():
                    if not history:
                        continue
                    
                    cpu_values = [entry["metrics"]["cpu_percent"] for entry in history]
                    memory_values = [entry["metrics"]["memory_mb"] for entry in history]
                    gpu_memory_values = [entry["metrics"]["gpu_memory_mb"] for entry in history]
                    
                    component_stats[comp_id] = {
                        "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0,
                        "max_cpu_percent": round(max(cpu_values), 2) if cpu_values else 0,
                        "avg_memory_mb": round(sum(memory_values) / len(memory_values), 2) if memory_values else 0,
                        "max_memory_mb": round(max(memory_values), 2) if memory_values else 0,
                        "avg_gpu_memory_mb": round(sum(gpu_memory_values) / len(gpu_memory_values), 2) if gpu_memory_values else 0,
                        "max_gpu_memory_mb": round(max(gpu_memory_values), 2) if gpu_memory_values else 0,
                        "sample_count": len(history)
                    }

                duration_min = len(self.gpu_history) * self.sampling_interval / 60.0
                recent_alerts = [a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(minutes=30)]
                
                return {
                    "monitoring_duration_minutes": round(duration_min, 2),
                    "sample_count": len(self.gpu_history),
                    "devices": devices,
                    "components": component_stats,
                    "active_components_count": len(self.active_components),
                    "recent_alerts": len(recent_alerts),
                    "alert_types": sorted(set(a["type"] for a in recent_alerts)),
                    "component_analysis": self.get_component_analysis()
                }
        except Exception as e:
            print(f"❌ Enhanced summary failed: {e}")
            return {"error": str(e)}

    def get_optimization_recommendations(self) -> List[str]:
        recs: List[str] = []
        try:
            snap = self.get_current_usage()
            gpus = snap.get("gpu_devices", [])
            sysm = snap.get("system", {})
            component_summary = snap.get("component_summary", {})

            if not gpus:
                recs.append("No GPU data available; utilization looks CPU-bound or system has no CUDA devices.")
                return recs

            # Original GPU recommendations
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

            # NEW: Component-level recommendations
            if component_summary:
                # Find high-resource components
                high_cpu_components = [
                    (comp_id, data) for comp_id, data in component_summary.items() 
                    if data["cpu_percent"] > 50.0
                ]
                high_memory_components = [
                    (comp_id, data) for comp_id, data in component_summary.items() 
                    if data["memory_mb"] > self.THRESH_COMPONENT_MEM_HIGH
                ]

                if high_cpu_components:
                    comp_names = [f"{comp_id}({data['type']})" for comp_id, data in high_cpu_components[:3]]
                    recs.append(f"High CPU components: {', '.join(comp_names)}. Consider optimizing these components.")

                if high_memory_components:
                    comp_names = [f"{comp_id}({data['type']})" for comp_id, data in high_memory_components[:3]]
                    recs.append(f"High memory components: {', '.join(comp_names)}. Consider memory optimization.")

                # Check for imbalances
                cpu_values = [data["cpu_percent"] for data in component_summary.values()]
                if len(cpu_values) > 1:
                    max_cpu = max(cpu_values)
                    min_cpu = min(cpu_values)
                    if max_cpu > 0 and (max_cpu - min_cpu) / max_cpu * 100 > self.THRESH_COMPONENT_IMBALANCE:
                        recs.append(f"Component CPU imbalance detected: {max_cpu:.1f}% vs {min_cpu:.1f}%. Consider load balancing.")

                # Component type recommendations
                component_types = {}
                for comp_id, data in component_summary.items():
                    comp_type = data["type"]
                    if comp_type not in component_types:
                        component_types[comp_type] = {"count": 0, "total_memory": 0, "total_cpu": 0}
                    component_types[comp_type]["count"] += 1
                    component_types[comp_type]["total_memory"] += data["memory_mb"]
                    component_types[comp_type]["total_cpu"] += data["cpu_percent"]

                for comp_type, stats in component_types.items():
                    avg_memory = stats["total_memory"] / stats["count"]
                    avg_cpu = stats["total_cpu"] / stats["count"]
                    
                    if avg_memory > self.THRESH_COMPONENT_MEM_HIGH:
                        recs.append(f"{comp_type.title()} components using high memory (avg {avg_memory:.1f}MB). Consider component-specific optimization.")
                    
                    if comp_type == "tasker" and avg_cpu < 10.0:
                        recs.append("Tasker components have low CPU usage. Consider increasing task complexity or parallelism.")
                    elif comp_type == "builder" and avg_cpu > 80.0:
                        recs.append("Builder components have high CPU usage. Consider optimizing building algorithms.")

            if not recs:
                recs.append("GPU and component utilization looks healthy — no immediate tuning needed.")
            
            return recs
        except Exception as e:
            print(f"❌ Enhanced recommendations failed: {e}")
            return ["Unable to generate recommendations due to an error."]

    # ---------------------------
    # Enhanced Export
    # ---------------------------
    def export_monitoring_data(self, filepath: str, include_history: bool = True) -> None:
        try:
            with self._lock:
                # Component data export
                component_export = {}
                for comp_id, component in self.active_components.items():
                    component_export[comp_id] = {
                        "component_type": component.component_type.value,
                        "current_metrics": {
                            "cpu_percent": component.cpu_percent,
                            "memory_mb": component.memory_mb,
                            "gpu_memory_mb": component.gpu_memory_mb,
                            "gpu_util_percent": component.gpu_util_percent,
                            "operations_count": component.operations_count
                        },
                        "peak_usage": self.component_peak_usage.get(comp_id, {}),
                        "start_time": component.start_time.isoformat(),
                        "uptime_seconds": (datetime.now() - component.start_time).total_seconds()
                    }

                export = {
                    "export_timestamp": datetime.now().isoformat(),
                    "experiment_id": self.experiment_id,
                    "device_info": self.devices_info,
                    "monitoring_config": {
                        "sampling_interval": self.sampling_interval,
                        "history_size": self.history_size,
                        "gpu_library": self.gpu_library,
                        "component_tracking_enabled": True
                    },
                    "performance_summary": self.get_performance_summary(),
                    "current_usage": self.get_current_usage(),
                    "component_analysis": self.get_component_analysis(),
                    "active_components": component_export,
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
                    export["component_history"] = {
                        comp_id: [
                            {"timestamp": entry["timestamp"].isoformat(), "metrics": entry["metrics"]}
                            for entry in history
                        ]
                        for comp_id, history in self.component_history.items()
                    }
                    export["resource_snapshots"] = [
                        {
                            "timestamp": s.timestamp.isoformat(),
                            "global_metrics": s.global_metrics,
                            "component_count": len(s.component_metrics)
                        }
                        for s in self.resource_snapshots
                    ]
                    
            with open(filepath, "w") as f:
                json.dump(export, f, indent=2, default=str)
            print(f"📁 Exported enhanced monitoring data to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to export enhanced monitoring data: {e}")

    # ---------------------------
    # NEW: Component context managers for easy tracking
    # ---------------------------
    def track_component(self, component_id: str, component_type: ComponentType, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for automatic component tracking."""
        return ComponentTracker(self, component_id, component_type, metadata)

    def track_tasker(self, tasker_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Convenience method for tracking tasker components."""
        return self.track_component(tasker_id, ComponentType.TASKER, metadata)

    def track_builder(self, builder_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Convenience method for tracking builder components."""
        return self.track_component(builder_id, ComponentType.BUILDER, metadata)

    def track_algorithm(self, algorithm_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Convenience method for tracking algorithm components."""
        return self.track_component(algorithm_id, ComponentType.ALGORITHM, metadata)

    def track_network(self, network_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Convenience method for tracking network components."""
        return self.track_component(network_id, ComponentType.NETWORK, metadata)

    def track_layer(self, layer_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Convenience method for tracking layer components."""
        return self.track_component(layer_id, ComponentType.LAYER, metadata)

    # ---------------------------
    # Enhanced Cleanup / Context
    # ---------------------------
    def cleanup(self) -> None:
        self.stop_monitoring()
        with self._lock:
            self.gpu_history.clear()
            self.system_history.clear()
            self.alerts.clear()
            self.active_components.clear()
            self.component_history.clear()
            self.component_start_times.clear()
            self.component_baseline_metrics.clear()
            self.component_io_baselines.clear()
            self.component_operations.clear()
            self.component_peak_usage.clear()
            self.resource_snapshots.clear()
        print("🧹 Enhanced GPU monitor cleaned up")

    def __enter__(self) -> "GPUMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


class ComponentTracker:
    """Context manager for automatic component registration/unregistration."""

    def __init__(self, monitor: GPUMonitor, component_id: str, component_type: ComponentType, metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.component_id = component_id
        self.component_type = component_type
        self.metadata = metadata or {}
        self.final_metrics: Optional[ComponentMetrics] = None

    def __enter__(self) -> "ComponentTracker":
        self.monitor.register_component(self.component_id, self.component_type, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.final_metrics = self.monitor.unregister_component(self.component_id)

    def record_operation(self, count: int = 1) -> None:
        """Record operations performed by this component."""
        self.monitor.record_component_operation(self.component_id, count)

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics for this component."""
        return self.monitor.get_component_metrics(self.component_id)

    def get_final_metrics(self) -> Optional[ComponentMetrics]:
        """Get final metrics after context exits."""
        return self.final_metrics# lezea_mlops/monitoring/gpu_monitor.py