"""
Prometheus metrics for LeZeA MLOps — FINAL
==========================================

- Training, resource, model, system, and data metrics
- Thread-safe background resource sampler (CPU/Mem/Disk/Net)
- Network counters use deltas (correct Prometheus semantics)
- GPU metrics updater (generic) + GPUMonitor wiring helper
- Graceful degradation when prometheus_client / psutil absent
- Convenience helpers: time_step(), record_*(), start/stop server
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Optional deps ---------------------------------------------------------------
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from prometheus_client import (  # type: ignore
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
    # Minimal no-op stubs to keep calls safe
    class _Noop:
        def __init__(self, *_, **__): pass
        def labels(self, *_, **__): return self
        def inc(self, *_, **__): pass
        def set(self, *_, **__): pass
        def observe(self, *_, **__): pass
        def time(self, *_, **__): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def info(self, *_, **__): pass
    Counter = Gauge = Histogram = Summary = Info = _Noop  # type: ignore
    REGISTRY = object()  # type: ignore
    CollectorRegistry = object  # type: ignore
    def generate_latest(*_, **__): return b"# prometheus client unavailable\n"
    def start_http_server(*_, **__): pass
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class LeZeAMetrics:
    """
    Comprehensive Prometheus metrics for LeZeA MLOps.

    Works even if Prometheus/psutil are missing (becomes a safe no-op).
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        metrics_port: int = 8000,
        auto_start_server: bool = False,
        resource_poll_interval: float = 10.0,
    ):
        self.enabled = PROMETHEUS_AVAILABLE
        if not self.enabled:
            logger.warning("Prometheus client not available. Metrics are disabled.")
        self.psutil_ok = PSUTIL_AVAILABLE
        if not self.psutil_ok:
            logger.warning("psutil not available. System resource metrics are disabled.")

        self.registry = registry or (REGISTRY if PROMETHEUS_AVAILABLE else None)
        self.metrics_port = int(metrics_port)
        self.server_started = False

        # Threads and stop events
        self._resource_thread: Optional[threading.Thread] = None
        self._resource_stop = threading.Event()
        self._resource_poll_interval = float(resource_poll_interval)

        self._gpu_thread: Optional[threading.Thread] = None
        self._gpu_stop = threading.Event()
        self._wired_monitor = None
        self._gpu_poll_interval = 5.0

        # Keep last network totals for delta counters
        self._net_last: Dict[str, Dict[str, int]] = {}

        # Metric families
        self._init_training_metrics()
        self._init_resource_metrics()
        self._init_model_metrics()
        self._init_system_metrics()
        self._init_data_metrics()

        # Background resource monitoring (CPU/Mem/Disk/Net)
        if self.enabled and self.psutil_ok:
            self._start_resource_monitoring()

        if auto_start_server:
            self.start_metrics_server()

    # ------------------------------------------------------------------ init
    def _init_training_metrics(self) -> None:
        self.training_step = Counter(
            "lezea_training_steps_total",
            "Total number of training steps completed",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.training_epoch = Counter(
            "lezea_training_epochs_total",
            "Total number of training epochs completed",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.step_duration = Histogram(
            "lezea_step_duration_seconds",
            "Time taken for each training step",
            ["experiment_id", "model_type"],
            buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0],
            registry=self.registry,
        )
        self.epoch_duration = Histogram(
            "lezea_epoch_duration_seconds",
            "Time taken for each training epoch",
            ["experiment_id", "model_type"],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
            registry=self.registry,
        )
        self.samples_per_second = Gauge(
            "lezea_training_samples_per_second",
            "Training throughput in samples per second",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.tokens_per_second = Gauge(
            "lezea_training_tokens_per_second",
            "Training throughput in tokens per second",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.current_loss = Gauge(
            "lezea_current_loss",
            "Current training loss value",
            ["experiment_id", "model_type", "loss_type"],
            registry=self.registry,
        )
        self.gradient_norm = Gauge(
            "lezea_gradient_norm",
            "Current gradient norm",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.learning_rate = Gauge(
            "lezea_learning_rate",
            "Current learning rate",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )

    def _init_resource_metrics(self) -> None:
        # GPU
        self.gpu_utilization = Gauge(
            "lezea_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id", "gpu_name"],
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            "lezea_gpu_memory_used_bytes",
            "GPU memory used (bytes)",
            ["gpu_id", "gpu_name"],
            registry=self.registry,
        )
        self.gpu_memory_total = Gauge(
            "lezea_gpu_memory_total_bytes",
            "GPU memory total (bytes)",
            ["gpu_id", "gpu_name"],
            registry=self.registry,
        )
        self.gpu_temperature = Gauge(
            "lezea_gpu_temperature_celsius",
            "GPU temperature (C)",
            ["gpu_id", "gpu_name"],
            registry=self.registry,
        )
        self.gpu_power_draw = Gauge(
            "lezea_gpu_power_draw_watts",
            "GPU power consumption (W)",
            ["gpu_id", "gpu_name"],
            registry=self.registry,
        )
        # CPU
        self.cpu_utilization = Gauge(
            "lezea_cpu_utilization_percent",
            "CPU utilization percent",
            registry=self.registry,
        )
        self.cpu_count = Gauge(
            "lezea_cpu_count",
            "CPU core count",
            registry=self.registry,
        )
        # Memory
        self.memory_used = Gauge(
            "lezea_memory_used_bytes",
            "System memory used (bytes)",
            registry=self.registry,
        )
        self.memory_total = Gauge(
            "lezea_memory_total_bytes",
            "System memory total (bytes)",
            registry=self.registry,
        )
        self.memory_available = Gauge(
            "lezea_memory_available_bytes",
            "System memory available (bytes)",
            registry=self.registry,
        )
        # Disk
        self.disk_used = Gauge(
            "lezea_disk_used_bytes",
            "Disk used (bytes)",
            ["mount_point"],
            registry=self.registry,
        )
        self.disk_total = Gauge(
            "lezea_disk_total_bytes",
            "Disk total (bytes)",
            ["mount_point"],
            registry=self.registry,
        )
        # Network (counters must be incremented, not set!)
        self.network_bytes_sent = Counter(
            "lezea_network_bytes_sent_total",
            "Total network bytes sent",
            ["interface"],
            registry=self.registry,
        )
        self.network_bytes_received = Counter(
            "lezea_network_bytes_received_total",
            "Total network bytes received",
            ["interface"],
            registry=self.registry,
        )

    def _init_model_metrics(self) -> None:
        self.model_accuracy = Gauge(
            "lezea_model_accuracy",
            "Current model accuracy",
            ["experiment_id", "model_type", "dataset"],
            registry=self.registry,
        )
        self.model_precision = Gauge(
            "lezea_model_precision",
            "Current model precision",
            ["experiment_id", "model_type", "dataset"],
            registry=self.registry,
        )
        self.model_recall = Gauge(
            "lezea_model_recall",
            "Current model recall",
            ["experiment_id", "model_type", "dataset"],
            registry=self.registry,
        )
        self.model_f1_score = Gauge(
            "lezea_model_f1_score",
            "Current model F1 score",
            ["experiment_id", "model_type", "dataset"],
            registry=self.registry,
        )
        self.model_parameters = Gauge(
            "lezea_model_parameters_total",
            "Total number of model parameters",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.model_size_bytes = Gauge(
            "lezea_model_size_bytes",
            "Model size (bytes)",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "lezea_inference_duration_seconds",
            "Model inference latency",
            ["experiment_id", "model_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )
        self.inference_throughput = Gauge(
            "lezea_inference_samples_per_second",
            "Inference throughput (samples/s)",
            ["experiment_id", "model_type"],
            registry=self.registry,
        )

    def _init_system_metrics(self) -> None:
        self.service_up = Gauge(
            "lezea_service_up",
            "Service availability: 1 up / 0 down",
            ["service_name", "service_type"],
            registry=self.registry,
        )
        self.service_response_time = Histogram(
            "lezea_service_response_seconds",
            "Service response time",
            ["service_name", "service_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry,
        )
        self.errors_total = Counter(
            "lezea_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self.registry,
        )
        self.experiments_total = Counter(
            "lezea_experiments_total",
            "Experiments counter by status",
            ["status", "model_type"],
            registry=self.registry,
        )
        self.system_info = Info(
            "lezea_system_info",
            "Static system information",
            registry=self.registry,
        )

    def _init_data_metrics(self) -> None:
        self.data_loading_time = Histogram(
            "lezea_data_loading_seconds",
            "Time to load data batches",
            ["dataset_name", "data_type"],
            buckets=[0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0],
            registry=self.registry,
        )
        self.data_processing_time = Histogram(
            "lezea_data_processing_seconds",
            "Time to process data",
            ["dataset_name", "operation"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry,
        )
        self.dataset_size = Gauge(
            "lezea_dataset_size_bytes",
            "Dataset size (bytes)",
            ["dataset_name", "version"],
            registry=self.registry,
        )
        self.dataset_samples = Gauge(
            "lezea_dataset_samples_total",
            "Dataset sample counts",
            ["dataset_name", "split"],
            registry=self.registry,
        )
        self.data_quality_score = Gauge(
            "lezea_data_quality_score",
            "Data quality score (0-1)",
            ["dataset_name", "metric_type"],
            registry=self.registry,
        )

    # --------------------------------------------------------------- background
    def _start_resource_monitoring(self) -> None:
        if self._resource_thread and self._resource_thread.is_alive():
            return

        def loop():
            # seed last net totals
            if self.psutil_ok:
                try:
                    nic = psutil.net_io_counters(pernic=True)
                    self._net_last = {
                        name: {"sent": int(stats.bytes_sent), "recv": int(stats.bytes_recv)}
                        for name, stats in nic.items()
                    }
                except Exception:
                    self._net_last = {}

            while not self._resource_stop.is_set():
                try:
                    if self.psutil_ok:
                        # CPU
                        cpu_percent = psutil.cpu_percent(interval=0.0)
                        self.cpu_utilization.set(cpu_percent)
                        try:
                            self.cpu_count.set(psutil.cpu_count())
                        except Exception:
                            pass

                        # Memory
                        vm = psutil.virtual_memory()
                        self.memory_used.set(int(vm.used))
                        self.memory_total.set(int(vm.total))
                        self.memory_available.set(int(vm.available))

                        # Disk (skip non-ready mountpoints)
                        for disk in psutil.disk_partitions(all=False):
                            try:
                                usage = psutil.disk_usage(disk.mountpoint)
                                self.disk_used.labels(mount_point=disk.mountpoint).set(int(usage.used))
                                self.disk_total.labels(mount_point=disk.mountpoint).set(int(usage.total))
                            except (PermissionError, FileNotFoundError):
                                continue
                            except Exception as e:
                                logger.debug(f"Disk usage failed for {disk.mountpoint}: {e}")

                        # Network (counters must inc by delta)
                        nic = psutil.net_io_counters(pernic=True)
                        for iface, stats in nic.items():
                            last = self._net_last.get(iface, {"sent": 0, "recv": 0})
                            dsent = int(stats.bytes_sent) - int(last["sent"])
                            drecv = int(stats.bytes_recv) - int(last["recv"])
                            # handle resets/rollovers
                            if dsent < 0: dsent = int(stats.bytes_sent)
                            if drecv < 0: drecv = int(stats.bytes_recv)
                            if dsent:
                                self.network_bytes_sent.labels(interface=iface).inc(dsent)
                            if drecv:
                                self.network_bytes_received.labels(interface=iface).inc(drecv)
                            self._net_last[iface] = {"sent": int(stats.bytes_sent), "recv": int(stats.bytes_recv)}

                    # sleep with stop support
                    self._resource_stop.wait(self._resource_poll_interval)
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    self._resource_stop.wait(max(self._resource_poll_interval * 3, 30.0))

        self._resource_stop.clear()
        self._resource_thread = threading.Thread(target=loop, name="LeZeAResourceMonitor", daemon=True)
        self._resource_thread.start()

    # --------------------------------------------------------------- server
    def start_metrics_server(self, port: Optional[int] = None) -> bool:
        if not self.enabled:
            logger.warning("Prometheus not available; cannot start server.")
            return False
        if self.server_started:
            return True
        try:
            start_http_server(int(port or self.metrics_port), registry=self.registry)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on :{port or self.metrics_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False

    def get_metrics_text(self) -> str:
        try:
            return generate_latest(self.registry).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to generate metrics text: {e}")
            return "# error generating metrics\n"

    # --------------------------------------------------------------- helpers (training)
    def record_training_step(
        self,
        experiment_id: str,
        model_type: str,
        step_time: float,
        loss: float,
        samples_per_sec: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        self.training_step.labels(experiment_id=experiment_id, model_type=model_type).inc()
        self.step_duration.labels(experiment_id=experiment_id, model_type=model_type).observe(float(step_time))
        self.current_loss.labels(
            experiment_id=experiment_id, model_type=model_type, loss_type="training"
        ).set(float(loss))
        if samples_per_sec is not None:
            self.samples_per_second.labels(experiment_id=experiment_id, model_type=model_type).set(float(samples_per_sec))
        if tokens_per_sec is not None:
            self.tokens_per_second.labels(experiment_id=experiment_id, model_type=model_type).set(float(tokens_per_sec))

    def record_epoch_completion(self, experiment_id: str, model_type: str, epoch_time: float) -> None:
        if not self.enabled:
            return
        self.training_epoch.labels(experiment_id=experiment_id, model_type=model_type).inc()
        self.epoch_duration.labels(experiment_id=experiment_id, model_type=model_type).observe(float(epoch_time))

    def time_step(
        self,
        experiment_id: str,
        model_type: str,
        *,
        samples: Optional[int] = None,
        tokens: Optional[int] = None,
    ):
        """
        Context manager to time a training step and emit throughput if counts provided.

        with metrics.time_step(exp_id, "tasker", samples=batch_size):
            loss = train_step(...)
            metrics.current_loss.labels(exp_id, "tasker", "training").set(loss)
        """
        class _Timer:
            def __init__(self, outer: "LeZeAMetrics"):
                self.outer = outer
                self.t0 = None
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self
            def __exit__(self, *_):
                dt = max(1e-9, time.perf_counter() - self.t0)
                self.outer.record_training_step(
                    experiment_id, model_type, step_time=dt, loss=float("nan"),
                    samples_per_sec=(samples / dt) if samples else None,
                    tokens_per_sec=(tokens / dt) if tokens else None,
                )
        return _Timer(self)

    # --------------------------------------------------------------- helpers (gpu)
    def update_gpu_metrics(self, gpu_stats: Iterable[Dict[str, Any]]) -> None:
        """
        Update GPU gauges from a list of device dicts. Accepts keys from either:
        - GPUMonitor snapshot: device_id, name, utilization_percent, memory_used_mb,
                               memory_total_mb, temperature_c, power_usage_w
        - Generic: id, name, utilization, memory_used, memory_total, temperature, power_draw
        """
        if not self.enabled:
            return

        for g in gpu_stats:
            gid = str(g.get("device_id", g.get("id", "unknown")))
            name = g.get("name", "unknown")

            # Utilization
            util = g.get("utilization_percent", g.get("utilization"))
            if isinstance(util, (int, float)):
                self.gpu_utilization.labels(gpu_id=gid, gpu_name=name).set(float(util))

            # Memory (convert MB to bytes if needed)
            m_used = g.get("memory_used_mb", None)
            if m_used is not None:
                self.gpu_memory_used.labels(gpu_id=gid, gpu_name=name).set(int(float(m_used) * 1024 * 1024))
            else:
                m_used_b = g.get("memory_used", None)
                if isinstance(m_used_b, (int, float)):
                    self.gpu_memory_used.labels(gpu_id=gid, gpu_name=name).set(int(m_used_b))

            m_total = g.get("memory_total_mb", None)
            if m_total is not None:
                self.gpu_memory_total.labels(gpu_id=gid, gpu_name=name).set(int(float(m_total) * 1024 * 1024))
            else:
                m_total_b = g.get("memory_total", None)
                if isinstance(m_total_b, (int, float)):
                    self.gpu_memory_total.labels(gpu_id=gid, gpu_name=name).set(int(m_total_b))

            # Temperature
            temp = g.get("temperature_c", g.get("temperature"))
            if isinstance(temp, (int, float)):
                self.gpu_temperature.labels(gpu_id=gid, gpu_name=name).set(float(temp))

            # Power
            pwr = g.get("power_usage_w", g.get("power_draw"))
            if isinstance(pwr, (int, float)):
                self.gpu_power_draw.labels(gpu_id=gid, gpu_name=name).set(float(pwr))

    def wire_gpu_monitor(self, monitor, poll_interval: float = 5.0) -> None:
        """
        Poll a GPUMonitor-like object (must expose get_current_usage()) and feed Prometheus.
        """
        self._wired_monitor = monitor
        self._gpu_poll_interval = float(poll_interval)
        if self._gpu_thread and self._gpu_thread.is_alive():
            return

        def loop():
            while not self._gpu_stop.is_set():
                try:
                    snap = monitor.get_current_usage() or {}
                    devices = snap.get("gpu_devices", [])
                    self.update_gpu_metrics(devices)
                except Exception as e:
                    logger.debug(f"wire_gpu_monitor polling error: {e}")
                self._gpu_stop.wait(self._gpu_poll_interval)

        self._gpu_stop.clear()
        self._gpu_thread = threading.Thread(target=loop, name="LeZeAGPUPoller", daemon=True)
        self._gpu_thread.start()

    # --------------------------------------------------------------- helpers (model, sys, data)
    def record_model_metrics(
        self,
        experiment_id: str,
        model_type: str,
        *,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        dataset: str = "validation",
    ) -> None:
        if not self.enabled:
            return
        labels = dict(experiment_id=experiment_id, model_type=model_type, dataset=dataset)
        if accuracy is not None:
            self.model_accuracy.labels(**labels).set(float(accuracy))
        if precision is not None:
            self.model_precision.labels(**labels).set(float(precision))
        if recall is not None:
            self.model_recall.labels(**labels).set(float(recall))
        if f1_score is not None:
            self.model_f1_score.labels(**labels).set(float(f1_score))

    def record_inference_metrics(self, experiment_id: str, model_type: str, latency: float, throughput: Optional[float] = None) -> None:
        if not self.enabled:
            return
        self.inference_latency.labels(experiment_id=experiment_id, model_type=model_type).observe(float(latency))
        if throughput is not None:
            self.inference_throughput.labels(experiment_id=experiment_id, model_type=model_type).set(float(throughput))

    def record_service_health(self, service_name: str, service_type: str, is_up: bool, response_time: Optional[float] = None) -> None:
        if not self.enabled:
            return
        self.service_up.labels(service_name=service_name, service_type=service_type).set(1 if is_up else 0)
        if response_time is not None:
            self.service_response_time.labels(service_name=service_name, service_type=service_type).observe(float(response_time))

    def record_error(self, error_type: str, component: str) -> None:
        if not self.enabled:
            return
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def record_experiment_start(self, model_type: str) -> None:
        if not self.enabled:
            return
        self.experiments_total.labels(status="started", model_type=model_type).inc()

    def record_experiment_completion(self, model_type: str, success: bool) -> None:
        if not self.enabled:
            return
        status = "completed" if success else "failed"
        self.experiments_total.labels(status=status, model_type=model_type).inc()

    def set_system_info(self, info_dict: Dict[str, str]) -> None:
        if not self.enabled:
            return
        # Values must be strings
        info = {k: str(v) for k, v in info_dict.items()}
        self.system_info.info(info)

    # --------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        """Stop background threads (resource + GPU pollers)."""
        try:
            self._resource_stop.set()
            if self._resource_thread and self._resource_thread.is_alive():
                self._resource_thread.join(timeout=5.0)
        except Exception:
            pass
        try:
            self._gpu_stop.set()
            if self._gpu_thread and self._gpu_thread.is_alive():
                self._gpu_thread.join(timeout=5.0)
        except Exception:
            pass


# --------------------------- Global singleton helpers ------------------------
_global_metrics: Optional[LeZeAMetrics] = None

def get_metrics() -> LeZeAMetrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = LeZeAMetrics()
    return _global_metrics

def init_metrics(port: int = 8000, auto_start: bool = True, resource_poll_interval: float = 10.0) -> LeZeAMetrics:
    global _global_metrics
    _global_metrics = LeZeAMetrics(metrics_port=port, auto_start_server=auto_start, resource_poll_interval=resource_poll_interval)
    return _global_metrics

__all__ = ["LeZeAMetrics", "get_metrics", "init_metrics", "PROMETHEUS_AVAILABLE", "PSUTIL_AVAILABLE"]
