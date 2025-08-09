"""
Prometheus metrics definitions for LeZeA MLOps
==============================================

Comprehensive metrics collection for training performance, resource usage,
model quality, and system health monitoring.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

class LeZeAMetrics:
    """
    Comprehensive Prometheus metrics for LeZeA MLOps
    
    Provides metrics for:
    - Training performance and progress
    - Resource utilization (GPU, CPU, Memory)
    - Model quality and validation metrics
    - System health and availability
    - Data pipeline performance
    """
    
    def __init__(self, registry: Optional[object] = None, 
                 metrics_port: int = 8000, auto_start_server: bool = False):
        """
        Initialize Prometheus metrics
        
        Args:
            registry: Custom Prometheus registry (uses default if None)
            metrics_port: Port for metrics HTTP server
            auto_start_server: Whether to automatically start metrics server
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics will be disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.registry = registry or REGISTRY
        self.metrics_port = metrics_port
        self.server_started = False
        
        # Initialize all metrics
        self._init_training_metrics()
        self._init_resource_metrics()
        self._init_model_metrics()
        self._init_system_metrics()
        self._init_data_metrics()
        
        # Start background resource monitoring
        self._start_resource_monitoring()
        
        # Auto-start metrics server if requested
        if auto_start_server:
            self.start_metrics_server()
    
    def _init_training_metrics(self):
        """Initialize training-related metrics"""
        # Training progress
        self.training_step = Counter(
            'lezea_training_steps_total',
            'Total number of training steps completed',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        self.training_epoch = Counter(
            'lezea_training_epochs_total', 
            'Total number of training epochs completed',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        # Training performance
        self.step_duration = Histogram(
            'lezea_step_duration_seconds',
            'Time taken for each training step',
            ['experiment_id', 'model_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.epoch_duration = Histogram(
            'lezea_epoch_duration_seconds',
            'Time taken for each training epoch', 
            ['experiment_id', 'model_type'],
            buckets=[60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0],
            registry=self.registry
        )
        
        # Throughput metrics
        self.samples_per_second = Gauge(
            'lezea_training_samples_per_second',
            'Training throughput in samples per second',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        self.tokens_per_second = Gauge(
            'lezea_training_tokens_per_second',
            'Training throughput in tokens per second',
            ['experiment_id', 'model_type'], 
            registry=self.registry
        )
        
        # Loss and gradient metrics
        self.current_loss = Gauge(
            'lezea_current_loss',
            'Current training loss value',
            ['experiment_id', 'model_type', 'loss_type'],
            registry=self.registry
        )
        
        self.gradient_norm = Gauge(
            'lezea_gradient_norm',
            'Current gradient norm',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        self.learning_rate = Gauge(
            'lezea_learning_rate',
            'Current learning rate',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
    
    def _init_resource_metrics(self):
        """Initialize resource utilization metrics"""
        # GPU metrics
        self.gpu_utilization = Gauge(
            'lezea_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_used = Gauge(
            'lezea_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_total = Gauge(
            'lezea_gpu_memory_total_bytes', 
            'Total GPU memory in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_temperature = Gauge(
            'lezea_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_power_draw = Gauge(
            'lezea_gpu_power_draw_watts',
            'GPU power consumption in watts',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        # CPU metrics
        self.cpu_utilization = Gauge(
            'lezea_cpu_utilization_percent',
            'CPU utilization percentage',
            registry=self.registry
        )
        
        self.cpu_count = Gauge(
            'lezea_cpu_count',
            'Number of CPU cores',
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_used = Gauge(
            'lezea_memory_used_bytes',
            'System memory used in bytes',
            registry=self.registry
        )
        
        self.memory_total = Gauge(
            'lezea_memory_total_bytes',
            'Total system memory in bytes', 
            registry=self.registry
        )
        
        self.memory_available = Gauge(
            'lezea_memory_available_bytes',
            'Available system memory in bytes',
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_used = Gauge(
            'lezea_disk_used_bytes',
            'Disk space used in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        self.disk_total = Gauge(
            'lezea_disk_total_bytes',
            'Total disk space in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_bytes_sent = Counter(
            'lezea_network_bytes_sent_total',
            'Total network bytes sent',
            ['interface'],
            registry=self.registry
        )
        
        self.network_bytes_received = Counter(
            'lezea_network_bytes_received_total',
            'Total network bytes received',
            ['interface'],
            registry=self.registry
        )
    
    def _init_model_metrics(self):
        """Initialize model quality and performance metrics"""
        # Model accuracy metrics
        self.model_accuracy = Gauge(
            'lezea_model_accuracy',
            'Current model accuracy',
            ['experiment_id', 'model_type', 'dataset'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'lezea_model_precision',
            'Current model precision',
            ['experiment_id', 'model_type', 'dataset'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'lezea_model_recall',
            'Current model recall',
            ['experiment_id', 'model_type', 'dataset'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'lezea_model_f1_score',
            'Current model F1 score',
            ['experiment_id', 'model_type', 'dataset'],
            registry=self.registry
        )
        
        # Model size and complexity
        self.model_parameters = Gauge(
            'lezea_model_parameters_total',
            'Total number of model parameters',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        self.model_size_bytes = Gauge(
            'lezea_model_size_bytes',
            'Model size in bytes',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
        
        # Inference performance
        self.inference_latency = Histogram(
            'lezea_inference_duration_seconds',
            'Model inference latency',
            ['experiment_id', 'model_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.inference_throughput = Gauge(
            'lezea_inference_samples_per_second',
            'Model inference throughput',
            ['experiment_id', 'model_type'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system health and availability metrics"""
        # Service availability
        self.service_up = Gauge(
            'lezea_service_up',
            'Whether service is available (1) or not (0)',
            ['service_name', 'service_type'],
            registry=self.registry
        )
        
        self.service_response_time = Histogram(
            'lezea_service_response_seconds',
            'Service response time',
            ['service_name', 'service_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Error tracking
        self.errors_total = Counter(
            'lezea_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.experiments_total = Counter(
            'lezea_experiments_total',
            'Total number of experiments',
            ['status', 'model_type'],
            registry=self.registry
        )
        
        # System information
        self.system_info = Info(
            'lezea_system_info',
            'System information',
            registry=self.registry
        )
    
    def _init_data_metrics(self):
        """Initialize data pipeline metrics"""
        # Data loading performance
        self.data_loading_time = Histogram(
            'lezea_data_loading_seconds',
            'Time taken to load data batches',
            ['dataset_name', 'data_type'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.data_processing_time = Histogram(
            'lezea_data_processing_seconds',
            'Time taken to process data',
            ['dataset_name', 'operation'],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Dataset metrics
        self.dataset_size = Gauge(
            'lezea_dataset_size_bytes',
            'Dataset size in bytes',
            ['dataset_name', 'version'],
            registry=self.registry
        )
        
        self.dataset_samples = Gauge(
            'lezea_dataset_samples_total',
            'Total number of samples in dataset',
            ['dataset_name', 'split'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'lezea_data_quality_score',
            'Data quality score (0-1)',
            ['dataset_name', 'metric_type'],
            registry=self.registry
        )
    
    def _start_resource_monitoring(self):
        """Start background thread for resource monitoring"""
        if not self.enabled:
            return
            
        def monitor_resources():
            while True:
                try:
                    # Update CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_utilization.set(cpu_percent)
                    self.cpu_count.set(psutil.cpu_count())
                    
                    # Update memory metrics
                    memory = psutil.virtual_memory()
                    self.memory_used.set(memory.used)
                    self.memory_total.set(memory.total)
                    self.memory_available.set(memory.available)
                    
                    # Update disk metrics
                    for disk in psutil.disk_partitions():
                        try:
                            disk_usage = psutil.disk_usage(disk.mountpoint)
                            self.disk_used.labels(mount_point=disk.mountpoint).set(disk_usage.used)
                            self.disk_total.labels(mount_point=disk.mountpoint).set(disk_usage.total)
                        except (PermissionError, FileNotFoundError):
                            continue
                    
                    # Update network metrics
                    network = psutil.net_io_counters(pernic=True)
                    for interface, stats in network.items():
                        self.network_bytes_sent.labels(interface=interface)._value._value = stats.bytes_sent
                        self.network_bytes_received.labels(interface=interface)._value._value = stats.bytes_recv
                    
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def start_metrics_server(self, port: Optional[int] = None):
        """Start Prometheus metrics HTTP server"""
        if not self.enabled:
            logger.warning("Prometheus not available, cannot start metrics server")
            return False
        
        if self.server_started:
            logger.info("Metrics server already started")
            return True
        
        port = port or self.metrics_port
        try:
            start_http_server(port, registry=self.registry)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def record_training_step(self, experiment_id: str, model_type: str, 
                           step_time: float, loss: float, 
                           samples_per_sec: Optional[float] = None,
                           tokens_per_sec: Optional[float] = None):
        """Record training step metrics"""
        if not self.enabled:
            return
        
        self.training_step.labels(experiment_id=experiment_id, model_type=model_type).inc()
        self.step_duration.labels(experiment_id=experiment_id, model_type=model_type).observe(step_time)
        self.current_loss.labels(experiment_id=experiment_id, model_type=model_type, loss_type='training').set(loss)
        
        if samples_per_sec is not None:
            self.samples_per_second.labels(experiment_id=experiment_id, model_type=model_type).set(samples_per_sec)
        
        if tokens_per_sec is not None:
            self.tokens_per_second.labels(experiment_id=experiment_id, model_type=model_type).set(tokens_per_sec)
    
    def record_epoch_completion(self, experiment_id: str, model_type: str, 
                              epoch_time: float):
        """Record epoch completion"""
        if not self.enabled:
            return
        
        self.training_epoch.labels(experiment_id=experiment_id, model_type=model_type).inc()
        self.epoch_duration.labels(experiment_id=experiment_id, model_type=model_type).observe(epoch_time)
    
    def update_gpu_metrics(self, gpu_stats: List[Dict[str, Any]]):
        """Update GPU metrics from monitoring data"""
        if not self.enabled:
            return
        
        for gpu in gpu_stats:
            gpu_id = str(gpu.get('id', 'unknown'))
            gpu_name = gpu.get('name', 'unknown')
            
            if 'utilization' in gpu:
                self.gpu_utilization.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(gpu['utilization'])
            
            if 'memory_used' in gpu:
                self.gpu_memory_used.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(gpu['memory_used'])
            
            if 'memory_total' in gpu:
                self.gpu_memory_total.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(gpu['memory_total'])
            
            if 'temperature' in gpu:
                self.gpu_temperature.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(gpu['temperature'])
            
            if 'power_draw' in gpu:
                self.gpu_power_draw.labels(gpu_id=gpu_id, gpu_name=gpu_name).set(gpu['power_draw'])
    
    def record_model_metrics(self, experiment_id: str, model_type: str,
                           accuracy: Optional[float] = None,
                           precision: Optional[float] = None,
                           recall: Optional[float] = None,
                           f1_score: Optional[float] = None,
                           dataset: str = 'validation'):
        """Record model performance metrics"""
        if not self.enabled:
            return
        
        labels = {'experiment_id': experiment_id, 'model_type': model_type, 'dataset': dataset}
        
        if accuracy is not None:
            self.model_accuracy.labels(**labels).set(accuracy)
        if precision is not None:
            self.model_precision.labels(**labels).set(precision)
        if recall is not None:
            self.model_recall.labels(**labels).set(recall)
        if f1_score is not None:
            self.model_f1_score.labels(**labels).set(f1_score)
    
    def record_inference_metrics(self, experiment_id: str, model_type: str,
                               latency: float, throughput: Optional[float] = None):
        """Record model inference metrics"""
        if not self.enabled:
            return
        
        self.inference_latency.labels(experiment_id=experiment_id, model_type=model_type).observe(latency)
        if throughput is not None:
            self.inference_throughput.labels(experiment_id=experiment_id, model_type=model_type).set(throughput)
    
    def record_service_health(self, service_name: str, service_type: str,
                            is_up: bool, response_time: Optional[float] = None):
        """Record service health status"""
        if not self.enabled:
            return
        
        self.service_up.labels(service_name=service_name, service_type=service_type).set(1 if is_up else 0)
        if response_time is not None:
            self.service_response_time.labels(service_name=service_name, service_type=service_type).observe(response_time)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        if not self.enabled:
            return
        
        self.errors_total.labels(error_type=error_type, component=component).inc()
    
    def record_experiment_start(self, model_type: str):
        """Record experiment start"""
        if not self.enabled:
            return
        
        self.experiments_total.labels(status='started', model_type=model_type).inc()
    
    def record_experiment_completion(self, model_type: str, success: bool):
        """Record experiment completion"""
        if not self.enabled:
            return
        
        status = 'completed' if success else 'failed'
        self.experiments_total.labels(status=status, model_type=model_type).inc()
    
    def set_system_info(self, info_dict: Dict[str, str]):
        """Set system information"""
        if not self.enabled:
            return
        
        self.system_info.info(info_dict)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


# Global metrics instance
_global_metrics: Optional[LeZeAMetrics] = None

def get_metrics() -> LeZeAMetrics:
    """Get global metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = LeZeAMetrics()
    return _global_metrics

def init_metrics(port: int = 8000, auto_start: bool = True) -> LeZeAMetrics:
    """Initialize global metrics instance"""
    global _global_metrics
    _global_metrics = LeZeAMetrics(metrics_port=port, auto_start_server=auto_start)
    return _global_metrics


# Export key classes and functions
__all__ = [
    'LeZeAMetrics',
    'get_metrics', 
    'init_metrics',
    'PROMETHEUS_AVAILABLE'
]