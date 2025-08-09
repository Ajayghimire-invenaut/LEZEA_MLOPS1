"""
GPU Monitor for LeZeA MLOps
==========================

Comprehensive GPU monitoring and resource tracking:
- Real-time GPU utilization and memory usage
- Multi-GPU system support
- Temperature and power monitoring
- Performance bottleneck detection
- Automatic optimization recommendations
- Integration with Prometheus metrics

Supports multiple GPU libraries (GPUtil, pynvml, torch) with automatic fallback.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import json

# GPU monitoring library imports with fallback
GPU_LIBRARIES = []

try:
    import GPUtil
    GPU_LIBRARIES.append('GPUtil')
except ImportError:
    GPUtil = None

try:
    import pynvml
    GPU_LIBRARIES.append('pynvml')
except ImportError:
    pynvml = None

try:
    import torch
    if torch.cuda.is_available():
        GPU_LIBRARIES.append('torch')
except ImportError:
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class GPUMonitor:
    """
    Comprehensive GPU monitoring system
    
    This class provides:
    - Real-time GPU metrics collection
    - Multi-GPU system support
    - Historical data tracking
    - Performance analysis and recommendations
    - Resource bottleneck detection
    - Integration with experiment tracking
    """
    
    def __init__(self, sampling_interval: float = 1.0, history_size: int = 1000):
        """
        Initialize GPU monitor
        
        Args:
            sampling_interval: Seconds between measurements
            history_size: Number of historical samples to keep
        """
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        # GPU detection and library selection
        self.gpu_library = None
        self.device_count = 0
        self.devices_info = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.experiment_id = None
        
        # Data storage
        self.gpu_history = deque(maxlen=history_size)
        self.system_history = deque(maxlen=history_size)
        
        # Performance tracking
        self.alerts = []
        self.bottlenecks_detected = []
        
        # Initialize GPU detection
        self._detect_gpus()
        
        print(f"🎮 GPU Monitor initialized")
        print(f"   GPUs detected: {self.device_count}")
        print(f"   Library: {self.gpu_library}")
        print(f"   Sampling: {sampling_interval}s")
    
    def _detect_gpus(self):
        """Detect available GPUs and select monitoring library"""
        # Try GPUtil first (most comprehensive)
        if 'GPUtil' in GPU_LIBRARIES:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_library = 'GPUtil'
                    self.device_count = len(gpus)
                    self.devices_info = [
                        {
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total': gpu.memoryTotal,
                            'driver_version': getattr(gpu, 'driver', 'unknown')
                        }
                        for gpu in gpus
                    ]
                    return
            except Exception as e:
                print(f"⚠️ GPUtil detection failed: {e}")
        
        # Try pynvml (NVIDIA management library)
        if 'pynvml' in GPU_LIBRARIES:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    self.gpu_library = 'pynvml'
                    self.device_count = device_count
                    self.devices_info = []
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        self.devices_info.append({
                            'id': i,
                            'name': name,
                            'memory_total': memory_info.total // (1024**2),  # MB
                            'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                        })
                    return
            except Exception as e:
                print(f"⚠️ pynvml detection failed: {e}")
        
        # Try PyTorch CUDA
        if 'torch' in GPU_LIBRARIES:
            try:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self.gpu_library = 'torch'
                    self.device_count = device_count
                    self.devices_info = []
                    
                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        self.devices_info.append({
                            'id': i,
                            'name': props.name,
                            'memory_total': props.total_memory // (1024**2),  # MB
                            'compute_capability': f"{props.major}.{props.minor}"
                        })
                    return
            except Exception as e:
                print(f"⚠️ PyTorch CUDA detection failed: {e}")
        
        # No GPUs detected
        print("ℹ️ No GPUs detected or GPU libraries unavailable")
    
    def start_monitoring(self, experiment_id: str = None):
        """
        Start GPU monitoring in background thread
        
        Args:
            experiment_id: Optional experiment identifier for logging
        """
        if self.is_monitoring:
            print("⚠️ GPU monitoring already active")
            return
        
        if self.device_count == 0:
            print("⚠️ No GPUs available for monitoring")
            return
        
        self.experiment_id = experiment_id
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="GPUMonitor"
        )
        self.monitor_thread.start()
        
        print(f"📈 Started GPU monitoring for {self.device_count} devices")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        print("📊 Stopped GPU monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.is_monitoring:
            try:
                # Collect GPU metrics
                gpu_metrics = self._collect_gpu_metrics()
                system_metrics = self._collect_system_metrics()
                
                timestamp = datetime.now()
                
                # Store metrics
                if gpu_metrics:
                    self.gpu_history.append({
                        'timestamp': timestamp,
                        'metrics': gpu_metrics
                    })
                
                if system_metrics:
                    self.system_history.append({
                        'timestamp': timestamp,
                        'metrics': system_metrics
                    })
                
                # Check for performance issues
                self._analyze_performance(gpu_metrics, system_metrics)
                
                # Sleep until next sampling
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"❌ Error in GPU monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics from all GPU devices"""
        if not self.gpu_library:
            return []
        
        try:
            if self.gpu_library == 'GPUtil':
                return self._collect_gputil_metrics()
            elif self.gpu_library == 'pynvml':
                return self._collect_pynvml_metrics()
            elif self.gpu_library == 'torch':
                return self._collect_torch_metrics()
        except Exception as e:
            print(f"❌ Failed to collect GPU metrics: {e}")
            return []
    
    def _collect_gputil_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics using GPUtil"""
        gpus = GPUtil.getGPUs()
        metrics = []
        
        for gpu in gpus:
            gpu_metrics = {
                'device_id': gpu.id,
                'name': gpu.name,
                'utilization_percent': round(gpu.load * 100, 1),
                'memory_used_mb': round(gpu.memoryUsed, 1),
                'memory_total_mb': round(gpu.memoryTotal, 1),
                'memory_free_mb': round(gpu.memoryFree, 1),
                'memory_percent': round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1),
                'temperature_c': gpu.temperature
            }
            metrics.append(gpu_metrics)
        
        return metrics
    
    def _collect_pynvml_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics using pynvml"""
        metrics = []
        
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
                except:
                    power = 0
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = memory_clock = 0
                
                gpu_metrics = {
                    'device_id': i,
                    'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                    'utilization_percent': util.gpu,
                    'memory_utilization_percent': util.memory,
                    'memory_used_mb': round(memory_info.used / (1024**2), 1),
                    'memory_total_mb': round(memory_info.total / (1024**2), 1),
                    'memory_free_mb': round(memory_info.free / (1024**2), 1),
                    'memory_percent': round((memory_info.used / memory_info.total) * 100, 1),
                    'temperature_c': temperature,
                    'power_usage_w': power,
                    'graphics_clock_mhz': graphics_clock,
                    'memory_clock_mhz': memory_clock
                }
                metrics.append(gpu_metrics)
                
            except Exception as e:
                print(f"⚠️ Failed to get metrics for GPU {i}: {e}")
        
        return metrics
    
    def _collect_torch_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics using PyTorch"""
        metrics = []
        
        for i in range(self.device_count):
            try:
                # Memory info
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_metrics = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated_mb': round(memory_allocated / (1024**2), 1),
                    'memory_reserved_mb': round(memory_reserved / (1024**2), 1),
                    'memory_total_mb': round(memory_total / (1024**2), 1),
                    'memory_percent': round((memory_reserved / memory_total) * 100, 1)
                }
                
                # Try to get utilization if possible
                try:
                    utilization = torch.cuda.utilization(i)
                    gpu_metrics['utilization_percent'] = utilization
                except:
                    pass
                
                metrics.append(gpu_metrics)
                
            except Exception as e:
                print(f"⚠️ Failed to get PyTorch metrics for GPU {i}: {e}")
        
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics"""
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # System load
            try:
                load_avg = psutil.getloadavg()
            except:
                load_avg = (0, 0, 0)
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2]
            }
            
        except Exception as e:
            print(f"⚠️ Failed to collect system metrics: {e}")
            return {}
    
    def _analyze_performance(self, gpu_metrics: List[Dict], system_metrics: Dict):
        """Analyze performance and detect bottlenecks"""
        try:
            current_time = datetime.now()
            
            # GPU performance analysis
            for gpu_metric in gpu_metrics:
                device_id = gpu_metric['device_id']
                
                # High memory usage warning
                memory_percent = gpu_metric.get('memory_percent', 0)
                if memory_percent > 90:
                    alert = {
                        'timestamp': current_time,
                        'type': 'gpu_memory_high',
                        'device_id': device_id,
                        'value': memory_percent,
                        'message': f"GPU {device_id} memory usage critical: {memory_percent:.1f}%"
                    }
                    self.alerts.append(alert)
                
                # High temperature warning
                temperature = gpu_metric.get('temperature_c', 0)
                if temperature > 80:
                    alert = {
                        'timestamp': current_time,
                        'type': 'gpu_temperature_high',
                        'device_id': device_id,
                        'value': temperature,
                        'message': f"GPU {device_id} temperature high: {temperature}°C"
                    }
                    self.alerts.append(alert)
                
                # Low utilization warning (potential bottleneck)
                utilization = gpu_metric.get('utilization_percent', 0)
                if utilization < 20 and memory_percent > 50:
                    alert = {
                        'timestamp': current_time,
                        'type': 'gpu_underutilized',
                        'device_id': device_id,
                        'utilization': utilization,
                        'memory': memory_percent,
                        'message': f"GPU {device_id} underutilized: {utilization:.1f}% usage, {memory_percent:.1f}% memory"
                    }
                    self.alerts.append(alert)
            
            # System bottleneck detection
            cpu_percent = system_metrics.get('cpu_percent', 0)
            memory_percent = system_metrics.get('memory_percent', 0)
            
            if cpu_percent > 90:
                alert = {
                    'timestamp': current_time,
                    'type': 'cpu_high',
                    'value': cpu_percent,
                    'message': f"High CPU usage: {cpu_percent:.1f}%"
                }
                self.alerts.append(alert)
            
            if memory_percent > 90:
                alert = {
                    'timestamp': current_time,
                    'type': 'memory_high',
                    'value': memory_percent,
                    'message': f"High system memory usage: {memory_percent:.1f}%"
                }
                self.alerts.append(alert)
            
            # Keep only recent alerts (last 100)
            self.alerts = self.alerts[-100:]
            
        except Exception as e:
            print(f"⚠️ Performance analysis failed: {e}")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current GPU and system usage"""
        try:
            gpu_metrics = self._collect_gpu_metrics()
            system_metrics = self._collect_system_metrics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'gpu_devices': gpu_metrics,
                'system': system_metrics,
                'device_count': self.device_count
            }
            
        except Exception as e:
            print(f"❌ Failed to get current usage: {e}")
            return {}
    
    def get_usage_history(self, minutes: int = 10) -> Dict[str, Any]:
        """
        Get usage history for the last N minutes
        
        Args:
            minutes: Number of minutes of history to return
        
        Returns:
            Dictionary with historical usage data
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Filter recent GPU history
            recent_gpu = [
                entry for entry in self.gpu_history
                if entry['timestamp'] > cutoff_time
            ]
            
            # Filter recent system history
            recent_system = [
                entry for entry in self.system_history
                if entry['timestamp'] > cutoff_time
            ]
            
            return {
                'time_range_minutes': minutes,
                'gpu_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'metrics': entry['metrics']
                    }
                    for entry in recent_gpu
                ],
                'system_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'metrics': entry['metrics']
                    }
                    for entry in recent_system
                ],
                'sample_count': len(recent_gpu)
            }
            
        except Exception as e:
            print(f"❌ Failed to get usage history: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics"""
        try:
            if not self.gpu_history:
                return {'error': 'No monitoring data available'}
            
            # Calculate averages for each GPU
            gpu_stats = {}
            
            for entry in self.gpu_history:
                for gpu_metric in entry['metrics']:
                    device_id = gpu_metric['device_id']
                    if device_id not in gpu_stats:
                        gpu_stats[device_id] = {
                            'utilization_samples': [],
                            'memory_samples': [],
                            'temperature_samples': []
                        }
                    
                    gpu_stats[device_id]['utilization_samples'].append(
                        gpu_metric.get('utilization_percent', 0)
                    )
                    gpu_stats[device_id]['memory_samples'].append(
                        gpu_metric.get('memory_percent', 0)
                    )
                    if 'temperature_c' in gpu_metric:
                        gpu_stats[device_id]['temperature_samples'].append(
                            gpu_metric['temperature_c']
                        )
            
            # Calculate statistics
            summary = {
                'monitoring_duration_minutes': len(self.gpu_history) * self.sampling_interval / 60,
                'sample_count': len(self.gpu_history),
                'devices': {}
            }
            
            for device_id, stats in gpu_stats.items():
                if stats['utilization_samples']:
                    device_summary = {
                        'avg_utilization': round(sum(stats['utilization_samples']) / len(stats['utilization_samples']), 1),
                        'max_utilization': round(max(stats['utilization_samples']), 1),
                        'avg_memory_usage': round(sum(stats['memory_samples']) / len(stats['memory_samples']), 1),
                        'max_memory_usage': round(max(stats['memory_samples']), 1)
                    }
                    
                    if stats['temperature_samples']:
                        device_summary.update({
                            'avg_temperature': round(sum(stats['temperature_samples']) / len(stats['temperature_samples']), 1),
                            'max_temperature': round(max(stats['temperature_samples']), 1)
                        })
                    
                    summary['devices'][device_id] = device_summary
            
            # Add alert summary
            recent_alerts = [
                alert for alert in self.alerts
                if alert['timestamp'] > datetime.now() - timedelta(minutes=30)
            ]
            
            summary['recent_alerts'] = len(recent_alerts)
            summary['alert_types'] = list(set(alert['type'] for alert in recent_alerts))
            
            return summary
            
        except Exception as e:
            print(f"❌ Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on monitoring data"""
        recommendations = []
        
        try:
            if not self.gpu_history:
                return ["No monitoring data available for recommendations"]
            
            current_usage = self.get_current_usage()
            gpu_devices = current_usage.get('gpu_devices', [])
            
            for gpu in gpu_devices:
                device_id = gpu['device_id']
                utilization = gpu.get('utilization_percent', 0)
                memory_percent = gpu.get('memory_percent', 0)
                
                # Low utilization recommendations
                if utilization < 30 and memory_percent > 60:
                    recommendations.append(
                        f"GPU {device_id}: Low utilization ({utilization:.1f}%) with high memory usage ({memory_percent:.1f}%) - "
                        "consider increasing batch size or model complexity"
                    )
                
                # High memory usage recommendations
                if memory_percent > 95:
                    recommendations.append(
                        f"GPU {device_id}: Critical memory usage ({memory_percent:.1f}%) - "
                        "reduce batch size or enable gradient checkpointing"
                    )
                elif memory_percent > 85:
                    recommendations.append(
                        f"GPU {device_id}: High memory usage ({memory_percent:.1f}%) - "
                        "consider reducing batch size for stability"
                    )
                
                # Temperature recommendations
                temperature = gpu.get('temperature_c', 0)
                if temperature > 85:
                    recommendations.append(
                        f"GPU {device_id}: High temperature ({temperature}°C) - "
                        "check cooling and reduce workload if necessary"
                    )
            
            # Multi-GPU recommendations
            if len(gpu_devices) > 1:
                utilizations = [gpu.get('utilization_percent', 0) for gpu in gpu_devices]
                util_std = (sum((u - sum(utilizations)/len(utilizations))**2 for u in utilizations) / len(utilizations))**0.5
                
                if util_std > 20:
                    recommendations.append(
                        "Uneven GPU utilization detected - check data loading and model parallelization"
                    )
            
            # System bottleneck recommendations
            system = current_usage.get('system', {})
            if system.get('cpu_percent', 0) > 90:
                recommendations.append("High CPU usage - consider optimizing data preprocessing or using more workers")
            
            if system.get('memory_percent', 0) > 90:
                recommendations.append("High system memory usage - reduce dataset size in memory or use data streaming")
            
            if not recommendations:
                recommendations.append("GPU utilization looks optimal - no specific recommendations")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def export_monitoring_data(self, filepath: str, include_history: bool = True):
        """
        Export monitoring data to file
        
        Args:
            filepath: Path to save the data
            include_history: Whether to include full history or just summary
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'experiment_id': self.experiment_id,
                'device_info': self.devices_info,
                'monitoring_config': {
                    'sampling_interval': self.sampling_interval,
                    'history_size': self.history_size,
                    'gpu_library': self.gpu_library
                },
                'performance_summary': self.get_performance_summary(),
                'current_usage': self.get_current_usage(),
                'recommendations': self.get_optimization_recommendations(),
                'recent_alerts': [
                    {
                        'timestamp': alert['timestamp'].isoformat(),
                        **{k: v for k, v in alert.items() if k != 'timestamp'}
                    }
                    for alert in self.alerts[-20:]  # Last 20 alerts
                ]
            }
            
            if include_history:
                export_data['gpu_history'] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'metrics': entry['metrics']
                    }
                    for entry in self.gpu_history
                ]
                export_data['system_history'] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'metrics': entry['metrics']
                    }
                    for entry in self.system_history
                ]
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📁 Exported monitoring data to: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to export monitoring data: {e}")
    
    def cleanup(self):
        """Clean up monitoring resources"""
        self.stop_monitoring()
        self.gpu_history.clear()
        self.system_history.clear()
        self.alerts.clear()
        print("🧹 GPU monitor cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()