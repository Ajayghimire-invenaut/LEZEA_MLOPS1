"""
Environment Tagger for LeZeA MLOps
=================================

Comprehensive environment detection and tagging system:
- Hardware specifications (CPU, GPU, memory)
- Software environment (Python, libraries, OS)
- Code version control information (Git)
- Dataset versions and hashes
- Container and virtualization detection
- Cloud platform identification

Provides structured tags for MLflow and metadata for other systems.
"""

import os
import sys
import platform
import subprocess
import hashlib
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

# Library imports with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    pkg_resources = None
    PKG_RESOURCES_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class EnvironmentTagger:
    """
    Comprehensive environment detection and tagging system
    
    This class provides:
    - Complete hardware and software profiling
    - Version control integration
    - Cloud platform detection
    - Reproducibility metadata generation
    - MLflow-compatible tag formatting
    - Environment comparison capabilities
    """
    
    def __init__(self):
        """Initialize environment tagger"""
        self.cache = {}  # Cache for expensive operations
        self.git_available = self._check_git_available()
        self.docker_available = self._check_docker_available()
        
        print("🔍 Environment tagger initialized")
    
    def _check_git_available(self) -> bool:
        """Check if Git is available"""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_docker_available(self) -> bool:
        """Check if running in Docker container"""
        try:
            return (
                Path('/.dockerenv').exists() or
                'docker' in Path('/proc/1/cgroup').read_text() or
                os.getenv('DOCKER_CONTAINER') is not None
            )
        except:
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get comprehensive environment information
        
        Returns:
            Dictionary with complete environment details
        """
        env_info = {
            'collection_timestamp': datetime.now().isoformat(),
            'system': self._get_system_info(),
            'hardware': self._get_hardware_info(),
            'software': self._get_software_info(),
            'python': self._get_python_info(),
            'packages': self._get_package_info(),
            'git': self._get_git_info(),
            'environment': self._get_environment_variables(),
            'platform': self._get_platform_info(),
            'cloud': self._detect_cloud_platform(),
            'container': self._get_container_info()
        }
        
        return env_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        try:
            system_info = {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'node': platform.node(),
                'boot_time': None,
                'uptime_seconds': None
            }
            
            if PSUTIL_AVAILABLE:
                boot_time = psutil.boot_time()
                system_info['boot_time'] = datetime.fromtimestamp(boot_time).isoformat()
                system_info['uptime_seconds'] = int(datetime.now().timestamp() - boot_time)
            
            return system_info
            
        except Exception as e:
            return {'error': f'Failed to get system info: {e}'}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information"""
        try:
            hardware_info = {
                'cpu': self._get_cpu_info(),
                'memory': self._get_memory_info(),
                'gpu': self._get_gpu_info(),
                'disk': self._get_disk_info(),
                'network': self._get_network_info()
            }
            
            return hardware_info
            
        except Exception as e:
            return {'error': f'Failed to get hardware info: {e}'}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            cpu_info = {
                'physical_cores': None,
                'logical_cores': None,
                'max_frequency': None,
                'current_frequency': None,
                'usage_percent': None,
                'model': platform.processor()
            }
            
            if PSUTIL_AVAILABLE:
                cpu_info.update({
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'usage_percent': psutil.cpu_percent(interval=1)
                })
                
                # CPU frequency
                try:
                    freq = psutil.cpu_freq()
                    if freq:
                        cpu_info.update({
                            'max_frequency': freq.max,
                            'current_frequency': freq.current
                        })
                except:
                    pass
            
            return cpu_info
            
        except Exception as e:
            return {'error': f'Failed to get CPU info: {e}'}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            memory_info = {
                'total_gb': None,
                'available_gb': None,
                'used_gb': None,
                'usage_percent': None,
                'swap_total_gb': None,
                'swap_used_gb': None
            }
            
            if PSUTIL_AVAILABLE:
                # Virtual memory
                vm = psutil.virtual_memory()
                memory_info.update({
                    'total_gb': round(vm.total / (1024**3), 2),
                    'available_gb': round(vm.available / (1024**3), 2),
                    'used_gb': round(vm.used / (1024**3), 2),
                    'usage_percent': vm.percent
                })
                
                # Swap memory
                try:
                    swap = psutil.swap_memory()
                    memory_info.update({
                        'swap_total_gb': round(swap.total / (1024**3), 2),
                        'swap_used_gb': round(swap.used / (1024**3), 2)
                    })
                except:
                    pass
            
            return memory_info
            
        except Exception as e:
            return {'error': f'Failed to get memory info: {e}'}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpu_info = {
                'available': False,
                'count': 0,
                'devices': [],
                'cuda_available': False,
                'cuda_version': None,
                'driver_version': None
            }
            
            # Try different GPU libraries
            devices = []
            
            # GPUtil
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        devices.append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total_mb': gpu.memoryTotal,
                            'memory_used_mb': gpu.memoryUsed,
                            'memory_free_mb': gpu.memoryFree,
                            'load_percent': round(gpu.load * 100, 1),
                            'temperature_c': gpu.temperature,
                            'uuid': getattr(gpu, 'uuid', None)
                        })
                except:
                    pass
            
            # pynvml
            if PYNVML_AVAILABLE and not devices:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        devices.append({
                            'id': i,
                            'name': name,
                            'memory_total_mb': memory_info.total // (1024**2),
                            'memory_used_mb': memory_info.used // (1024**2),
                            'memory_free_mb': memory_info.free // (1024**2)
                        })
                    
                    # Get driver version
                    try:
                        gpu_info['driver_version'] = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    except:
                        pass
                        
                except:
                    pass
            
            # PyTorch CUDA
            if TORCH_AVAILABLE:
                try:
                    gpu_info['cuda_available'] = torch.cuda.is_available()
                    if gpu_info['cuda_available']:
                        gpu_info['cuda_version'] = torch.version.cuda
                        device_count = torch.cuda.device_count()
                        
                        if not devices:  # Only if we haven't found devices yet
                            for i in range(device_count):
                                props = torch.cuda.get_device_properties(i)
                                devices.append({
                                    'id': i,
                                    'name': props.name,
                                    'memory_total_mb': props.total_memory // (1024**2),
                                    'compute_capability': f"{props.major}.{props.minor}",
                                    'multiprocessor_count': props.multi_processor_count
                                })
                except:
                    pass
            
            if devices:
                gpu_info.update({
                    'available': True,
                    'count': len(devices),
                    'devices': devices
                })
            
            return gpu_info
            
        except Exception as e:
            return {'error': f'Failed to get GPU info: {e}'}
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information"""
        try:
            disk_info = {
                'partitions': [],
                'total_gb': 0,
                'used_gb': 0,
                'free_gb': 0
            }
            
            if PSUTIL_AVAILABLE:
                partitions = psutil.disk_partitions()
                
                for partition in partitions:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        partition_info = {
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'fstype': partition.fstype,
                            'total_gb': round(usage.total / (1024**3), 2),
                            'used_gb': round(usage.used / (1024**3), 2),
                            'free_gb': round(usage.free / (1024**3), 2),
                            'usage_percent': round((usage.used / usage.total) * 100, 1)
                        }
                        disk_info['partitions'].append(partition_info)
                        
                        # Sum up main drives
                        if partition.mountpoint in ['/', 'C:\\']:
                            disk_info['total_gb'] += partition_info['total_gb']
                            disk_info['used_gb'] += partition_info['used_gb']
                            disk_info['free_gb'] += partition_info['free_gb']
                            
                    except PermissionError:
                        continue
            
            return disk_info
            
        except Exception as e:
            return {'error': f'Failed to get disk info: {e}'}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network interface information"""
        try:
            network_info = {
                'interfaces': [],
                'hostname': socket.gethostname(),
                'fqdn': socket.getfqdn()
            }
            
            if PSUTIL_AVAILABLE:
                interfaces = psutil.net_if_addrs()
                
                for interface_name, addresses in interfaces.items():
                    interface_info = {
                        'name': interface_name,
                        'addresses': []
                    }
                    
                    for addr in addresses:
                        interface_info['addresses'].append({
                            'family': str(addr.family),
                            'address': addr.address,
                            'netmask': getattr(addr, 'netmask', None),
                            'broadcast': getattr(addr, 'broadcast', None)
                        })
                    
                    network_info['interfaces'].append(interface_info)
            
            return network_info
            
        except Exception as e:
            return {'error': f'Failed to get network info: {e}'}
    
    def _get_software_info(self) -> Dict[str, Any]:
        """Get software environment information"""
        try:
            software_info = {
                'os_name': os.name,
                'os_environ_count': len(os.environ),
                'shell': os.getenv('SHELL'),
                'terminal': os.getenv('TERM'),
                'user': os.getenv('USER') or os.getenv('USERNAME'),
                'home': os.getenv('HOME') or os.getenv('USERPROFILE'),
                'path_entries': len(os.getenv('PATH', '').split(os.pathsep)),
                'locale': {
                    'lang': os.getenv('LANG'),
                    'lc_all': os.getenv('LC_ALL'),
                    'timezone': os.getenv('TZ')
                }
            }
            
            return software_info
            
        except Exception as e:
            return {'error': f'Failed to get software info: {e}'}
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information"""
        try:
            python_info = {
                'version': sys.version,
                'version_info': {
                    'major': sys.version_info.major,
                    'minor': sys.version_info.minor,
                    'micro': sys.version_info.micro
                },
                'executable': sys.executable,
                'prefix': sys.prefix,
                'path': sys.path[:5],  # First 5 path entries
                'path_count': len(sys.path),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler(),
                'build': platform.python_build(),
                'api_version': sys.api_version,
                'maxsize': sys.maxsize,
                'float_info': {
                    'max': sys.float_info.max,
                    'epsilon': sys.float_info.epsilon,
                    'dig': sys.float_info.dig
                }
            }
            
            # Virtual environment detection
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                python_info['virtual_env'] = True
                python_info['virtual_env_path'] = sys.prefix
            else:
                python_info['virtual_env'] = False
            
            # Conda environment detection
            conda_env = os.getenv('CONDA_DEFAULT_ENV')
            if conda_env:
                python_info['conda_env'] = conda_env
                python_info['conda_prefix'] = os.getenv('CONDA_PREFIX')
            
            return python_info
            
        except Exception as e:
            return {'error': f'Failed to get Python info: {e}'}
    
    def _get_package_info(self) -> Dict[str, Any]:
        """Get installed package information"""
        try:
            package_info = {
                'package_count': 0,
                'key_packages': {},
                'pip_packages': []
            }
            
            # Key packages to specifically track
            key_packages = [
                'torch', 'tensorflow', 'numpy', 'pandas', 'scikit-learn',
                'matplotlib', 'seaborn', 'jupyter', 'mlflow', 'dvc',
                'boto3', 'pymongo', 'psycopg2', 'requests', 'fastapi'
            ]
            
            if PKG_RESOURCES_AVAILABLE:
                installed_packages = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
                package_info['package_count'] = len(installed_packages)
                
                # Track key packages
                for pkg_name in key_packages:
                    if pkg_name in installed_packages:
                        package_info['key_packages'][pkg_name] = installed_packages[pkg_name]
                
                # Get all packages (limited to prevent overflow)
                package_info['pip_packages'] = [
                    f"{name}=={version}" for name, version in 
                    list(installed_packages.items())[:100]  # Limit to 100 packages
                ]
            
            return package_info
            
        except Exception as e:
            return {'error': f'Failed to get package info: {e}'}
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information"""
        try:
            git_info = {
                'available': self.git_available,
                'is_repo': False,
                'commit': None,
                'branch': None,
                'remote_url': None,
                'status': None,
                'tag': None,
                'commit_count': None,
                'last_commit_date': None
            }
            
            if not self.git_available:
                return git_info
            
            # Check if we're in a git repository
            try:
                subprocess.run(['git', 'rev-parse', '--git-dir'], 
                             capture_output=True, check=True)
                git_info['is_repo'] = True
            except subprocess.CalledProcessError:
                return git_info
            
            # Get commit hash
            try:
                result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      capture_output=True, text=True, check=True)
                git_info['commit'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
            
            # Get branch name
            try:
                result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                      capture_output=True, text=True, check=True)
                git_info['branch'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
            
            # Get remote URL
            try:
                result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                                      capture_output=True, text=True, check=True)
                git_info['remote_url'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
            
            # Get status
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, check=True)
                git_info['status'] = 'clean' if not result.stdout.strip() else 'dirty'
            except subprocess.CalledProcessError:
                pass
            
            # Get latest tag
            try:
                result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                                      capture_output=True, text=True, check=True)
                git_info['tag'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
            
            # Get commit count
            try:
                result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                      capture_output=True, text=True, check=True)
                git_info['commit_count'] = int(result.stdout.strip())
            except subprocess.CalledProcessError:
                pass
            
            # Get last commit date
            try:
                result = subprocess.run(['git', 'log', '-1', '--format=%ci'], 
                                      capture_output=True, text=True, check=True)
                git_info['last_commit_date'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
            
            return git_info
            
        except Exception as e:
            return {'error': f'Failed to get git info: {e}'}
    
    def _get_environment_variables(self) -> Dict[str, Any]:
        """Get relevant environment variables"""
        try:
            # Key environment variables to track
            key_vars = [
                'PATH', 'PYTHONPATH', 'HOME', 'USER', 'SHELL',
                'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV',
                'AWS_REGION', 'AWS_DEFAULT_REGION',
                'MLFLOW_TRACKING_URI', 'MONGO_CONNECTION_STRING'
            ]
            
            env_vars = {}
            for var in key_vars:
                value = os.getenv(var)
                if value is not None:
                    # Truncate very long values
                    if len(value) > 200:
                        value = value[:197] + "..."
                    env_vars[var] = value
            
            return {
                'tracked_variables': env_vars,
                'total_variables': len(os.environ)
            }
            
        except Exception as e:
            return {'error': f'Failed to get environment variables: {e}'}
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get platform-specific information"""
        try:
            platform_info = {
                'system': platform.system(),
                'platform_details': {}
            }
            
            system = platform.system()
            
            if system == 'Linux':
                platform_info['platform_details'] = self._get_linux_info()
            elif system == 'Darwin':
                platform_info['platform_details'] = self._get_macos_info()
            elif system == 'Windows':
                platform_info['platform_details'] = self._get_windows_info()
            
            return platform_info
            
        except Exception as e:
            return {'error': f'Failed to get platform info: {e}'}
    
    def _get_linux_info(self) -> Dict[str, Any]:
        """Get Linux-specific information"""
        try:
            linux_info = {}
            
            # Distribution info
            try:
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('ID='):
                            linux_info['distro_id'] = line.split('=')[1].strip().strip('"')
                        elif line.startswith('VERSION_ID='):
                            linux_info['distro_version'] = line.split('=')[1].strip().strip('"')
                        elif line.startswith('PRETTY_NAME='):
                            linux_info['distro_name'] = line.split('=')[1].strip().strip('"')
            except:
                pass
            
            # Kernel info
            linux_info['kernel_version'] = platform.release()
            
            # CPU info from /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    if 'model name' in content:
                        for line in content.split('\n'):
                            if line.startswith('model name'):
                                linux_info['cpu_model'] = line.split(':')[1].strip()
                                break
            except:
                pass
            
            return linux_info
            
        except Exception as e:
            return {'error': f'Failed to get Linux info: {e}'}
    
    def _get_macos_info(self) -> Dict[str, Any]:
        """Get macOS-specific information"""
        try:
            macos_info = {
                'version': platform.mac_ver()[0],
                'machine': platform.machine()
            }
            
            # Try to get more detailed system info
            try:
                result = subprocess.run(['sw_vers'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('ProductName:'):
                            macos_info['product_name'] = line.split(':')[1].strip()
                        elif line.startswith('ProductVersion:'):
                            macos_info['product_version'] = line.split(':')[1].strip()
                        elif line.startswith('BuildVersion:'):
                            macos_info['build_version'] = line.split(':')[1].strip()
            except:
                pass
            
            return macos_info
            
        except Exception as e:
            return {'error': f'Failed to get macOS info: {e}'}
    
    def _get_windows_info(self) -> Dict[str, Any]:
        """Get Windows-specific information"""
        try:
            windows_info = {
                'version': platform.version(),
                'edition': platform.win32_edition() if hasattr(platform, 'win32_edition') else None,
                'is_64bit': platform.machine().endswith('64')
            }
            
            return windows_info
            
        except Exception as e:
            return {'error': f'Failed to get Windows info: {e}'}
    
    def _detect_cloud_platform(self) -> Dict[str, Any]:
        """Detect if running on a cloud platform"""
        try:
            cloud_info = {
                'detected': False,
                'platform': None,
                'instance_type': None,
                'region': None,
                'availability_zone': None
            }
            
            # AWS detection
            try:
                # Check for AWS metadata service
                import urllib.request
                import urllib.error
                
                request = urllib.request.Request(
                    'http://169.254.169.254/latest/meta-data/instance-type',
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    cloud_info.update({
                        'detected': True,
                        'platform': 'AWS',
                        'instance_type': response.read().decode('utf-8')
                    })
                    
                    # Get additional AWS metadata
                    try:
                        region_request = urllib.request.Request(
                            'http://169.254.169.254/latest/meta-data/placement/region'
                        )
                        with urllib.request.urlopen(region_request, timeout=1) as region_response:
                            cloud_info['region'] = region_response.read().decode('utf-8')
                    except:
                        pass
                        
                    try:
                        az_request = urllib.request.Request(
                            'http://169.254.169.254/latest/meta-data/placement/availability-zone'
                        )
                        with urllib.request.urlopen(az_request, timeout=1) as az_response:
                            cloud_info['availability_zone'] = az_response.read().decode('utf-8')
                    except:
                        pass
                        
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            
            # GCP detection
            if not cloud_info['detected']:
                try:
                    request = urllib.request.Request(
                        'http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                        headers={'Metadata-Flavor': 'Google'}
                    )
                    with urllib.request.urlopen(request, timeout=2) as response:
                        machine_type = response.read().decode('utf-8')
                        cloud_info.update({
                            'detected': True,
                            'platform': 'GCP',
                            'instance_type': machine_type.split('/')[-1]
                        })
                except:
                    pass
            
            # Azure detection
            if not cloud_info['detected']:
                try:
                    request = urllib.request.Request(
                        'http://169.254.169.254/metadata/instance/compute/vmSize',
                        headers={'Metadata': 'true'}
                    )
                    with urllib.request.urlopen(request, timeout=2) as response:
                        cloud_info.update({
                            'detected': True,
                            'platform': 'Azure',
                            'instance_type': response.read().decode('utf-8')
                        })
                except:
                    pass
            
            return cloud_info
            
        except Exception as e:
            return {'error': f'Failed to detect cloud platform: {e}'}
    
    def _get_container_info(self) -> Dict[str, Any]:
        """Get container environment information"""
        try:
            container_info = {
                'docker': self.docker_available,
                'kubernetes': False,
                'container_id': None,
                'image_name': None
            }
            
            # Kubernetes detection
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
                container_info['kubernetes'] = True
                
                # Try to get namespace
                try:
                    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as f:
                        container_info['k8s_namespace'] = f.read().strip()
                except:
                    pass
            
            # Docker container ID
            if self.docker_available:
                try:
                    with open('/proc/self/cgroup', 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if 'docker' in line:
                                container_info['container_id'] = line.split('/')[-1][:12]
                                break
                except:
                    pass
            
            return container_info
            
        except Exception as e:
            return {'error': f'Failed to get container info: {e}'}
    
    def get_mlflow_tags(self, env_info: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Convert environment info to MLflow-compatible tags
        
        Args:
            env_info: Environment info dictionary (if None, will collect)
        
        Returns:
            Dictionary with MLflow tags
        """
        if env_info is None:
            env_info = self.get_environment_info()
        
        try:
            tags = {}
            
            # System tags
            system = env_info.get('system', {})
            tags['system.hostname'] = system.get('hostname', 'unknown')
            tags['system.platform'] = system.get('system', 'unknown')
            tags['system.architecture'] = system.get('architecture', 'unknown')
            
            # Hardware tags
            hardware = env_info.get('hardware', {})
            
            # CPU tags
            cpu = hardware.get('cpu', {})
            if cpu.get('physical_cores'):
                tags['hardware.cpu_cores'] = str(cpu['physical_cores'])
            if cpu.get('logical_cores'):
                tags['hardware.cpu_threads'] = str(cpu['logical_cores'])
            
            # Memory tags
            memory = hardware.get('memory', {})
            if memory.get('total_gb'):
                tags['hardware.memory_gb'] = str(int(memory['total_gb']))
            
            # GPU tags
            gpu = hardware.get('gpu', {})
            tags['hardware.gpu_available'] = str(gpu.get('available', False))
            if gpu.get('count'):
                tags['hardware.gpu_count'] = str(gpu['count'])
            if gpu.get('devices'):
                gpu_names = [device.get('name', 'unknown') for device in gpu['devices'][:3]]  # Max 3
                tags['hardware.gpu_types'] = ','.join(gpu_names)
            if gpu.get('cuda_available'):
                tags['hardware.cuda_available'] = str(gpu['cuda_available'])
            if gpu.get('cuda_version'):
                tags['hardware.cuda_version'] = gpu['cuda_version']
            
            # Python tags
            python = env_info.get('python', {})
            version_info = python.get('version_info', {})
            if version_info:
                tags['python.version'] = f"{version_info.get('major', 0)}.{version_info.get('minor', 0)}.{version_info.get('micro', 0)}"
            tags['python.implementation'] = python.get('implementation', 'unknown')
            tags['python.virtual_env'] = str(python.get('virtual_env', False))
            if python.get('conda_env'):
                tags['python.conda_env'] = python['conda_env']
            
            # Package tags
            packages = env_info.get('packages', {})
            key_packages = packages.get('key_packages', {})
            for pkg_name, version in key_packages.items():
                if pkg_name in ['torch', 'tensorflow', 'numpy', 'pandas', 'scikit-learn']:
                    tags[f'package.{pkg_name}'] = version
            
            # Git tags
            git = env_info.get('git', {})
            if git.get('is_repo'):
                tags['git.available'] = 'true'
                if git.get('commit'):
                    tags['git.commit'] = git['commit'][:8]  # Short hash
                if git.get('branch'):
                    tags['git.branch'] = git['branch']
                if git.get('status'):
                    tags['git.status'] = git['status']
                if git.get('tag'):
                    tags['git.tag'] = git['tag']
            else:
                tags['git.available'] = 'false'
            
            # Cloud tags
            cloud = env_info.get('cloud', {})
            if cloud.get('detected'):
                tags['cloud.platform'] = cloud.get('platform', 'unknown')
                if cloud.get('instance_type'):
                    tags['cloud.instance_type'] = cloud['instance_type']
                if cloud.get('region'):
                    tags['cloud.region'] = cloud['region']
            else:
                tags['cloud.platform'] = 'local'
            
            # Container tags
            container = env_info.get('container', {})
            tags['container.docker'] = str(container.get('docker', False))
            tags['container.kubernetes'] = str(container.get('kubernetes', False))
            
            # Platform-specific tags
            platform_info = env_info.get('platform', {})
            if platform_info.get('platform_details'):
                details = platform_info['platform_details']
                if 'distro_name' in details:
                    tags['os.distribution'] = details['distro_name']
                elif 'product_name' in details:
                    tags['os.product'] = details['product_name']
            
            # Environment variable tags
            env_vars = env_info.get('environment', {}).get('tracked_variables', {})
            if 'CUDA_VISIBLE_DEVICES' in env_vars:
                tags['env.cuda_visible_devices'] = env_vars['CUDA_VISIBLE_DEVICES']
            if 'OMP_NUM_THREADS' in env_vars:
                tags['env.omp_threads'] = env_vars['OMP_NUM_THREADS']
            
            # Convert all values to strings and limit length
            final_tags = {}
            for key, value in tags.items():
                str_value = str(value)
                if len(str_value) > 250:  # MLflow tag limit
                    str_value = str_value[:247] + "..."
                final_tags[key] = str_value
            
            return final_tags
            
        except Exception as e:
            return {'environment.error': f'Failed to generate tags: {e}'}
    
    def compare_environments(self, other_env_file: str) -> Dict[str, Any]:
        """
        Compare current environment with another environment file
        
        Args:
            other_env_file: Path to another environment info file
        
        Returns:
            Dictionary with comparison results
        """
        try:
            current_env = self.get_environment_info()
            
            with open(other_env_file, 'r') as f:
                other_env = json.load(f)
            
            comparison = {
                'comparison_timestamp': datetime.now().isoformat(),
                'current_file': 'current_environment',
                'other_file': other_env_file,
                'differences': {},
                'similarities': {},
                'compatibility_score': 0.0
            }
            
            # Compare key sections
            sections_to_compare = ['system', 'hardware', 'python', 'packages']
            total_score = 0
            max_score = len(sections_to_compare)
            
            for section in sections_to_compare:
                current_section = current_env.get(section, {})
                other_section = other_env.get(section, {})
                
                section_diff, section_sim, section_score = self._compare_section(
                    current_section, other_section, section
                )
                
                if section_diff:
                    comparison['differences'][section] = section_diff
                if section_sim:
                    comparison['similarities'][section] = section_sim
                
                total_score += section_score
            
            comparison['compatibility_score'] = total_score / max_score if max_score > 0 else 0.0
            
            return comparison
            
        except Exception as e:
            return {'error': f'Failed to compare environments: {e}'}
    
    def _compare_section(self, current: Dict, other: Dict, section_name: str) -> Tuple[Dict, Dict, float]:
        """Compare a specific section of environment info"""
        differences = {}
        similarities = {}
        score = 0.0
        
        try:
            if section_name == 'system':
                # Compare system info
                if current.get('platform') == other.get('platform'):
                    similarities['platform'] = current.get('platform')
                    score += 0.5
                else:
                    differences['platform'] = {
                        'current': current.get('platform'),
                        'other': other.get('platform')
                    }
                
                if current.get('architecture') == other.get('architecture'):
                    similarities['architecture'] = current.get('architecture')
                    score += 0.5
                else:
                    differences['architecture'] = {
                        'current': current.get('architecture'),
                        'other': other.get('architecture')
                    }
            
            elif section_name == 'hardware':
                # Compare CPU
                current_cpu = current.get('cpu', {})
                other_cpu = other.get('cpu', {})
                
                if current_cpu.get('physical_cores') == other_cpu.get('physical_cores'):
                    similarities['cpu_cores'] = current_cpu.get('physical_cores')
                    score += 0.25
                else:
                    differences['cpu_cores'] = {
                        'current': current_cpu.get('physical_cores'),
                        'other': other_cpu.get('physical_cores')
                    }
                
                # Compare memory
                current_mem = current.get('memory', {})
                other_mem = other.get('memory', {})
                
                current_mem_gb = current_mem.get('total_gb', 0)
                other_mem_gb = other_mem.get('total_gb', 0)
                
                if abs(current_mem_gb - other_mem_gb) < 1:  # Within 1GB
                    similarities['memory_gb'] = current_mem_gb
                    score += 0.25
                else:
                    differences['memory_gb'] = {
                        'current': current_mem_gb,
                        'other': other_mem_gb
                    }
                
                # Compare GPU
                current_gpu = current.get('gpu', {})
                other_gpu = other.get('gpu', {})
                
                if current_gpu.get('available') == other_gpu.get('available'):
                    similarities['gpu_available'] = current_gpu.get('available')
                    score += 0.25
                    
                    if current_gpu.get('available'):
                        if current_gpu.get('count') == other_gpu.get('count'):
                            similarities['gpu_count'] = current_gpu.get('count')
                            score += 0.25
                        else:
                            differences['gpu_count'] = {
                                'current': current_gpu.get('count'),
                                'other': other_gpu.get('count')
                            }
                else:
                    differences['gpu_available'] = {
                        'current': current_gpu.get('available'),
                        'other': other_gpu.get('available')
                    }
            
            elif section_name == 'python':
                # Compare Python version
                current_version = current.get('version_info', {})
                other_version = other.get('version_info', {})
                
                if (current_version.get('major') == other_version.get('major') and
                    current_version.get('minor') == other_version.get('minor')):
                    similarities['python_version'] = f"{current_version.get('major')}.{current_version.get('minor')}"
                    score += 0.5
                else:
                    differences['python_version'] = {
                        'current': f"{current_version.get('major', 0)}.{current_version.get('minor', 0)}",
                        'other': f"{other_version.get('major', 0)}.{other_version.get('minor', 0)}"
                    }
                
                # Compare virtual environment
                if current.get('virtual_env') == other.get('virtual_env'):
                    similarities['virtual_env'] = current.get('virtual_env')
                    score += 0.5
                else:
                    differences['virtual_env'] = {
                        'current': current.get('virtual_env'),
                        'other': other.get('virtual_env')
                    }
            
            elif section_name == 'packages':
                # Compare key packages
                current_packages = current.get('key_packages', {})
                other_packages = other.get('key_packages', {})
                
                common_packages = set(current_packages.keys()) & set(other_packages.keys())
                different_packages = {}
                
                for pkg in common_packages:
                    if current_packages[pkg] == other_packages[pkg]:
                        similarities[f'package_{pkg}'] = current_packages[pkg]
                    else:
                        different_packages[pkg] = {
                            'current': current_packages[pkg],
                            'other': other_packages[pkg]
                        }
                
                if different_packages:
                    differences['package_versions'] = different_packages
                
                # Score based on package compatibility
                if common_packages:
                    score = len([pkg for pkg in common_packages 
                               if current_packages[pkg] == other_packages[pkg]]) / len(common_packages)
            
            return differences, similarities, score
            
        except Exception as e:
            return {'error': str(e)}, {}, 0.0
    
    def export_environment(self, filepath: str, include_sensitive: bool = False):
        """
        Export environment information to file
        
        Args:
            filepath: Path to save the environment info
            include_sensitive: Whether to include potentially sensitive information
        """
        try:
            env_info = self.get_environment_info()
            
            if not include_sensitive:
                # Remove potentially sensitive information
                if 'environment' in env_info:
                    sensitive_vars = ['AWS_SECRET_ACCESS_KEY', 'MONGO_PASSWORD', 'API_KEY']
                    tracked_vars = env_info['environment'].get('tracked_variables', {})
                    filtered_vars = {k: v for k, v in tracked_vars.items() 
                                   if not any(sens in k.upper() for sens in sensitive_vars)}
                    env_info['environment']['tracked_variables'] = filtered_vars
                
                # Remove network interface details
                if 'hardware' in env_info and 'network' in env_info['hardware']:
                    env_info['hardware']['network'] = {
                        'hostname': env_info['hardware']['network'].get('hostname'),
                        'interface_count': len(env_info['hardware']['network'].get('interfaces', []))
                    }
            
            with open(filepath, 'w') as f:
                json.dump(env_info, f, indent=2, default=str)
            
            print(f"💾 Exported environment info to: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to export environment: {e}")
    
    def get_reproducibility_hash(self) -> str:
        """
        Generate a hash representing the reproducibility context
        
        Returns:
            SHA256 hash of key reproducibility factors
        """
        try:
            env_info = self.get_environment_info()
            
            # Extract key reproducibility factors
            repro_data = {
                'python_version': env_info.get('python', {}).get('version_info', {}),
                'key_packages': env_info.get('packages', {}).get('key_packages', {}),
                'git_commit': env_info.get('git', {}).get('commit'),
                'platform': env_info.get('system', {}).get('platform'),
                'architecture': env_info.get('system', {}).get('architecture')
            }
            
            # Create deterministic string
            repro_string = json.dumps(repro_data, sort_keys=True)
            
            # Generate hash
            hash_obj = hashlib.sha256(repro_string.encode('utf-8'))
            return hash_obj.hexdigest()
            
        except Exception as e:
            return f"error_{hash(str(e))}"
    
    def cleanup(self):
        """Clean up any resources"""
        self.cache.clear()
        print("🧹 Environment tagger cleaned up")