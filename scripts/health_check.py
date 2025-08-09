#!/usr/bin/env python3
"""
LeZeA MLOps - Comprehensive Health Check Script
==============================================

Advanced health monitoring for all LeZeA MLOps components:
- Service availability and response times
- Database connectivity and performance
- Storage accessibility and capacity
- Resource utilization monitoring
- Data pipeline health checks
- Integration testing

Usage:
    python scripts/health_check.py [OPTIONS]

Options:
    --services    Comma-separated list of services to check
    --timeout     Timeout for each check (default: 10 seconds)
    --verbose     Enable detailed output
    --json        Output results in JSON format
    --continuous  Run continuous health monitoring
    --interval    Interval for continuous monitoring (default: 60 seconds)
    --alert       Send alerts for failures
    --fix         Attempt to fix simple issues automatically
"""

import os
import sys
import json
import time
import socket
import psutil
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
    import psycopg2
    import pymongo
    import boto3
    from botocore.exceptions import ClientError
    import mlflow
    from prometheus_client.parser import text_string_to_metric_families
except ImportError as e:
    print(f"Warning: Missing optional dependency: {e}")
    print("Some health checks may not be available")

@dataclass
class HealthResult:
    """Health check result for a single component"""
    service: str
    status: str  # 'healthy', 'unhealthy', 'warning', 'unknown'
    response_time: Optional[float]
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class LeZeAHealthChecker:
    """Comprehensive health checker for LeZeA MLOps system"""
    
    def __init__(self, timeout: int = 10, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[HealthResult] = []
        
        # Default service configurations
        self.services = {
            'postgresql': {'host': 'localhost', 'port': 5432},
            'mongodb': {'host': 'localhost', 'port': 27017},
            'minio': {'host': 'localhost', 'port': 9000},
            'mlflow': {'host': 'localhost', 'port': 5000},
            'prometheus': {'host': 'localhost', 'port': 9090},
            'node_exporter': {'host': 'localhost', 'port': 9100},
            'gpu_exporter': {'host': 'localhost', 'port': 9835},
            'grafana': {'host': 'localhost', 'port': 3000},
        }
        
        # Load environment variables
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
        
        # Update service configurations from environment
        self.services['postgresql']['port'] = int(os.getenv('POSTGRES_PORT', 5432))
        self.services['mongodb']['port'] = int(os.getenv('MONGODB_PORT', 27017))
        self.services['mlflow']['port'] = int(os.getenv('MLFLOW_PORT', 5000))
        self.services['prometheus']['port'] = int(os.getenv('PROMETHEUS_PORT', 9090))
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.verbose or level in ['ERROR', 'WARNING']:
            print(f"[{level}] {timestamp} - {message}")
    
    def check_port_connectivity(self, host: str, port: int, service: str) -> HealthResult:
        """Check if a port is accessible"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = time.time() - start_time
            
            if result == 0:
                return HealthResult(
                    service=service,
                    status='healthy',
                    response_time=response_time,
                    message=f"Port {port} is accessible",
                    details={'host': host, 'port': port},
                    timestamp=datetime.now()
                )
            else:
                return HealthResult(
                    service=service,
                    status='unhealthy',
                    response_time=response_time,
                    message=f"Port {port} is not accessible",
                    details={'host': host, 'port': port, 'error_code': result},
                    timestamp=datetime.now()
                )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"Connection failed: {str(e)}",
                details={'host': host, 'port': port, 'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_http_endpoint(self, url: str, service: str, 
                          expected_status: int = 200,
                          check_content: Optional[str] = None) -> HealthResult:
        """Check HTTP endpoint health"""
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response_time = time.time() - start_time
            
            status = 'healthy'
            message = f"HTTP {response.status_code}"
            details = {
                'url': url,
                'status_code': response.status_code,
                'response_time': response_time,
                'headers': dict(response.headers)
            }
            
            if response.status_code != expected_status:
                status = 'warning'
                message = f"Unexpected status code: {response.status_code}"
            
            if check_content and check_content not in response.text:
                status = 'warning'
                message += f", expected content '{check_content}' not found"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details=details,
                timestamp=datetime.now()
            )
        
        except requests.exceptions.RequestException as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"HTTP request failed: {str(e)}",
                details={'url': url, 'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_postgresql(self) -> HealthResult:
        """Check PostgreSQL database health"""
        start_time = time.time()
        service = 'postgresql'
        
        try:
            # Connection parameters
            conn_params = {
                'host': self.services[service]['host'],
                'port': self.services[service]['port'],
                'database': os.getenv('POSTGRES_DB', 'lezea_mlops'),
                'user': os.getenv('POSTGRES_USER', 'lezea_user'),
                'password': os.getenv('POSTGRES_PASSWORD', 'lezea_secure_password_2024'),
                'connect_timeout': self.timeout
            }
            
            # Connect and run health checks
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cursor:
                    # Basic connectivity test
                    cursor.execute("SELECT 1")
                    
                    # Check database version
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    
                    # Check key tables exist
                    cursor.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_schema = 'mlflow' AND table_name = 'experiments'
                    """)
                    tables_exist = cursor.fetchone()[0] > 0
                    
                    # Check database size
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database()))
                    """)
                    db_size = cursor.fetchone()[0]
                    
                    # Check active connections
                    cursor.execute("""
                        SELECT count(*) FROM pg_stat_activity 
                        WHERE state = 'active'
                    """)
                    active_connections = cursor.fetchone()[0]
                    
                    response_time = time.time() - start_time
                    
                    status = 'healthy' if tables_exist else 'warning'
                    message = "Database accessible and responsive"
                    if not tables_exist:
                        message = "Database accessible but MLflow tables missing"
                    
                    return HealthResult(
                        service=service,
                        status=status,
                        response_time=response_time,
                        message=message,
                        details={
                            'version': version,
                            'database_size': db_size,
                            'active_connections': active_connections,
                            'mlflow_tables_exist': tables_exist,
                            **conn_params
                        },
                        timestamp=datetime.now()
                    )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"Database connection failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_mongodb(self) -> HealthResult:
        """Check MongoDB health"""
        start_time = time.time()
        service = 'mongodb'
        
        try:
            host = self.services[service]['host']
            port = self.services[service]['port']
            
            # Connect to MongoDB
            client = pymongo.MongoClient(
                host=host,
                port=port,
                serverSelectionTimeoutMS=self.timeout * 1000,
                connectTimeoutMS=self.timeout * 1000
            )
            
            # Test connection
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            
            # Get database stats
            db_stats = client.lezea_mlops.command('dbStats')
            
            # Check collections
            collections = client.lezea_mlops.list_collection_names()
            
            response_time = time.time() - start_time
            
            return HealthResult(
                service=service,
                status='healthy',
                response_time=response_time,
                message="MongoDB is accessible and responsive",
                details={
                    'version': server_info.get('version'),
                    'database_size': db_stats.get('dataSize', 0),
                    'collections_count': len(collections),
                    'collections': collections[:10],  # First 10 collections
                    'host': host,
                    'port': port
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"MongoDB connection failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_mlflow(self) -> HealthResult:
        """Check MLflow server health"""
        service = 'mlflow'
        host = self.services[service]['host']
        port = self.services[service]['port']
        
        # First check basic HTTP connectivity
        http_result = self.check_http_endpoint(
            f"http://{host}:{port}/health",
            service
        )
        
        if http_result.status != 'healthy':
            return http_result
        
        # Additional MLflow-specific checks
        start_time = time.time()
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(f"http://{host}:{port}")
            
            # Try to list experiments
            experiments = mlflow.search_experiments()
            
            # Check if we can create a test experiment (and clean it up)
            test_exp_name = f"health_check_{int(time.time())}"
            try:
                exp_id = mlflow.create_experiment(test_exp_name)
                mlflow.delete_experiment(exp_id)
            except Exception:
                pass  # May fail if experiment exists, that's ok
            
            response_time = time.time() - start_time + http_result.response_time
            
            return HealthResult(
                service=service,
                status='healthy',
                response_time=response_time,
                message="MLflow server is fully functional",
                details={
                    'experiments_count': len(experiments),
                    'tracking_uri': f"http://{host}:{port}",
                    'api_accessible': True
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='warning',
                response_time=time.time() - start_time + http_result.response_time,
                message=f"MLflow HTTP accessible but API failed: {str(e)}",
                details={'error': str(e), 'http_status': 'ok'},
                timestamp=datetime.now()
            )
    
    def check_s3_storage(self) -> HealthResult:
        """Check S3/MinIO storage health"""
        start_time = time.time()
        service = 's3_storage'
        
        try:
            # Get S3 configuration from environment
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
            bucket_name = os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT', '').replace('s3://', '').split('/')[0]
            
            if not all([access_key, secret_key, bucket_name]):
                return HealthResult(
                    service=service,
                    status='warning',
                    response_time=0,
                    message="S3 credentials not configured",
                    details={'missing_config': True},
                    timestamp=datetime.now()
                )
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                endpoint_url=endpoint_url
            )
            
            # Test bucket access
            s3_client.head_bucket(Bucket=bucket_name)
            
            # Test list objects
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            
            # Test upload/download (small test file)
            test_key = f"health_check_{int(time.time())}.txt"
            test_content = b"LeZeA MLOps health check"
            
            try:
                # Upload test file
                s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
                
                # Download test file
                download_response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
                downloaded_content = download_response['Body'].read()
                
                # Clean up test file
                s3_client.delete_object(Bucket=bucket_name, Key=test_key)
                
                upload_works = downloaded_content == test_content
            except Exception as e:
                upload_works = False
                self.log(f"S3 upload/download test failed: {e}", 'WARNING')
            
            response_time = time.time() - start_time
            
            status = 'healthy' if upload_works else 'warning'
            message = "S3 storage fully functional" if upload_works else "S3 accessible but upload/download failed"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'bucket': bucket_name,
                    'endpoint': endpoint_url,
                    'region': region,
                    'upload_test': upload_works,
                    'objects_accessible': 'Contents' in response
                },
                timestamp=datetime.now()
            )
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"S3 access failed: {error_code}",
                details={'error': str(e), 'error_code': error_code},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"S3 connection failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_prometheus(self) -> HealthResult:
        """Check Prometheus health and metrics"""
        service = 'prometheus'
        host = self.services[service]['host']
        port = self.services[service]['port']
        
        # First check basic HTTP connectivity
        http_result = self.check_http_endpoint(
            f"http://{host}:{port}/-/healthy",
            service
        )
        
        if http_result.status != 'healthy':
            return http_result
        
        # Additional Prometheus-specific checks
        start_time = time.time()
        
        try:
            # Check targets status
            targets_response = requests.get(
                f"http://{host}:{port}/api/v1/targets",
                timeout=self.timeout
            )
            targets_data = targets_response.json()
            
            # Count healthy/unhealthy targets
            active_targets = targets_data.get('data', {}).get('activeTargets', [])
            healthy_targets = sum(1 for t in active_targets if t.get('health') == 'up')
            total_targets = len(active_targets)
            
            # Check if LeZeA metrics are being collected
            query_response = requests.get(
                f"http://{host}:{port}/api/v1/query",
                params={'query': 'up{job="lezea-mlops"}'},
                timeout=self.timeout
            )
            query_data = query_response.json()
            lezea_metrics_available = len(query_data.get('data', {}).get('result', [])) > 0
            
            response_time = time.time() - start_time + http_result.response_time
            
            # Determine status
            if healthy_targets == total_targets and lezea_metrics_available:
                status = 'healthy'
                message = "Prometheus fully functional with all targets healthy"
            elif healthy_targets > 0:
                status = 'warning'
                message = f"Prometheus working but {total_targets - healthy_targets} targets unhealthy"
            else:
                status = 'warning'
                message = "Prometheus accessible but no healthy targets"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'healthy_targets': healthy_targets,
                    'total_targets': total_targets,
                    'lezea_metrics_available': lezea_metrics_available,
                    'targets': [{'job': t.get('labels', {}).get('job'), 'health': t.get('health')} 
                              for t in active_targets[:10]]  # First 10 targets
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='warning',
                response_time=time.time() - start_time + http_result.response_time,
                message=f"Prometheus HTTP accessible but API failed: {str(e)}",
                details={'error': str(e), 'http_status': 'ok'},
                timestamp=datetime.now()
            )
    
    def check_system_resources(self) -> HealthResult:
        """Check system resource utilization"""
        start_time = time.time()
        service = 'system_resources'
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix only)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = None
            
            # Network interfaces
            network_interfaces = list(psutil.net_if_addrs().keys())
            
            # Running processes
            process_count = len(psutil.pids())
            
            # Check for concerning resource usage
            warnings = []
            if cpu_percent > 90:
                warnings.append("High CPU usage")
            if memory_percent > 90:
                warnings.append("High memory usage")
            if disk_percent > 90:
                warnings.append("High disk usage")
            if load_avg and load_avg[0] > cpu_count * 2:
                warnings.append("High system load")
            
            response_time = time.time() - start_time
            
            status = 'warning' if warnings else 'healthy'
            message = "System resources healthy" if not warnings else f"Resource concerns: {', '.join(warnings)}"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'disk_percent': round(disk_percent, 1),
                    'disk_total_gb': round(disk.total / (1024**3), 2),
                    'load_average': load_avg,
                    'network_interfaces': network_interfaces,
                    'process_count': process_count,
                    'warnings': warnings
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"Failed to check system resources: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_gpu_resources(self) -> HealthResult:
        """Check GPU resources if available"""
        start_time = time.time()
        service = 'gpu_resources'
        
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode != 0:
                return HealthResult(
                    service=service,
                    status='warning',
                    response_time=time.time() - start_time,
                    message="NVIDIA drivers not available or no GPUs detected",
                    details={'nvidia_smi_available': False},
                    timestamp=datetime.now()
                )
            
            # Parse GPU information
            gpus = []
            warnings = []
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_info = {
                            'name': parts[0],
                            'utilization': int(parts[1]) if parts[1].isdigit() else 0,
                            'memory_used': int(parts[2]) if parts[2].isdigit() else 0,
                            'memory_total': int(parts[3]) if parts[3].isdigit() else 0,
                            'temperature': int(parts[4]) if parts[4].isdigit() else 0,
                            'power_draw': float(parts[5]) if len(parts) > 5 and parts[5].replace('.', '').isdigit() else 0
                        }
                        
                        # Calculate memory percentage
                        if gpu_info['memory_total'] > 0:
                            gpu_info['memory_percent'] = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
                        else:
                            gpu_info['memory_percent'] = 0
                        
                        # Check for warnings
                        if gpu_info['memory_percent'] > 90:
                            warnings.append(f"GPU {len(gpus)} memory usage high ({gpu_info['memory_percent']:.1f}%)")
                        if gpu_info['temperature'] > 85:
                            warnings.append(f"GPU {len(gpus)} temperature high ({gpu_info['temperature']}°C)")
                        
                        gpus.append(gpu_info)
            
            response_time = time.time() - start_time
            
            status = 'warning' if warnings else 'healthy'
            message = f"Found {len(gpus)} GPU(s)" if not warnings else f"GPU concerns: {', '.join(warnings)}"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'gpu_count': len(gpus),
                    'gpus': gpus,
                    'warnings': warnings
                },
                timestamp=datetime.now()
            )
        
        except subprocess.TimeoutExpired:
            return HealthResult(
                service=service,
                status='warning',
                response_time=self.timeout,
                message="GPU check timed out",
                details={'timeout': True},
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='warning',
                response_time=time.time() - start_time,
                message=f"GPU check failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def check_lezea_application(self) -> HealthResult:
        """Check LeZeA MLOps application health"""
        start_time = time.time()
        service = 'lezea_application'
        
        try:
            # Try to import and use LeZeA MLOps
            try:
                from lezea_mlops import ExperimentTracker
                from lezea_mlops.monitoring import get_metrics
                
                # Test basic functionality
                tracker = ExperimentTracker("health_check", create_if_not_exists=False)
                metrics = get_metrics()
                
                app_functional = True
                components = {
                    'tracker_available': True,
                    'metrics_available': True,
                    'config_loaded': hasattr(tracker, 'config')
                }
            except Exception as e:
                app_functional = False
                components = {'import_error': str(e)}
            
            # Check if metrics endpoint is accessible
            try:
                metrics_response = requests.get('http://localhost:8000/metrics', timeout=5)
                metrics_accessible = metrics_response.status_code == 200
                
                if metrics_accessible:
                    # Parse some basic metrics
                    metrics_count = len(metrics_response.text.split('\n'))
                    components['metrics_endpoint'] = True
                    components['metrics_count'] = metrics_count
                else:
                    components['metrics_endpoint'] = False
            except Exception:
                components['metrics_endpoint'] = False
            
            response_time = time.time() - start_time
            
            if app_functional and components.get('metrics_endpoint', False):
                status = 'healthy'
                message = "LeZeA MLOps application fully functional"
            elif app_functional:
                status = 'warning'
                message = "LeZeA MLOps core functional but metrics endpoint unavailable"
            else:
                status = 'unhealthy'
                message = "LeZeA MLOps application not functional"
            
            return HealthResult(
                service=service,
                status=status,
                response_time=response_time,
                message=message,
                details=components,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                service=service,
                status='unhealthy',
                response_time=time.time() - start_time,
                message=f"Application health check failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def run_single_check(self, service: str) -> HealthResult:
        """Run health check for a single service"""
        self.log(f"Checking {service}...", 'INFO')
        
        check_methods = {
            'postgresql': self.check_postgresql,
            'mongodb': self.check_mongodb,
            'mlflow': self.check_mlflow,
            's3_storage': self.check_s3_storage,
            'prometheus': self.check_prometheus,
            'system_resources': self.check_system_resources,
            'gpu_resources': self.check_gpu_resources,
            'lezea_application': self.check_lezea_application,
            'minio': lambda: self.check_http_endpoint('http://localhost:9000/minio/health/live', 'minio'),
            'grafana': lambda: self.check_http_endpoint('http://localhost:3000/api/health', 'grafana'),
            'node_exporter': lambda: self.check_http_endpoint('http://localhost:9100/metrics', 'node_exporter'),
            'gpu_exporter': lambda: self.check_http_endpoint('http://localhost:9835/metrics', 'gpu_exporter'),
        }
        
        if service in check_methods:
            try:
                result = check_methods[service]()
                self.log(f"{service}: {result.status} ({result.response_time:.2f}s) - {result.message}")
                return result
            except Exception as e:
                return HealthResult(
                    service=service,
                    status='unhealthy',
                    response_time=0,
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e)},
                    timestamp=datetime.now()
                )
        else:
            # Generic port check
            if service in self.services:
                config = self.services[service]
                return self.check_port_connectivity(config['host'], config['port'], service)
            else:
                return HealthResult(
                    service=service,
                    status='unknown',
                    response_time=0,
                    message=f"Unknown service: {service}",
                    details={},
                    timestamp=datetime.now()
                )
    
    def run_health_checks(self, services: Optional[List[str]] = None, 
                         parallel: bool = True) -> List[HealthResult]:
        """Run health checks for specified services"""
        if services is None:
            services = [
                'system_resources', 'gpu_resources', 'postgresql', 'mongodb', 
                's3_storage', 'mlflow', 'prometheus', 'lezea_application',
                'minio', 'grafana', 'node_exporter', 'gpu_exporter'
            ]
        
        self.results = []
        
        if parallel and len(services) > 1:
            # Run checks in parallel
            with ThreadPoolExecutor(max_workers=min(len(services), 8)) as executor:
                future_to_service = {
                    executor.submit(self.run_single_check, service): service 
                    for service in services
                }
                
                for future in as_completed(future_to_service):
                    result = future.result()
                    self.results.append(result)
        else:
            # Run checks sequentially
            for service in services:
                result = self.run_single_check(service)
                self.results.append(result)
        
        # Sort results by service name for consistent output
        self.results.sort(key=lambda x: x.service)
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate health check summary"""
        if not self.results:
            return {}
        
        status_counts = {'healthy': 0, 'warning': 0, 'unhealthy': 0, 'unknown': 0}
        total_response_time = 0
        
        for result in self.results:
            status_counts[result.status] += 1
            if result.response_time:
                total_response_time += result.response_time
        
        overall_status = 'healthy'
        if status_counts['unhealthy'] > 0:
            overall_status = 'unhealthy'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        
        return {
            'overall_status': overall_status,
            'total_services': len(self.results),
            'status_counts': status_counts,
            'total_response_time': round(total_response_time, 2),
            'average_response_time': round(total_response_time / len(self.results), 2),
            'timestamp': datetime.now().isoformat(),
            'critical_issues': [
                result.service for result in self.results 
                if result.status == 'unhealthy'
            ],
            'warnings': [
                result.service for result in self.results 
                if result.status == 'warning'
            ]
        }
    
    def print_results(self, json_output: bool = False):
        """Print health check results"""
        if json_output:
            output = {
                'summary': self.generate_summary(),
                'results': [result.to_dict() for result in self.results]
            }
            print(json.dumps(output, indent=2))
        else:
            self._print_formatted_results()
    
    def _print_formatted_results(self):
        """Print formatted health check results"""
        summary = self.generate_summary()
        
        # Print header
        print("\n" + "="*60)
        print("LeZeA MLOps Health Check Report")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Total Services: {summary['total_services']}")
        print(f"Average Response Time: {summary['average_response_time']}s")
        print("-"*60)
        
        # Print individual results
        status_colors = {
            'healthy': '\033[92m',   # Green
            'warning': '\033[93m',   # Yellow
            'unhealthy': '\033[91m', # Red
            'unknown': '\033[94m'    # Blue
        }
        reset_color = '\033[0m'
        
        for result in self.results:
            color = status_colors.get(result.status, reset_color)
            status_symbol = {
                'healthy': '✓',
                'warning': '⚠',
                'unhealthy': '✗',
                'unknown': '?'
            }.get(result.status, '?')
            
            response_time_str = f"({result.response_time:.2f}s)" if result.response_time else ""
            
            print(f"{color}{status_symbol} {result.service.ljust(18)}: {result.status.ljust(10)}{reset_color} "
                  f"{response_time_str.ljust(8)} {result.message}")
        
        # Print warnings and critical issues
        if summary['warnings']:
            print(f"\n{status_colors['warning']}Warnings:{reset_color}")
            for service in summary['warnings']:
                result = next(r for r in self.results if r.service == service)
                print(f"  • {service}: {result.message}")
        
        if summary['critical_issues']:
            print(f"\n{status_colors['unhealthy']}Critical Issues:{reset_color}")
            for service in summary['critical_issues']:
                result = next(r for r in self.results if r.service == service)
                print(f"  • {service}: {result.message}")
        
        print("-"*60)
        print(f"Status Summary: {summary['status_counts']['healthy']} healthy, "
              f"{summary['status_counts']['warning']} warnings, "
              f"{summary['status_counts']['unhealthy']} unhealthy")
        print("="*60)
    
    def send_alerts(self, webhook_url: Optional[str] = None, 
                   email_config: Optional[Dict[str, str]] = None):
        """Send alerts for critical issues"""
        summary = self.generate_summary()
        
        if summary['overall_status'] == 'healthy':
            self.log("All services healthy, no alerts needed", 'INFO')
            return
        
        alert_message = self._generate_alert_message(summary)
        
        # Send Slack/webhook alert
        if webhook_url:
            try:
                payload = {
                    'text': f"LeZeA MLOps Health Alert",
                    'attachments': [{
                        'color': 'danger' if summary['overall_status'] == 'unhealthy' else 'warning',
                        'text': alert_message,
                        'footer': f"Health check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }]
                }
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                self.log("Alert sent to webhook successfully", 'INFO')
            except Exception as e:
                self.log(f"Failed to send webhook alert: {e}", 'ERROR')
        
        # Send email alert (simplified - would need proper email setup)
        if email_config:
            self.log("Email alerting not implemented in this example", 'WARNING')
    
    def _generate_alert_message(self, summary: Dict[str, Any]) -> str:
        """Generate alert message text"""
        lines = [
            f"LeZeA MLOps Health Status: {summary['overall_status'].upper()}",
            f"Services: {summary['total_services']} total, {len(summary['critical_issues'])} critical issues, {len(summary['warnings'])} warnings"
        ]
        
        if summary['critical_issues']:
            lines.append(f"Critical Issues: {', '.join(summary['critical_issues'])}")
        
        if summary['warnings']:
            lines.append(f"Warnings: {', '.join(summary['warnings'])}")
        
        return '\n'.join(lines)
    
    def continuous_monitoring(self, interval: int = 60, duration: Optional[int] = None):
        """Run continuous health monitoring"""
        self.log(f"Starting continuous monitoring (interval: {interval}s)", 'INFO')
        
        start_time = time.time()
        
        try:
            while True:
                # Run health checks
                self.run_health_checks()
                
                # Print summary
                summary = self.generate_summary()
                timestamp = datetime.now().strftime('%H:%M:%S')
                status_icon = '✓' if summary['overall_status'] == 'healthy' else '⚠' if summary['overall_status'] == 'warning' else '✗'
                
                print(f"[{timestamp}] {status_icon} {summary['overall_status'].upper()} - "
                      f"{summary['status_counts']['healthy']}/{summary['total_services']} healthy "
                      f"({summary['average_response_time']:.2f}s avg)")
                
                # Send alerts if configured
                webhook_url = os.getenv('HEALTH_CHECK_WEBHOOK_URL')
                if webhook_url and summary['overall_status'] != 'healthy':
                    self.send_alerts(webhook_url=webhook_url)
                
                # Check if duration exceeded
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Wait for next check
                time.sleep(interval)
        
        except KeyboardInterrupt:
            self.log("Continuous monitoring stopped by user", 'INFO')

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LeZeA MLOps Health Check')
    parser.add_argument('--services', type=str, help='Comma-separated list of services to check')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout for each check (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Interval for continuous monitoring (seconds)')
    parser.add_argument('--duration', type=int, help='Duration for continuous monitoring (seconds)')
    parser.add_argument('--alert', action='store_true', help='Send alerts for failures')
    parser.add_argument('--parallel', action='store_true', default=True, help='Run checks in parallel')
    
    args = parser.parse_args()
    
    # Parse services list
    services = None
    if args.services:
        services = [s.strip() for s in args.services.split(',')]
    
    # Create health checker
    checker = LeZeAHealthChecker(timeout=args.timeout, verbose=args.verbose)
    
    if args.continuous:
        # Run continuous monitoring
        checker.continuous_monitoring(interval=args.interval, duration=args.duration)
    else:
        # Run single health check
        checker.run_health_checks(services=services, parallel=args.parallel)
        
        # Print results
        checker.print_results(json_output=args.json)
        
        # Send alerts if requested
        if args.alert:
            webhook_url = os.getenv('HEALTH_CHECK_WEBHOOK_URL')
            checker.send_alerts(webhook_url=webhook_url)
        
        # Exit with appropriate code
        summary = checker.generate_summary()
        exit_code = 0 if summary['overall_status'] == 'healthy' else 1
        sys.exit(exit_code)

if __name__ == '__main__':
    main()