#!/usr/bin/env python3
"""
LeZeA MLOps - Comprehensive Health Check Script
==============================================

Advanced health monitoring for all LeZeA MLOps components:
- Service availability and response times
- Database connectivity and performance
- Storage accessibility and capacity (prefers DVC remote)
- DVC remote validation (no '(default)' parsing)
- Resource utilization monitoring
- Integration testing

macOS niceties:
- gpu_exporter is skipped on Darwin (no NVIDIA/DCGM there)

Usage:
    python scripts/health_check.py [OPTIONS]

Options:
    --services      Comma-separated list of services to check
    --timeout       Timeout for each check (default: 10 seconds)
    --verbose       Enable detailed output
    --json          Output results in JSON format
    --continuous    Run continuous health monitoring
    --interval      Interval for continuous monitoring (default: 60 seconds)
    --alert         Send alerts for failures
    --no-parallel   Run checks sequentially
"""

import os
import sys
import json
import time
import socket
import psutil
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
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
except ImportError as e:
    print(f"Warning: Missing optional dependency: {e}")
    print("Some health checks may not be available")

# ----------------------------- Utils & Models --------------------------------

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
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


def _safe_bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _normalize_env():
    """Keep env keys consistent across scripts."""
    # prefer MLFLOW_ARTIFACT_URI over MLFLOW_ARTIFACT_ROOT
    if not os.getenv("MLFLOW_ARTIFACT_URI") and os.getenv("MLFLOW_ARTIFACT_ROOT"):
        os.environ["MLFLOW_ARTIFACT_URI"] = os.getenv("MLFLOW_ARTIFACT_ROOT")

    # prefer AWS_DEFAULT_REGION, fall back to AWS_REGION, default eu-central-1
    if not os.getenv("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION", "eu-central-1")

    # pass-through DVC hints (used by both bash + this checker)
    os.environ.setdefault("DVC_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-central-1"))
    os.environ.setdefault("DVC_REMOTE_URL", os.getenv("DVC_REMOTE_URL", ""))
    os.environ.setdefault("DVC_ENDPOINT_URL", os.getenv("DVC_ENDPOINT_URL", ""))


# ------------------------------ Health Checker --------------------------------

class LeZeAHealthChecker:
    """Comprehensive health checker for LeZeA MLOps system"""

    def __init__(self, timeout: int = 10, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[HealthResult] = []

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

        self._load_config()

    # --------------------------- config & logging -----------------------------

    def _load_config(self):
        """Load configuration from .env and normalize."""
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        k, v = line.strip().split('=', 1)
                        # don't clobber existing env if already set
                        os.environ.setdefault(k, v.strip('"\''))
        _normalize_env()

        # ports from env (optional)
        self.services['postgresql']['port'] = int(os.getenv('POSTGRES_PORT', 5432))
        self.services['mongodb']['port'] = int(os.getenv('MONGODB_PORT', 27017))
        self.services['mlflow']['port'] = int(os.getenv('MLFLOW_PORT', 5000))
        self.services['prometheus']['port'] = int(os.getenv('PROMETHEUS_PORT', 9090))

    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.verbose or level in ['ERROR', 'WARNING']:
            print(f"[{level}] {timestamp} - {message}")

    # --------------------------- primitive checks -----------------------------

    def check_port_connectivity(self, host: str, port: int, service: str) -> HealthResult:
        start = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            rc = sock.connect_ex((host, port))
            sock.close()
            rt = time.time() - start
            if rc == 0:
                return HealthResult(service, 'healthy', rt, f"Port {port} is accessible",
                                    {'host': host, 'port': port}, datetime.now())
            return HealthResult(service, 'unhealthy', rt, f"Port {port} is not accessible",
                                {'host': host, 'port': port, 'error_code': rc}, datetime.now())
        except Exception as e:
            return HealthResult(service, 'unhealthy', time.time() - start,
                                f"Connection failed: {e}", {'host': host, 'port': port}, datetime.now())

    def check_http_endpoint(self, url: str, service: str,
                            expected_status: int = 200,
                            check_content: Optional[str] = None) -> HealthResult:
        start = time.time()
        try:
            r = requests.get(url, timeout=self.timeout)
            rt = time.time() - start
            status = 'healthy'
            msg = f"HTTP {r.status_code}"
            if r.status_code != expected_status:
                status = 'warning'
                msg = f"Unexpected status {r.status_code}"
            if check_content and check_content not in r.text:
                status = 'warning'
                msg += f"; missing content '{check_content}'"
            return HealthResult(service, status, rt, msg,
                                {'url': url, 'status_code': r.status_code}, datetime.now())
        except requests.RequestException as e:
            return HealthResult(service, 'unhealthy', time.time() - start,
                                f"HTTP request failed: {e}", {'url': url}, datetime.now())

    # ------------------------------- services ---------------------------------

    def check_postgresql(self) -> HealthResult:
        start = time.time()
        svc = 'postgresql'

        # Skip if MLflow is using SQLite (common on dev laptops)
        backend = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")
        if backend.strip().lower().startswith("sqlite"):
            return HealthResult(
                svc, 'warning', 0.0, 'skipped — MLflow backend is SQLite',
                {'backend': backend}, datetime.now()
            )

        try:
            params = {
                'host': self.services[svc]['host'],
                'port': self.services[svc]['port'],
                'database': os.getenv('POSTGRES_DB', 'lezea_mlops'),
                'user': os.getenv('POSTGRES_USER', 'lezea_user'),
                'password': os.getenv('POSTGRES_PASSWORD', 'lezea_secure_password_2024'),
                'connect_timeout': self.timeout
            }
            with psycopg2.connect(**params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    cur.execute("""
                        SELECT COUNT(*) FROM information_schema.tables
                        WHERE table_schema = 'mlflow' AND table_name = 'experiments'
                    """)
                    tables_exist = cur.fetchone()[0] > 0
                    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
                    db_size = cur.fetchone()[0]
                    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                    active = cur.fetchone()[0]
            rt = time.time() - start
            status = 'healthy' if tables_exist else 'warning'
            msg = "Database OK" if tables_exist else "DB OK but MLflow tables missing"
            return HealthResult(
                svc, status, rt, msg,
                {
                    'version': version,
                    'database_size': db_size,
                    'active_connections': active,
                    'host': params['host'], 'port': params['port'],
                    'database': params['database'], 'user': params['user']
                }, datetime.now()
            )
        except Exception as e:
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"Database connection failed: {e}", {}, datetime.now())

    def check_mongodb(self) -> HealthResult:
        start = time.time()
        svc = 'mongodb'
        try:
            # Prefer full connection string if provided (Atlas etc.)
            conn_str = os.getenv('MONGO_CONNECTION_STRING') or os.getenv('MONGODB_URI')
            db_name = os.getenv('MONGO_DATABASE', 'lezea_mlops')

            if conn_str:
                client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=self.timeout * 1000)
            else:
                client = pymongo.MongoClient(
                    host=self.services[svc]['host'],
                    port=self.services[svc]['port'],
                    serverSelectionTimeoutMS=self.timeout * 1000,
                    connectTimeoutMS=self.timeout * 1000
                )

            client.admin.command('ping')
            server_info = client.server_info()
            db = client[db_name]
            db_stats = db.command('dbStats')
            collections = db.list_collection_names()

            rt = time.time() - start
            return HealthResult(
                svc, 'healthy', rt, "MongoDB is accessible and responsive",
                {
                    'version': server_info.get('version'),
                    'database': db_name,
                    'database_size': db_stats.get('dataSize', 0),
                    'collections_count': len(collections),
                    'collections_sample': collections[:10]
                }, datetime.now()
            )
        except Exception as e:
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"MongoDB connection failed: {e}", {}, datetime.now())

    def check_mlflow(self) -> HealthResult:
        svc = 'mlflow'
        host = self.services[svc]['host']
        port = self.services[svc]['port']
        tracking_uri = f"http://{host}:{port}"

        start = time.time()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            exps = mlflow.search_experiments()  # primary health signal
            rt = time.time() - start
            # optional: peek at UI root, but don't gate health on it
            try:
                r = requests.get(f"{tracking_uri}/", timeout=self.timeout)
                http_note = f"UI {r.status_code}"
            except Exception as e:
                http_note = f"UI check failed: {e}"

            return HealthResult(
                svc, 'healthy', rt,
                f"MLflow API OK ({len(exps)} experiments); {http_note}",
                {'experiments_count': len(exps), 'tracking_uri': tracking_uri}, datetime.now()
            )
        except Exception as api_err:
            # fall back to a simple HTTP reachability probe
            http = self.check_http_endpoint(f"{tracking_uri}/", svc)
            if http.status == 'healthy':
                return HealthResult(
                    svc, 'warning', (http.response_time or 0.0),
                    f"MLflow UI reachable but API failed: {api_err}",
                    {'tracking_uri': tracking_uri}, datetime.now()
                )
            return http


    # ------------------------------- DVC & S3 ---------------------------------

    def _dvc_run(self, args: List[str]) -> str:
        return subprocess.check_output(args, text=True, timeout=self.timeout).strip()

    def _dvc_available(self) -> bool:
        try:
            subprocess.check_output(["dvc", "--version"], text=True, timeout=self.timeout)
            return True
        except Exception:
            return False

    def _dvc_get_default_remote_name(self) -> Optional[str]:
        try:
            return self._dvc_run(["dvc", "config", "core.remote"]) or None
        except Exception:
            return None

    def _dvc_get_remote_url(self, remote_name: Optional[str]) -> Optional[str]:
        if not remote_name:
            return None
        try:
            return self._dvc_run(["dvc", "config", f"remote.{remote_name}.url"]) or None
        except Exception:
            return None

    def check_dvc(self) -> HealthResult:
        """Validate DVC default remote without parsing `dvc remote list`."""
        start = time.time()
        svc = 'dvc'
        try:
            if not self._dvc_available():
                return HealthResult(svc, 'warning', time.time() - start,
                                    "DVC not installed", {}, datetime.now())

            name = self._dvc_get_default_remote_name()
            if not name:
                return HealthResult(svc, 'warning', time.time() - start,
                                    "No default DVC remote configured (core.remote empty)", {}, datetime.now())
            url = self._dvc_get_remote_url(name)
            if not url:
                return HealthResult(svc, 'unhealthy', time.time() - start,
                                    f'DVC remote "{name}" missing URL', {}, datetime.now())

            # If it's s3://, head the bucket using boto3 with region/endpoint from env
            if url.startswith("s3://"):
                bucket = url[5:].split("/", 1)[0]
                region = os.getenv("DVC_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
                endpoint = os.getenv("DVC_ENDPOINT_URL") or None
                session = boto3.session.Session(region_name=region)
                s3 = session.client("s3", endpoint_url=endpoint) if endpoint else session.client("s3")
                s3.head_bucket(Bucket=bucket)
                rt = time.time() - start
                return HealthResult(svc, 'healthy', rt, "DVC remote OK (S3 head passed)",
                                    {'remote': url, 'region': region, 'endpoint': endpoint}, datetime.now())

            # Other schemes: basic OK
            rt = time.time() - start
            return HealthResult(svc, 'healthy', rt, "DVC remote OK",
                                {'remote': url}, datetime.now())

        except Exception as e:
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"DVC remote check failed: {e}", {}, datetime.now())

    def check_s3_storage(self) -> HealthResult:
        """Prefer DVC remote; else MLflow artifact root; validate access and simple IO."""
        start = time.time()
        svc = 's3_storage'
        try:
            # choose target URL
            target_url = (os.getenv("DVC_REMOTE_URL") or "").strip()
            if not target_url.startswith("s3://"):
                target_url = (os.getenv("MLFLOW_ARTIFACT_URI") or "").strip()

            if not target_url.startswith("s3://"):
                return HealthResult(svc, 'warning', 0.0, "No S3 URL configured (DVC_REMOTE_URL or MLFLOW_ARTIFACT_URI)",
                                    {}, datetime.now())

            # parse
            after = target_url[5:]
            bucket = after.split("/", 1)[0]
            prefix = after.split("/", 1)[1] if "/" in after else ""

            region = os.getenv("DVC_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
            endpoint = os.getenv("DVC_ENDPOINT_URL") or os.getenv("MLFLOW_S3_ENDPOINT_URL") or None

            session = boto3.session.Session(region_name=region)
            s3 = session.client("s3", endpoint_url=endpoint) if endpoint else session.client("s3")

            # head + light list
            s3.head_bucket(Bucket=bucket)
            _ = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)

            # optional small IO test (may fail due to policy; treat as warning if so)
            key = (prefix + "/" if prefix else "") + f"health_check_{int(time.time())}.txt"
            payload = b"LeZeA MLOps health check"

            upload_ok = True
            try:
                s3.put_object(Bucket=bucket, Key=key, Body=payload)
                obj = s3.get_object(Bucket=bucket, Key=key)
                data = obj["Body"].read()
                s3.delete_object(Bucket=bucket, Key=key)
                upload_ok = data == payload
            except Exception as e:
                upload_ok = False
                self.log(f"S3 upload/download test failed (policy?): {e}", 'WARNING')

            rt = time.time() - start
            status = 'healthy' if upload_ok else 'warning'
            msg = "S3 storage fully functional" if upload_ok else "S3 accessible (IO test failed; check bucket policy)"
            return HealthResult(svc, status, rt, msg,
                                {'bucket': bucket, 'region': region, 'endpoint': endpoint, 'prefix': prefix,
                                 'io_test': upload_ok}, datetime.now())

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "ClientError")
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"S3 access failed: {code}", {'error': str(e)}, datetime.now())
        except Exception as e:
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"S3 connection failed: {e}", {}, datetime.now())

    # ---------------------------- observability -------------------------------

    def check_prometheus(self) -> HealthResult:
        svc = 'prometheus'
        host = self.services[svc]['host']; port = self.services[svc]['port']

        http = self.check_http_endpoint(f"http://{host}:{port}/-/healthy", svc)
        if http.status != 'healthy':
            return http

        start = time.time()
        try:
            targets = requests.get(f"http://{host}:{port}/api/v1/targets", timeout=self.timeout).json()
            active = targets.get('data', {}).get('activeTargets', [])
            healthy = sum(1 for t in active if t.get('health') == 'up')
            total = len(active)

            # sample query for a LeZeA job label if present
            try:
                q = requests.get(f"http://{host}:{port}/api/v1/query",
                                 params={'query': 'up{job="lezea-mlops"}'}, timeout=self.timeout).json()
                lezea_ok = len(q.get('data', {}).get('result', [])) > 0
            except Exception:
                lezea_ok = False

            rt = time.time() - start + (http.response_time or 0)
            if healthy == total and total > 0:
                status = 'healthy'
                msg = "Prometheus OK; all targets up"
            elif healthy > 0:
                status = 'warning'
                msg = f"Prometheus running; {total - healthy}/{total} targets down"
            else:
                status = 'warning'
                msg = "Prometheus reachable; no healthy targets"

            return HealthResult(svc, status, rt, msg,
                                {'healthy_targets': healthy, 'total_targets': total,
                                 'lezea_metrics_available': lezea_ok}, datetime.now())
        except Exception as e:
            return HealthResult(svc, 'warning', time.time() - start + (http.response_time or 0),
                                f"Prometheus API failed: {e}", {}, datetime.now())

    # ----------------------------- system checks ------------------------------

    def check_system_resources(self) -> HealthResult:
        start = time.time(); svc = 'system_resources'
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = None

            warnings = []
            if cpu_percent > 90: warnings.append("High CPU")
            if mem.percent > 90: warnings.append("High memory")
            if (disk.used / disk.total) * 100 > 90: warnings.append("High disk")
            if load_avg and load_avg[0] > (psutil.cpu_count() or 1) * 2:
                warnings.append("High load")

            rt = time.time() - start
            status = 'warning' if warnings else 'healthy'
            msg = "System resources healthy" if not warnings else ", ".join(warnings)

            return HealthResult(
                svc, status, rt, msg,
                {
                    'cpu_percent': cpu_percent,
                    'memory_percent': mem.percent,
                    'memory_total_gb': round(mem.total / (1024**3), 2),
                    'disk_percent': round((disk.used / disk.total) * 100, 1),
                    'disk_total_gb': round(disk.total / (1024**3), 2),
                    'load_average': load_avg,
                    'process_count': len(psutil.pids()),
                }, datetime.now()
            )
        except Exception as e:
            return HealthResult(svc, 'unhealthy', time.time() - start,
                                f"System check failed: {e}", {}, datetime.now())

    def check_gpu_resources(self) -> HealthResult:
        start = time.time(); svc = 'gpu_resources'
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=self.timeout
            )
            if r.returncode != 0:
                return HealthResult(svc, 'warning', time.time() - start,
                                    "NVIDIA drivers not available / no GPUs", {'nvidia_smi': False}, datetime.now())

            gpus = []; warnings = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    name, util, mu, mt, temp = parts[:5]
                    power = float(parts[5]) if len(parts) > 5 and parts[5].replace('.', '', 1).isdigit() else 0.0
                    util = int(util) if util.isdigit() else 0
                    mu = int(mu) if mu.isdigit() else 0
                    mt = int(mt) if mt.isdigit() else 0
                    temp = int(temp) if temp.isdigit() else 0
                    mp = (mu / mt * 100) if mt else 0
                    if mp > 90: warnings.append(f"GPU mem >90% ({mp:.1f}%)")
                    if temp > 85: warnings.append(f"GPU temp >85C ({temp}°C)")
                    gpus.append({'name': name, 'util': util, 'mem_used': mu, 'mem_total': mt,
                                 'mem_percent': round(mp, 1), 'temp': temp, 'power': power})
            rt = time.time() - start
            status = 'warning' if warnings else 'healthy'
            msg = f"{len(gpus)} GPU(s) detected" if not warnings else "; ".join(warnings)
            return HealthResult(svc, status, rt, msg, {'gpus': gpus}, datetime.now())
        except subprocess.TimeoutExpired:
            return HealthResult(svc, 'warning', self.timeout, "GPU check timed out", {}, datetime.now())
        except Exception as e:
            return HealthResult(svc, 'warning', time.time() - start, f"GPU check failed: {e}", {}, datetime.now())

    def check_lezea_application(self) -> HealthResult:
        """
        LeZeA app check (lightweight):
        - Green if the package imports.
        - Optional symbols (ExperimentTracker, metrics) are best-effort and DO NOT gate health.
        """
        start = time.time(); svc = 'lezea_application'
        try:
            import lezea_mlops  # noqa: F401
            details = {
                'package_path': getattr(lezea_mlops, '__file__', '(unknown)'),
            }

            # Optional: try (but do not require) a couple of symbols
            try:
                from lezea_mlops import ExperimentTracker  # type: ignore
                details['tracker_available'] = True
            except Exception as e:
                details['tracker_available'] = False
                details['tracker_error'] = str(e)[:200]

            # Optional: metrics endpoint probe (doesn't gate health)
            metrics_ok = False
            try:
                r = requests.get('http://localhost:8000/metrics', timeout=3)
                metrics_ok = (r.status_code == 200)
            except Exception:
                pass
            details['metrics_endpoint'] = metrics_ok

            return HealthResult(
                svc, 'healthy', time.time() - start, "LeZeA package import OK",
                details, datetime.now()
            )
        except Exception as e:
            return HealthResult(
                svc, 'unhealthy', time.time() - start,
                f"Package import failed: {e}", {'error': str(e)}, datetime.now()
            )

    # ------------------------------- runner -----------------------------------

    def run_single_check(self, service: str) -> HealthResult:
        self.log(f"Checking {service}...", 'INFO')

        # Skip GPU exporter on macOS hosts (no NVIDIA/DCGM)
        if service == 'gpu_exporter' and platform.system() == 'Darwin':
            return HealthResult(
                service='gpu_exporter',
                status='warning',
                response_time=0.0,
                message='skipped — macOS has no NVIDIA/DCGM exporter',
                details={'platform': platform.system()},
                timestamp=datetime.now()
            )

        checks = {
            'postgresql': self.check_postgresql,
            'mongodb': self.check_mongodb,
            'mlflow': self.check_mlflow,
            'dvc': self.check_dvc,
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
        if service in checks:
            try:
                result = checks[service]()
                self.log(f"{service}: {result.status} ({result.response_time:.2f}s) - {result.message}")
                return result
            except Exception as e:
                return HealthResult(service, 'unhealthy', 0.0,
                                    f"Health check failed: {e}", {}, datetime.now())
        # fallback: raw port check if known service
        if service in self.services:
            cfg = self.services[service]
            return self.check_port_connectivity(cfg['host'], cfg['port'], service)
        return HealthResult(service, 'unknown', 0.0, f"Unknown service: {service}", {}, datetime.now())

    def run_health_checks(self, services: Optional[List[str]] = None, parallel: bool = True) -> List[HealthResult]:
        if services is None:
            services = [
                'system_resources', 'gpu_resources',
                'postgresql', 'mongodb',
                'dvc', 's3_storage',
                'mlflow', 'prometheus',
                'lezea_application',
                'minio', 'grafana', 'node_exporter', 'gpu_exporter'
            ]
        self.results = []
        if parallel and len(services) > 1:
            with ThreadPoolExecutor(max_workers=min(len(services), 8)) as ex:
                futures = {ex.submit(self.run_single_check, s): s for s in services}
                for fut in as_completed(futures):
                    self.results.append(fut.result())
        else:
            for s in services:
                self.results.append(self.run_single_check(s))
        self.results.sort(key=lambda x: x.service)
        return self.results

    # ------------------------------- printing ---------------------------------

    def generate_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        counts = {'healthy': 0, 'warning': 0, 'unhealthy': 0, 'unknown': 0}
        total_rt = 0.0
        for r in self.results:
            counts[r.status] += 1
            if r.response_time:
                total_rt += r.response_time
        overall = 'healthy'
        if counts['unhealthy'] > 0:
            overall = 'unhealthy'
        elif counts['warning'] > 0:
            overall = 'warning'
        return {
            'overall_status': overall,
            'total_services': len(self.results),
            'status_counts': counts,
            'total_response_time': round(total_rt, 2),
            'average_response_time': round(total_rt / max(len(self.results), 1), 2),
            'timestamp': datetime.now().isoformat(),
            'critical_issues': [r.service for r in self.results if r.status == 'unhealthy'],
            'warnings': [r.service for r in self.results if r.status == 'warning'],
        }

    def print_results(self, json_output: bool = False):
        if json_output:
            print(json.dumps({'summary': self.generate_summary(),
                              'results': [r.to_dict() for r in self.results]}, indent=2))
            return
        self._print_formatted_results()

    def _print_formatted_results(self):
        summary = self.generate_summary()
        print("\n" + "="*60)
        print("LeZeA MLOps Health Check Report")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {summary.get('overall_status','N/A').upper()}")
        print(f"Total Services: {summary.get('total_services', 0)}")
        print(f"Average Response Time: {summary.get('average_response_time', 0.0)}s")
        print("-"*60)

        status_colors = {
            'healthy': '\033[92m', 'warning': '\033[93m',
            'unhealthy': '\033[91m', 'unknown': '\033[94m'
        }
        reset = '\033[0m'
        symbols = {'healthy': '✓', 'warning': '⚠', 'unhealthy': '✗', 'unknown': '?'}

        for r in self.results:
            color = status_colors.get(r.status, reset)
            sym = symbols.get(r.status, '?')
            rt = f"({r.response_time:.2f}s)" if r.response_time is not None else ""
            print(f"{color}{sym} {r.service.ljust(18)}: {r.status.ljust(10)}{reset} {rt.ljust(8)} {r.message}")

        if summary.get('warnings'):
            print(f"\n{status_colors['warning']}Warnings:{reset}")
            for s in summary['warnings']:
                rr = next(x for x in self.results if x.service == s)
                print(f"  • {s}: {rr.message}")

        if summary.get('critical_issues'):
            print(f"\n{status_colors['unhealthy']}Critical Issues:{reset}")
            for s in summary['critical_issues']:
                rr = next(x for x in self.results if x.service == s)
                print(f"  • {s}: {rr.message}")

        c = summary.get('status_counts', {})
        print("-"*60)
        print(f"Status Summary: {c.get('healthy',0)} healthy, {c.get('warning',0)} warnings, {c.get('unhealthy',0)} unhealthy")
        print("="*60)

    # ------------------------------- alerts -----------------------------------

    def send_alerts(self, webhook_url: Optional[str] = None):
        summary = self.generate_summary()
        if summary.get('overall_status') == 'healthy':
            self.log("All services healthy, no alerts needed", 'INFO')
            return
        if not webhook_url:
            self.log("No webhook configured; skipping alerts", 'WARNING')
            return
        try:
            payload = {
                'text': "LeZeA MLOps Health Alert",
                'attachments': [{
                    'color': 'danger' if summary['overall_status'] == 'unhealthy' else 'warning',
                    'text': self._generate_alert_message(summary),
                    'footer': f"Health check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }]
            }
            r = requests.post(webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            self.log("Alert sent to webhook successfully", 'INFO')
        except Exception as e:
            self.log(f"Failed to send webhook alert: {e}", 'ERROR')

    def _generate_alert_message(self, summary: Dict[str, Any]) -> str:
        lines = [
            f"LeZeA MLOps Health Status: {summary['overall_status'].upper()}",
            f"Services: {summary['total_services']} total, "
            f"{len(summary['critical_issues'])} critical, {len(summary['warnings'])} warnings"
        ]
        if summary['critical_issues']:
            lines.append("Critical: " + ", ".join(summary['critical_issues']))
        if summary['warnings']:
            lines.append("Warnings: " + ", ".join(summary['warnings']))
        return "\n".join(lines)

    # ----------------------------- monitoring loop ----------------------------

    def continuous_monitoring(self, interval: int = 60, duration: Optional[int] = None):
        self.log(f"Starting continuous monitoring (interval: {interval}s)", 'INFO')
        start_t = time.time()
        try:
            while True:
                self.run_health_checks()
                summ = self.generate_summary()
                t = datetime.now().strftime('%H:%M:%S')
                icon = '✓' if summ['overall_status'] == 'healthy' else ('⚠' if summ['overall_status'] == 'warning' else '✗')
                print(f"[{t}] {icon} {summ['overall_status'].upper()} - "
                      f"{summ['status_counts']['healthy']}/{summ['total_services']} healthy "
                      f"({summ['average_response_time']:.2f}s avg)")
                webhook = os.getenv('HEALTH_CHECK_WEBHOOK_URL')
                if webhook and summ['overall_status'] != 'healthy':
                    self.send_alerts(webhook_url=webhook)
                if duration and (time.time() - start_t) >= duration:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            self.log("Continuous monitoring stopped by user", 'INFO')


# ---------------------------------- CLI --------------------------------------

def main():
    parser = argparse.ArgumentParser(description='LeZeA MLOps Health Check')
    parser.add_argument('--services', type=str, help='Comma-separated list of services to check')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout for each check (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Interval for continuous monitoring (seconds)')
    parser.add_argument('--duration', type=int, help='Duration for continuous monitoring (seconds)')
    parser.add_argument('--alert', action='store_true', help='Send alerts for failures')
    parser.add_argument('--no-parallel', action='store_true', help='Run checks sequentially')

    args = parser.parse_args()

    # Services selection
    services = None
    if args.services:
        services = [s.strip() for s in args.services.split(',') if s.strip()]

    checker = LeZeAHealthChecker(timeout=args.timeout, verbose=args.verbose)

    if args.continuous:
        checker.continuous_monitoring(interval=args.interval, duration=args.duration)
        return

    checker.run_health_checks(services=services, parallel=not args.no_parallel)
    checker.print_results(json_output=args.json)

    if args.alert:
        checker.send_alerts(webhook_url=os.getenv('HEALTH_CHECK_WEBHOOK_URL'))

    summary = checker.generate_summary()
    sys.exit(0 if summary.get('overall_status') == 'healthy' else 1)


if __name__ == '__main__':
    main()
