"""
Environment Tagger for LeZeA MLOps — FINAL
==========================================

- Hardware specs (CPU, GPU, RAM, disk, net)
- Software env (OS, Python, key pkgs)
- Git repo info + last commit/date
- Container (Docker/K8s) + cloud detection (AWS/GCP/Azure)
- Optional dataset fingerprinting (directory hash/size/count)
- MLflow-safe tag generation
- Export + reproducibility hash

Public methods used by your tracker:
- get_environment_info()
- get_mlflow_tags(env_info=None)
- get_git_info()   <-- new public wrapper
- export_environment(filepath, include_sensitive=False)
- get_reproducibility_hash()
"""

from __future__ import annotations

import os
import sys
import json
import socket
import platform
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional libs
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except Exception:
    pkg_resources = None
    PKG_RESOURCES_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except Exception:
    GPUtil = None
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# urllib (for cloud metadata)
try:
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    _URL_OK = True
except Exception:
    _URL_OK = False
    _urlreq = None
    _urlerr = None


class EnvironmentTagger:
    def __init__(self) -> None:
        self.cache: Dict[str, Any] = {}
        self.git_available = self._check_git_available()
        self.docker_available = self._check_docker_available()
        print("🔍 Environment tagger initialized")

    # ------------------------
    # Small utilities
    # ------------------------
    def _run(self, cmd: List[str], *, timeout: float = 3.0) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=timeout
            )
        except Exception:
            return None

    def _check_git_available(self) -> bool:
        r = self._run(["git", "--version"])
        return bool(r and r.returncode == 0)

    def _check_docker_available(self) -> bool:
        try:
            if Path("/.dockerenv").exists() or os.getenv("DOCKER_CONTAINER"):
                return True
            cg = Path("/proc/1/cgroup")
            if cg.exists():
                txt = cg.read_text(errors="ignore")
                return "docker" in txt or "containerd" in txt
        except Exception:
            pass
        return False

    # ------------------------
    # Public surface
    # ------------------------
    def get_environment_info(self) -> Dict[str, Any]:
        env = {
            "collection_timestamp": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "hardware": self._get_hardware_info(),
            "software": self._get_software_info(),
            "python": self._get_python_info(),
            "packages": self._get_package_info(),
            "git": self._get_git_info(),
            "environment": self._get_environment_variables(),
            "platform": self._get_platform_info(),
            "cloud": self._detect_cloud_platform(),
            "container": self._get_container_info(),
        }
        # Optional: light dataset fingerprint (safe & fast)
        try:
            ds_root = os.getenv("LEZEA_DATASET_ROOT", "data")
            if Path(ds_root).exists():
                env["dataset_fingerprint"] = self.get_dataset_fingerprint(ds_root)
        except Exception:
            pass
        return env

    def get_git_info(self) -> Dict[str, Any]:
        """Public wrapper (your tracker calls hasattr(...,'get_git_info'))."""
        return self._get_git_info()

    def get_mlflow_tags(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        if env_info is None:
            env_info = self.get_environment_info()

        try:
            tags: Dict[str, str] = {}

            # System
            system = env_info.get("system", {})
            tags["system.hostname"] = system.get("hostname", "unknown")
            tags["system.platform"] = system.get("system", "unknown")
            tags["system.architecture"] = system.get("architecture", "unknown")

            # Hardware
            hw = env_info.get("hardware", {})
            cpu = hw.get("cpu", {})
            mem = hw.get("memory", {})
            gpu = hw.get("gpu", {})

            if cpu.get("physical_cores") is not None:
                tags["hardware.cpu_cores"] = str(cpu["physical_cores"])
            if cpu.get("logical_cores") is not None:
                tags["hardware.cpu_threads"] = str(cpu["logical_cores"])
            if mem.get("total_gb") is not None:
                tags["hardware.memory_gb"] = str(int(mem["total_gb"]))
            tags["hardware.gpu_available"] = str(bool(gpu.get("available", False)))
            if gpu.get("count"):
                tags["hardware.gpu_count"] = str(gpu["count"])
            if gpu.get("devices"):
                names = [d.get("name", "gpu") for d in gpu["devices"][:3]]
                tags["hardware.gpu_types"] = ",".join(names)
            if gpu.get("cuda_available"):
                tags["hardware.cuda_available"] = str(gpu.get("cuda_available"))
            if gpu.get("cuda_version"):
                tags["hardware.cuda_version"] = str(gpu.get("cuda_version"))
            if gpu.get("driver_version"):
                tags["hardware.gpu_driver"] = str(gpu.get("driver_version"))

            # Python
            py = env_info.get("python", {})
            vi = py.get("version_info", {})
            if vi:
                tags["python.version"] = f"{vi.get('major',0)}.{vi.get('minor',0)}.{vi.get('micro',0)}"
            tags["python.implementation"] = py.get("implementation", "unknown")
            tags["python.virtual_env"] = str(py.get("virtual_env", False))
            if py.get("conda_env"):
                tags["python.conda_env"] = py["conda_env"]

            # Packages (key ones only)
            pk = env_info.get("packages", {}).get("key_packages", {})
            for k in ("torch", "tensorflow", "numpy", "pandas", "scikit-learn", "mlflow", "dvc"):
                if k in pk:
                    tags[f"package.{k}"] = pk[k]

            # Git
            git = env_info.get("git", {})
            tags["git.available"] = "true" if git.get("is_repo") else "false"
            if git.get("commit"):
                tags["git.commit"] = git["commit"][:8]
            if git.get("branch"):
                tags["git.branch"] = git["branch"]
            if git.get("status"):
                tags["git.status"] = git["status"]
            if git.get("tag"):
                tags["git.tag"] = git["tag"]

            # Cloud + container
            cloud = env_info.get("cloud", {})
            tags["cloud.platform"] = cloud.get("platform", "local") if cloud.get("detected") else "local"
            if cloud.get("instance_type"):
                tags["cloud.instance_type"] = cloud["instance_type"]
            if cloud.get("region"):
                tags["cloud.region"] = cloud["region"]

            container = env_info.get("container", {})
            tags["container.docker"] = str(container.get("docker", False))
            tags["container.kubernetes"] = str(container.get("kubernetes", False))

            # OS distro or product
            plat = env_info.get("platform", {}).get("platform_details", {})
            if "distro_name" in plat:
                tags["os.distribution"] = plat["distro_name"]
            elif "product_name" in plat:
                tags["os.product"] = plat["product_name"]

            # Env vars (safe subset)
            env_vars = env_info.get("environment", {}).get("tracked_variables", {})
            for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MLFLOW_TRACKING_URI"):
                if env_vars.get(k):
                    tags[f"env.{k.lower()}"] = env_vars[k]

            # Dataset fingerprint (if present)
            dsf = env_info.get("dataset_fingerprint", {})
            if dsf.get("hash_short"):
                tags["dataset.version"] = dsf["hash_short"]
            if dsf.get("file_count") is not None:
                tags["dataset.files"] = str(dsf["file_count"])

            # Enforce MLflow tag value length (250-ish is safe)
            final: Dict[str, str] = {}
            for k, v in tags.items():
                s = str(v)
                if len(s) > 250:
                    s = s[:247] + "..."
                final[k] = s
            return final
        except Exception as e:
            return {"environment.error": f"Failed to generate tags: {e}"}

    def export_environment(self, filepath: str, include_sensitive: bool = False) -> None:
        try:
            info = self.get_environment_info()
            if not include_sensitive:
                # Scrub sensitive env vars and network detail
                if "environment" in info:
                    tracked = info["environment"].get("tracked_variables", {})
                    redact = {k: v for k, v in tracked.items() if "KEY" not in k.upper() and "SECRET" not in k.upper() and "TOKEN" not in k.upper()}
                    info["environment"]["tracked_variables"] = redact
                if "hardware" in info and "network" in info["hardware"]:
                    net = info["hardware"]["network"]
                    info["hardware"]["network"] = {
                        "hostname": net.get("hostname"),
                        "interface_count": len(net.get("interfaces", [])),
                    }
            with open(filepath, "w") as f:
                json.dump(info, f, indent=2, default=str)
            print(f"💾 Exported environment info to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to export environment: {e}")

    def get_reproducibility_hash(self) -> str:
        try:
            info = self.get_environment_info()
            data = {
                "python_version": info.get("python", {}).get("version_info", {}),
                "key_packages": info.get("packages", {}).get("key_packages", {}),
                "git_commit": info.get("git", {}).get("commit"),
                "platform": info.get("system", {}).get("platform"),
                "architecture": info.get("system", {}).get("architecture"),
            }
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception as e:
            return f"error_{abs(hash(str(e)))}"

    # ------------------------
    # Sections
    # ------------------------
    def _get_system_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "node": platform.node(),
                "boot_time": None,
                "uptime_seconds": None,
            }
            if PSUTIL_AVAILABLE:
                bt = psutil.boot_time()
                out["boot_time"] = datetime.fromtimestamp(bt).isoformat()
                out["uptime_seconds"] = int(datetime.now().timestamp() - bt)
            # Firmware/BIOS (best-effort, Linux)
            try:
                bios_v = Path("/sys/class/dmi/id/bios_version")
                if bios_v.exists():
                    out["bios_version"] = bios_v.read_text().strip()
            except Exception:
                pass
            return out
        except Exception as e:
            return {"error": f"Failed to get system info: {e}"}

    def _get_hardware_info(self) -> Dict[str, Any]:
        try:
            return {
                "cpu": self._get_cpu_info(),
                "memory": self._get_memory_info(),
                "gpu": self._get_gpu_info(),
                "disk": self._get_disk_info(),
                "network": self._get_network_info(),
            }
        except Exception as e:
            return {"error": f"Failed to get hardware info: {e}"}

    def _get_cpu_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "physical_cores": None,
                "logical_cores": None,
                "max_frequency": None,
                "current_frequency": None,
                "usage_percent": None,
                "model": platform.processor(),
            }
            if PSUTIL_AVAILABLE:
                out["physical_cores"] = psutil.cpu_count(logical=False)
                out["logical_cores"] = psutil.cpu_count(logical=True)
                try:
                    # Non-blocking instantaneous pct
                    out["usage_percent"] = psutil.cpu_percent(interval=0.0)
                except Exception:
                    pass
                try:
                    freq = psutil.cpu_freq()
                    if freq:
                        out["max_frequency"] = freq.max
                        out["current_frequency"] = freq.current
                except Exception:
                    pass
            return out
        except Exception as e:
            return {"error": f"Failed to get CPU info: {e}"}

    def _get_memory_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "total_gb": None,
                "available_gb": None,
                "used_gb": None,
                "usage_percent": None,
                "swap_total_gb": None,
                "swap_used_gb": None,
            }
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                out.update({
                    "total_gb": round(vm.total / (1024**3), 2),
                    "available_gb": round(vm.available / (1024**3), 2),
                    "used_gb": round(vm.used / (1024**3), 2),
                    "usage_percent": vm.percent,
                })
                try:
                    sw = psutil.swap_memory()
                    out["swap_total_gb"] = round(sw.total / (1024**3), 2)
                    out["swap_used_gb"] = round(sw.used / (1024**3), 2)
                except Exception:
                    pass
            return out
        except Exception as e:
            return {"error": f"Failed to get memory info: {e}"}

    def _nvidia_smi_query(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            r = self._run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
            if not r or r.returncode != 0:
                return info
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            devices = []
            drv = None
            for idx, ln in enumerate(lines):
                # Example: "NVIDIA A100-SXM4-40GB, 535.86.10, 40536 MiB"
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) >= 3:
                    name = parts[0]
                    drv = parts[1]
                    mem = parts[2].split()[0]
                    try:
                        mem_mb = float(mem)
                    except Exception:
                        mem_mb = None
                    devices.append({"id": idx, "name": name, "memory_total_mb": mem_mb})
            if devices:
                info["devices"] = devices
            if drv:
                info["driver_version"] = drv
        except Exception:
            pass
        return info

    def _get_gpu_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "available": False,
                "count": 0,
                "devices": [],
                "cuda_available": False,
                "cuda_version": None,
                "driver_version": None,
            }
            devices: List[Dict[str, Any]] = []

            # Fast path: GPUtil
            if GPUTIL_AVAILABLE:
                try:
                    for g in GPUtil.getGPUs():
                        devices.append({
                            "id": g.id,
                            "name": g.name,
                            "memory_total_mb": g.memoryTotal,
                            "memory_used_mb": g.memoryUsed,
                            "memory_free_mb": g.memoryFree,
                            "load_percent": round(getattr(g, "load", 0.0) * 100, 1),
                            "temperature_c": getattr(g, "temperature", None),
                            "uuid": getattr(g, "uuid", None),
                        })
                except Exception:
                    pass

            # NVML
            if not devices and PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    n = pynvml.nvmlDeviceGetCount()
                    for i in range(n):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        devices.append({
                            "id": i,
                            "name": name,
                            "memory_total_mb": mem.total // (1024**2),
                            "memory_used_mb": mem.used // (1024**2),
                            "memory_free_mb": mem.free // (1024**2),
                        })
                    try:
                        out["driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
                    except Exception:
                        pass
                except Exception:
                    pass
                finally:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass

            # nvidia-smi (for driver/mem if still empty)
            if not devices:
                smi = self._nvidia_smi_query()
                if smi.get("devices"):
                    devices = smi["devices"]
                if smi.get("driver_version"):
                    out["driver_version"] = smi["driver_version"]

            # Torch CUDA for version + count if available
            if TORCH_AVAILABLE:
                try:
                    out["cuda_available"] = torch.cuda.is_available()
                    if out["cuda_available"]:
                        out["cuda_version"] = torch.version.cuda
                        # If no devices yet, at least list names from torch
                        if not devices:
                            n = torch.cuda.device_count()
                            for i in range(n):
                                p = torch.cuda.get_device_properties(i)
                                devices.append({
                                    "id": i,
                                    "name": p.name,
                                    "memory_total_mb": p.total_memory // (1024**2),
                                    "compute_capability": f"{p.major}.{p.minor}",
                                    "multiprocessor_count": p.multi_processor_count,
                                })
                except Exception:
                    pass

            if devices:
                out["available"] = True
                out["count"] = len(devices)
                out["devices"] = devices
            return out
        except Exception as e:
            return {"error": f"Failed to get GPU info: {e}"}

    def _get_disk_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {"partitions": [], "total_gb": 0, "used_gb": 0, "free_gb": 0}
            if PSUTIL_AVAILABLE:
                for part in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        row = {
                            "device": part.device,
                            "mountpoint": part.mountpoint,
                            "fstype": part.fstype,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "used_gb": round(usage.used / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                            "usage_percent": round(usage.percent, 1),
                        }
                        out["partitions"].append(row)
                        if part.mountpoint in ("/", "C:\\"):
                            out["total_gb"] += row["total_gb"]
                            out["used_gb"] += row["used_gb"]
                            out["free_gb"] += row["free_gb"]
                    except Exception:
                        continue
            return out
        except Exception as e:
            return {"error": f"Failed to get disk info: {e}"}

    def _get_network_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {"interfaces": [], "hostname": socket.gethostname(), "fqdn": socket.getfqdn()}
            if PSUTIL_AVAILABLE:
                for name, addrs in psutil.net_if_addrs().items():
                    iface = {"name": name, "addresses": []}
                    for a in addrs:
                        iface["addresses"].append({
                            "family": str(a.family),
                            "address": a.address,
                            "netmask": getattr(a, "netmask", None),
                            "broadcast": getattr(a, "broadcast", None),
                        })
                    out["interfaces"].append(iface)
            return out
        except Exception as e:
            return {"error": f"Failed to get network info: {e}"}

    def _get_software_info(self) -> Dict[str, Any]:
        try:
            return {
                "os_name": os.name,
                "os_environ_count": len(os.environ),
                "shell": os.getenv("SHELL"),
                "terminal": os.getenv("TERM"),
                "user": os.getenv("USER") or os.getenv("USERNAME"),
                "home": os.getenv("HOME") or os.getenv("USERPROFILE"),
                "path_entries": len(os.getenv("PATH", "").split(os.pathsep)),
                "locale": {
                    "lang": os.getenv("LANG"),
                    "lc_all": os.getenv("LC_ALL"),
                    "timezone": os.getenv("TZ"),
                },
            }
        except Exception as e:
            return {"error": f"Failed to get software info: {e}"}

    def _get_python_info(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {
                "version": sys.version,
                "version_info": {"major": sys.version_info.major, "minor": sys.version_info.minor, "micro": sys.version_info.micro},
                "executable": sys.executable,
                "prefix": sys.prefix,
                "path": sys.path[:5],
                "path_count": len(sys.path),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
                "build": platform.python_build(),
                "api_version": sys.api_version,
                "maxsize": sys.maxsize,
                "float_info": {"max": sys.float_info.max, "epsilon": sys.float_info.epsilon, "dig": sys.float_info.dig},
            }
            # venv
            info["virtual_env"] = bool(
                getattr(sys, "real_prefix", None) or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
            )
            # conda
            if os.getenv("CONDA_DEFAULT_ENV"):
                info["conda_env"] = os.getenv("CONDA_DEFAULT_ENV")
                info["conda_prefix"] = os.getenv("CONDA_PREFIX")
            return info
        except Exception as e:
            return {"error": f"Failed to get Python info: {e}"}

    def _get_package_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {"package_count": 0, "key_packages": {}, "pip_packages": []}
            keys = [
                "torch", "tensorflow", "numpy", "pandas", "scikit-learn",
                "matplotlib", "jupyter", "mlflow", "dvc", "boto3", "pymongo", "psycopg2", "requests", "fastapi",
            ]
            if PKG_RESOURCES_AVAILABLE:
                installed = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
                out["package_count"] = len(installed)
                for k in keys:
                    if k in installed:
                        out["key_packages"][k] = installed[k]
                # keep list short for logs
                out["pip_packages"] = [f"{n}=={v}" for n, v in list(installed.items())[:100]]
            return out
        except Exception as e:
            return {"error": f"Failed to get package info: {e}"}

    def _get_git_info(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {
                "available": self.git_available,
                "is_repo": False,
                "commit": None,
                "branch": None,
                "remote_url": None,
                "status": None,
                "tag": None,
                "commit_count": None,
                "last_commit_date": None,
            }
            if not self.git_available:
                return info

            # In repo?
            r = self._run(["git", "rev-parse", "--git-dir"])
            if not r or r.returncode != 0:
                return info
            info["is_repo"] = True

            def _grab(cmd: List[str]) -> Optional[str]:
                rr = self._run(cmd)
                return rr.stdout.strip() if rr and rr.stdout else None

            info["commit"] = _grab(["git", "rev-parse", "HEAD"])
            info["branch"] = _grab(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            info["remote_url"] = _grab(["git", "config", "--get", "remote.origin.url"])
            # porcelain status (clean/dirty)
            st = self._run(["git", "status", "--porcelain"])
            info["status"] = "clean" if (st and not st.stdout.strip()) else "dirty"
            info["tag"] = _grab(["git", "describe", "--tags", "--abbrev=0"]) or None
            cc = _grab(["git", "rev-list", "--count", "HEAD"])
            info["commit_count"] = int(cc) if (cc and cc.isdigit()) else None
            info["last_commit_date"] = _grab(["git", "log", "-1", "--format=%ci"])
            return info
        except Exception as e:
            return {"error": f"Failed to get git info: {e}"}

    def _get_environment_variables(self) -> Dict[str, Any]:
        try:
            keys = [
                "PATH", "PYTHONPATH", "HOME", "USER", "SHELL",
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "CONDA_DEFAULT_ENV", "VIRTUAL_ENV",
                "AWS_REGION", "AWS_DEFAULT_REGION",
                "MLFLOW_TRACKING_URI", "MONGO_CONNECTION_STRING",
            ]
            tracked: Dict[str, str] = {}
            for k in keys:
                v = os.getenv(k)
                if v is None:
                    continue
                if len(v) > 200:
                    v = v[:197] + "..."
                tracked[k] = v
            return {"tracked_variables": tracked, "total_variables": len(os.environ)}
        except Exception as e:
            return {"error": f"Failed to get environment variables: {e}"}

    def _get_platform_info(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {"system": platform.system(), "platform_details": {}}
            s = out["system"]
            if s == "Linux":
                out["platform_details"] = self._get_linux_info()
            elif s == "Darwin":
                out["platform_details"] = self._get_macos_info()
            elif s == "Windows":
                out["platform_details"] = self._get_windows_info()
            return out
        except Exception as e:
            return {"error": f"Failed to get platform info: {e}"}

    def _get_linux_info(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {"kernel_version": platform.release()}
            try:
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("ID="):
                            info["distro_id"] = line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("VERSION_ID="):
                            info["distro_version"] = line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("PRETTY_NAME="):
                            info["distro_name"] = line.split("=", 1)[1].strip().strip('"')
            except Exception:
                pass
            # CPU model from /proc/cpuinfo
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for ln in f:
                        if ln.lower().startswith("model name"):
                            info["cpu_model"] = ln.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
            return info
        except Exception as e:
            return {"error": f"Failed to get Linux info: {e}"}

    def _get_macos_info(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {"version": platform.mac_ver()[0], "machine": platform.machine()}
            r = self._run(["sw_vers"])
            if r and r.returncode == 0:
                for ln in r.stdout.splitlines():
                    if ln.startswith("ProductName:"):
                        info["product_name"] = ln.split(":", 1)[1].strip()
                    elif ln.startswith("ProductVersion:"):
                        info["product_version"] = ln.split(":", 1)[1].strip()
                    elif ln.startswith("BuildVersion:"):
                        info["build_version"] = ln.split(":", 1)[1].strip()
            return info
        except Exception as e:
            return {"error": f"Failed to get macOS info: {e}"}

    def _get_windows_info(self) -> Dict[str, Any]:
        try:
            return {
                "version": platform.version(),
                "edition": getattr(platform, "win32_edition", lambda: None)(),
                "is_64bit": platform.machine().endswith("64"),
            }
        except Exception as e:
            return {"error": f"Failed to get Windows info: {e}"}

    def _detect_cloud_platform(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "detected": False,
            "platform": None,
            "instance_type": None,
            "region": None,
            "availability_zone": None,
        }
        if not _URL_OK:
            return out
        # AWS
        try:
            req = _urlreq.Request("http://169.254.169.254/latest/meta-data/instance-type", headers={"User-Agent": "curl/7"})
            with _urlreq.urlopen(req, timeout=1.5) as resp:
                out.update({"detected": True, "platform": "AWS", "instance_type": resp.read().decode("utf-8")})
            # Region & AZ (best-effort)
            try:
                with _urlreq.urlopen("http://169.254.169.254/latest/meta-data/placement/region", timeout=1.0) as r2:
                    out["region"] = r2.read().decode("utf-8")
            except Exception:
                pass
            try:
                with _urlreq.urlopen("http://169.254.169.254/latest/meta-data/placement/availability-zone", timeout=1.0) as r3:
                    out["availability_zone"] = r3.read().decode("utf-8")
            except Exception:
                pass
            return out
        except Exception:
            pass
        # GCP
        try:
            req = _urlreq.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
                headers={"Metadata-Flavor": "Google"},
            )
            with _urlreq.urlopen(req, timeout=1.5) as resp:
                mt = resp.read().decode("utf-8")
                out.update({"detected": True, "platform": "GCP", "instance_type": mt.split("/")[-1]})
            return out
        except Exception:
            pass
        # Azure
        try:
            req = _urlreq.Request(
                "http://169.254.169.254/metadata/instance/compute/vmSize",
                headers={"Metadata": "true"},
            )
            with _urlreq.urlopen(req, timeout=1.5) as resp:
                out.update({"detected": True, "platform": "Azure", "instance_type": resp.read().decode("utf-8")})
            return out
        except Exception:
            pass
        return out

    def _get_container_info(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {"docker": self.docker_available, "kubernetes": False, "container_id": None, "image_name": None}
            # K8s
            if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount") or os.getenv("KUBERNETES_SERVICE_HOST"):
                info["kubernetes"] = True
                try:
                    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
                        info["k8s_namespace"] = f.read().strip()
                except Exception:
                    pass
            # Docker container ID
            if self.docker_available:
                try:
                    with open("/proc/self/cgroup", "r") as f:
                        for line in f:
                            if "docker" in line or "containerd" in line:
                                info["container_id"] = line.strip().split("/")[-1][:12]
                                break
                except Exception:
                    pass
            return info
        except Exception as e:
            return {"error": f"Failed to get container info: {e}"}

    # ------------------------
    # Dataset fingerprint (optional)
    # ------------------------
    def get_dataset_fingerprint(self, dataset_root: str = "data", max_files: int = 5000) -> Dict[str, Any]:
        """
        Compute a lightweight fingerprint of a dataset directory (no DVC required).
        - Hash over relative paths + sizes (not file content — fast).
        - Limits to `max_files` entries to avoid long walks.

        Returns: {root, hash, hash_short, file_count, total_size_mb}
        """
        root = Path(dataset_root)
        h = hashlib.sha256()
        total = 0
        count = 0
        try:
            if root.exists():
                for i, fp in enumerate(sorted(root.rglob("*"))):
                    if i >= max_files:
                        break
                    if fp.is_file():
                        try:
                            rel = str(fp.relative_to(root))
                        except Exception:
                            rel = str(fp)
                        size = fp.stat().st_size
                        total += size
                        count += 1
                        h.update(rel.encode("utf-8"))
                        h.update(str(size).encode("utf-8"))
            digest = h.hexdigest()
            return {
                "root": str(root),
                "hash": digest,
                "hash_short": digest[:8],
                "file_count": count,
                "total_size_mb": round(total / (1024 * 1024), 2),
            }
        except Exception as e:
            return {"root": str(root), "error": str(e)}

    # ------------------------
    # Compare environments (optional utility)
    # ------------------------
    def compare_environments(self, other_env_file: str) -> Dict[str, Any]:
        try:
            current_env = self.get_environment_info()
            with open(other_env_file, "r") as f:
                other_env = json.load(f)
            comp: Dict[str, Any] = {
                "comparison_timestamp": datetime.now().isoformat(),
                "current_file": "current_environment",
                "other_file": other_env_file,
                "differences": {},
                "similarities": {},
                "compatibility_score": 0.0,
            }
            sections = ["system", "hardware", "python", "packages"]
            total = 0.0
            for s in sections:
                d, sim, score = self._compare_section(current_env.get(s, {}), other_env.get(s, {}), s)
                if d:
                    comp["differences"][s] = d
                if sim:
                    comp["similarities"][s] = sim
                total += score
            comp["compatibility_score"] = total / len(sections) if sections else 0.0
            return comp
        except Exception as e:
            return {"error": f"Failed to compare environments: {e}"}

    def _compare_section(self, current: Dict, other: Dict, section_name: str) -> Tuple[Dict, Dict, float]:
        differences: Dict[str, Any] = {}
        similarities: Dict[str, Any] = {}
        score = 0.0
        try:
            if section_name == "system":
                if current.get("platform") == other.get("platform"):
                    similarities["platform"] = current.get("platform")
                    score += 0.5
                else:
                    differences["platform"] = {"current": current.get("platform"), "other": other.get("platform")}
                if current.get("architecture") == other.get("architecture"):
                    similarities["architecture"] = current.get("architecture")
                    score += 0.5
                else:
                    differences["architecture"] = {"current": current.get("architecture"), "other": other.get("architecture")}
            elif section_name == "hardware":
                c_cpu, o_cpu = current.get("cpu", {}), other.get("cpu", {})
                if c_cpu.get("physical_cores") == o_cpu.get("physical_cores"):
                    similarities["cpu_cores"] = c_cpu.get("physical_cores")
                    score += 0.25
                else:
                    differences["cpu_cores"] = {"current": c_cpu.get("physical_cores"), "other": o_cpu.get("physical_cores")}
                c_mem, o_mem = current.get("memory", {}), other.get("memory", {})
                cg, og = c_mem.get("total_gb", 0) or 0, o_mem.get("total_gb", 0) or 0
                if abs(cg - og) < 1:
                    similarities["memory_gb"] = cg
                    score += 0.25
                else:
                    differences["memory_gb"] = {"current": cg, "other": og}
                c_gpu, o_gpu = current.get("gpu", {}), other.get("gpu", {})
                if c_gpu.get("available") == o_gpu.get("available"):
                    similarities["gpu_available"] = c_gpu.get("available")
                    score += 0.25
                    if c_gpu.get("available") and c_gpu.get("count") == o_gpu.get("count"):
                        similarities["gpu_count"] = c_gpu.get("count")
                        score += 0.25
                    elif c_gpu.get("available"):
                        differences["gpu_count"] = {"current": c_gpu.get("count"), "other": o_gpu.get("count")}
                else:
                    differences["gpu_available"] = {"current": c_gpu.get("available"), "other": o_gpu.get("available")}
            elif section_name == "python":
                cv, ov = current.get("version_info", {}), other.get("version_info", {})
                if (cv.get("major"), cv.get("minor")) == (ov.get("major"), ov.get("minor")):
                    similarities["python_version"] = f"{cv.get('major')}.{cv.get('minor')}"
                    score += 0.5
                else:
                    differences["python_version"] = {
                        "current": f"{cv.get('major',0)}.{cv.get('minor',0)}",
                        "other": f"{ov.get('major',0)}.{ov.get('minor',0)}",
                    }
                if current.get("virtual_env") == other.get("virtual_env"):
                    similarities["virtual_env"] = current.get("virtual_env")
                    score += 0.5
                else:
                    differences["virtual_env"] = {"current": current.get("virtual_env"), "other": other.get("virtual_env")}
            elif section_name == "packages":
                cp, op = current.get("key_packages", {}), other.get("key_packages", {})
                common = set(cp) & set(op)
                diffs: Dict[str, Any] = {}
                same = 0
                for k in common:
                    if cp[k] == op[k]:
                        similarities[f"package_{k}"] = cp[k]
                        same += 1
                    else:
                        diffs[k] = {"current": cp[k], "other": op[k]}
                if diffs:
                    differences["package_versions"] = diffs
                if common:
                    score = same / len(common)
            return differences, similarities, score
        except Exception as e:
            return {"error": str(e)}, {}, 0.0

    # ------------------------
    def cleanup(self) -> None:
        self.cache.clear()
        print("🧹 Environment tagger cleaned up")
