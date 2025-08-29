# lezea_mlops/backends/dvc_backend.py
# DVC Backend for LeZeA MLOps — hardened

from __future__ import annotations

import os
import json
import subprocess
import hashlib
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Optional deps
try:
    import dvc.api  # noqa: F401
    import dvc.repo
    DVC_AVAILABLE = True
except Exception:
    DVC_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


class DVCBackend:
    """
    DVC backend for dataset versioning and data pipeline management.

    Notes:
    - Exposes `.available` and `ping()` so the tracker health check works.
    - Degrades gracefully when DVC / PyYAML / git CLI are missing.
    """

    def __init__(self, config):
        self.config = config
        self.project_root: Path = Path(getattr(config, "project_root", Path.cwd()))
        self.dvc_dir = self.project_root / ".dvc"
        self.data_dir = self.project_root / "data"

        self.repo = None
        self.available: bool = False
        self.is_initialized: bool = False
        self._dvc_cli_ok: bool = shutil.which("dvc") is not None
        self._git_cli_ok: bool = shutil.which("git") is not None

        if not DVC_AVAILABLE:
            print("⚠️  DVC python package not available. Install: pip install 'dvc[s3]'")
            return
        if not YAML_AVAILABLE:
            print("⚠️  PyYAML not available. Install: pip install PyYAML")
            return

        self.is_initialized = self._check_dvc_initialized()
        if self.is_initialized:
            try:
                self.repo = dvc.repo.Repo(str(self.project_root))
                self.available = True
                print("✅ DVC backend ready")
            except Exception as e:
                print(f"⚠️  DVC repo open failed: {e}")
        else:
            print("⚠️  DVC not initialized in this project (run `dvc init`).")

    # ---------------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------------
    def ping(self) -> bool:
        """Used by tracker._health_check()."""
        return bool(self.available and self.is_initialized)

    # ---------------------------------------------------------------------
    # Shell helpers
    # ---------------------------------------------------------------------
    def _run(self, cmd: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=capture_output,
            text=True,
            check=True,
        )

    def _dvc(self, args: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        if not self._dvc_cli_ok:
            raise FileNotFoundError("`dvc` CLI not found in PATH")
        return self._run(["dvc"] + args, capture_output=capture_output)

    def _git(self, args: List[str], *, capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        if not self._git_cli_ok:
            return None
        try:
            return self._run(["git"] + args, capture_output=capture_output)
        except FileNotFoundError:
            return None

    def _git_is_repo(self) -> bool:
        try:
            r = self._git(["rev-parse", "--is-inside-work-tree"])
            return bool(r and r.stdout.strip() == "true")
        except Exception:
            return False

    def _git_commit(self, message: str, add_paths: Optional[List[str]] = None) -> Optional[str]:
        if not self._git_is_repo():
            return None
        try:
            if add_paths:
                self._git(["add"] + add_paths)
            else:
                self._git(["add", "-A"])
            self._git(["commit", "-m", message])
            sha = self._git(["rev-parse", "--short", "HEAD"])
            return sha.stdout.strip() if sha else None
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Basics / initialization
    # ---------------------------------------------------------------------
    def _check_dvc_initialized(self) -> bool:
        return self.dvc_dir.exists() and (self.dvc_dir / "config").exists()

    def initialize_dvc(self, remote_url: Optional[str] = None) -> bool:
        """Initialize DVC and optionally set a default remote."""
        if not (DVC_AVAILABLE and YAML_AVAILABLE and self._dvc_cli_ok):
            return False
        try:
            if self.is_initialized:
                print("✅ DVC already initialized")
                return True

            self._dvc(["init"])
            print("✅ DVC initialized")

            for p in [self.data_dir, self.data_dir / "raw", self.data_dir / "processed", self.data_dir / "external"]:
                p.mkdir(parents=True, exist_ok=True)

            if remote_url:
                self.add_remote(name, remote_url, default=True)

            self.repo = dvc.repo.Repo(str(self.project_root))
            self.is_initialized = True
            self.available = True
            return True
        except Exception as e:
            print(f"❌ Failed to initialize DVC: {e}")
            return False

    def add_remote(self, name: str, url: str, default: bool = False) -> bool:
        try:
            self._dvc(["remote", "add", name, url])
            if default:
                self._dvc(["remote", "default", name])
            print(f"✅ Added DVC remote: {name} -> {url}")
            return True
        except Exception as e:
            print(f"❌ Failed to add remote: {e}")
            return False

    # ---------------------------------------------------------------------
    # Tracking
    # ---------------------------------------------------------------------
    def track_dataset(self, data_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Track a dataset/file/dir via `dvc add`. Optionally injects metadata into the .dvc file.
        """
        if not self.is_initialized:
            print("⚠️  DVC not initialized; skipping track.")
            return None
        try:
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data path does not exist: {path}")
            self._dvc(["add", str(path)])
            dvc_file = f"{str(path)}.dvc"

            if metadata and YAML_AVAILABLE and Path(dvc_file).exists():
                self._add_dataset_metadata(dvc_file, metadata)

            print(f"📁 Dataset tracked: {path} -> {dvc_file}")
            return dvc_file
        except Exception as e:
            print(f"❌ Failed to track dataset: {e}")
            return None

    def _add_dataset_metadata(self, dvc_file: str, metadata: Dict[str, Any]) -> None:
        try:
            with open(dvc_file, "r") as f:
                d = yaml.safe_load(f) or {}
            d.setdefault("meta", {})
            d["meta"].update({
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
            })
            with open(dvc_file, "w") as f:
                yaml.safe_dump(d, f, default_flow_style=False)
        except Exception as e:
            print(f"⚠️  Could not add metadata to {dvc_file}: {e}")

    # ---------------------------------------------------------------------
    # Versioning APIs
    # ---------------------------------------------------------------------
    def version_dataset(
        self,
        experiment_id: str,
        dataset_name: str,
        data_paths: List[str],
        description: str = "",
        *,
        git_tag: bool = True,
        push: bool = False,
    ) -> Dict[str, Any]:
        """Create a dataset version manifest and (optionally) tag/push."""
        if not self.is_initialized:
            # Minimal marker for tracker
            return {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "version_tag": "manual",
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "data_paths": data_paths,
                "dvc_files": [],
                "total_size_mb": 0.0,
                "file_count": 0,
                "data_hash": "0" * 64,
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"{dataset_name}_v{timestamp}"

        dvc_files: List[str] = []
        total_size = 0
        file_count = 0

        for p in data_paths:
            path = Path(p)
            if not path.exists():
                continue
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
            else:
                for fp in path.rglob("*"):
                    if fp.is_file():
                        file_count += 1
                        try:
                            total_size += fp.stat().st_size
                        except Exception:
                            pass
            dvc_file = self.track_dataset(
                p,
                {
                    "experiment_id": experiment_id,
                    "dataset_name": dataset_name,
                    "version_tag": version_tag,
                    "description": description,
                },
            )
            if dvc_file:
                dvc_files.append(dvc_file)

        version_info = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "version_tag": version_tag,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "data_paths": data_paths,
            "dvc_files": dvc_files,
            "total_size_bytes": int(total_size),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": int(file_count),
            "data_hash": self._calculate_dataset_hash(data_paths),
        }

        # Write and track manifest (best-effort)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self.data_dir / f"{version_tag}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(version_info, f, indent=2)
            self.track_dataset(str(manifest_path))
        except Exception as e:
            print(f"⚠️  Could not write/track manifest: {e}")

        # Optional git tag + push
        if git_tag:
            self._create_git_tag(version_tag, f"Dataset version: {dataset_name}")
        if push:
            try:
                self.push_data()
            except Exception:
                pass

        print(f"📦 Dataset version: {version_tag} | files={file_count}, size={version_info['total_size_mb']} MB")
        return version_info

    def get_active_dataset_version(self, dataset_root: str = "data/") -> Dict[str, Any]:
        """
        Fingerprint of current dataset state:
        - Prefer `dvc.lock` (stable) → lock_hash
        - Fallback to hashing all `.dvc` outs under dataset_root
        - Add current git commit (short), file_count, total_size_mb
        """
        root = Path(dataset_root)
        info: Dict[str, Any] = {"root": str(root)}

        # git commit (short)
        git_sha_short = None
        try:
            if self._git_is_repo():
                r = self._git(["rev-parse", "--short", "HEAD"])
                git_sha_short = r.stdout.strip() if r else None
        except Exception:
            pass

        lock_path = self.project_root / "dvc.lock"
        if YAML_AVAILABLE and lock_path.exists():
            try:
                with open(lock_path, "r") as f:
                    lock = yaml.safe_load(f) or {}
                outs = []
                for _, stage in (lock.get("stages", {}) or {}).items():
                    for out in stage.get("outs", []) or []:
                        key = out.get("md5") or out.get("etag") or out.get("hash") or out.get("path")
                        if key:
                            outs.append(str(key))
                h = hashlib.sha256()
                for x in sorted(outs):
                    h.update(x.encode("utf-8"))
                lock_hash = h.hexdigest()
                info.update({
                    "version_tag": f"lock-{lock_hash[:7]}",
                    "lock_hash": lock_hash,
                    "git_commit": git_sha_short,
                })
            except Exception as e:
                print(f"⚠️  Could not read dvc.lock: {e}")

        if "version_tag" not in info:
            try:
                dvc_files = list((root if root.is_absolute() else (self.project_root / root)).rglob("*.dvc"))
                keys = []
                for df in dvc_files:
                    try:
                        with open(df, "r") as f:
                            d = yaml.safe_load(f) or {}
                        for out in d.get("outs", []) or []:
                            keys.append(out.get("md5") or out.get("etag") or out.get("hash") or out.get("path"))
                    except Exception:
                        continue
                h = hashlib.sha256()
                for x in sorted(filter(None, keys)):
                    h.update(str(x).encode("utf-8"))
                fallback_hash = h.hexdigest() if keys else "0" * 64
                info.update({
                    "version_tag": f"state-{fallback_hash[:7]}",
                    "lock_hash": fallback_hash,
                    "git_commit": git_sha_short,
                })
            except Exception:
                pass

        # Size + count (best-effort)
        try:
            total_size = 0
            file_count = 0
            rr = root if root.is_absolute() else (self.project_root / root)
            if rr.exists():
                for fp in rr.rglob("*"):
                    if fp.is_file():
                        file_count += 1
                        try:
                            total_size += fp.stat().st_size
                        except Exception:
                            pass
            info["file_count"] = int(file_count)
            info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        except Exception:
            pass

        return info

    # ---------------------------------------------------------------------
    # Hash helpers
    # ---------------------------------------------------------------------
    def _calculate_dataset_hash(self, data_paths: List[str]) -> str:
        """Content hash across provided paths (file bytes + relative names)."""
        hasher = hashlib.sha256()
        try:
            for data_path in sorted(map(str, data_paths)):
                path = Path(data_path)
                if not path.exists():
                    continue
                if path.is_file():
                    hasher.update(str(path.name).encode())
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                else:
                    for fp in sorted(path.rglob("*")):
                        if fp.is_file():
                            try:
                                rel = str(fp.relative_to(self.project_root))
                            except Exception:
                                rel = str(fp)
                            hasher.update(rel.encode())
                            with open(fp, "rb") as f:
                                for chunk in iter(lambda: f.read(4096), b""):
                                    hasher.update(chunk)
        except Exception:
            pass
        return hasher.hexdigest()

    def _create_git_tag(self, tag_name: str, message: str) -> None:
        try:
            r = self._git(["tag", "-a", tag_name, "-m", message])
            if r is not None:
                print(f"🏷️  Created Git tag: {tag_name}")
        except Exception as e:
            print(f"⚠️  Could not create Git tag: {e}")

    # ---------------------------------------------------------------------
    # Remote ops
    # ---------------------------------------------------------------------
    def push_data(self, remote_name: Optional[str] = None) -> bool:
        try:
            cmd = ["push"]
            if remote_name:
                cmd += ["-r", remote_name]
            self._dvc(cmd)
            print("📤 Data pushed to DVC remote")
            return True
        except Exception as e:
            print(f"❌ Failed to push data: {e}")
            return False

    def pull_data(self, remote_name: Optional[str] = None) -> bool:
        try:
            cmd = ["pull"]
            if remote_name:
                cmd += ["-r", remote_name]
            self._dvc(cmd)
            print("📥 Data pulled from DVC remote")
            return True
        except Exception as e:
            print(f"❌ Failed to pull data: {e}")
            return False

    # ---------------------------------------------------------------------
    # Cleanup / ctx
    # ---------------------------------------------------------------------
    def close(self) -> None:
        if self.repo:
            try:
                self.repo.close()
            except Exception:
                pass
        print("🧹 DVC backend cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()