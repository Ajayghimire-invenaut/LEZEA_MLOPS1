"""
DVC Backend for LeZeA MLOps — FINAL
===================================

What this gives you
-------------------
- Safe, graceful setup: works even if DVC or PyYAML aren't installed.
- Dataset tracking (`track_dataset`) with optional metadata injected into the .dvc file.
- One-call dataset versioning (`version_dataset`) that:
  * tracks paths, builds a manifest, tags in Git, and (optionally) pushes.
- Active dataset fingerprinting (`get_active_dataset_version`) used by the tracker:
  * reads `dvc.lock` (or all .dvc files) to compute a stable version hash,
    returns `version_tag`, `file_count`, `total_size_mb`, `git_commit`, etc.
- Remotes, push/pull, integrity validation, lineage, and pipeline helpers.
- Clean JSON structures, consistent with what the tracker expects.

All operations degrade cleanly when DVC/PyYAML/Git aren’t available.
"""

from __future__ import annotations

import os
import json
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Optional deps
try:
    import dvc.api  # noqa: F401
    import dvc.repo
    from dvc.exceptions import DvcException
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
    """

    def __init__(self, config):
        """
        Initialize the DVC backend. No-op if DVC (or PyYAML) is missing.

        Args:
            config: Your project config object (not strictly required).
        """
        self.config = config
        # Try to honor a project root from config, otherwise CWD
        self.project_root: Path = Path(getattr(config, "project_root", Path.cwd()))
        self.dvc_dir = self.project_root / ".dvc"
        self.data_dir = self.project_root / "data"

        self.repo = None
        self.available = False
        self.is_initialized = False

        if not DVC_AVAILABLE:
            print("⚠️  DVC not available. Install with: pip install 'dvc[s3]'")
            return
        if not YAML_AVAILABLE:
            print("⚠️  PyYAML not available. Install with: pip install PyYAML")
            return

        self.is_initialized = self._check_dvc_initialized()
        if self.is_initialized:
            try:
                self.repo = dvc.repo.Repo(str(self.project_root))
                self.available = True
                print("✅ DVC backend ready")
            except Exception as e:
                print(f"⚠️  DVC repo initialization failed: {e}")
        else:
            print("⚠️  DVC not initialized in this project (run `dvc init`).")

    # ---------------------------------------------------------------------
    # Basics / utilities
    # ---------------------------------------------------------------------
    def _check_dvc_initialized(self) -> bool:
        return self.dvc_dir.exists() and (self.dvc_dir / "config").exists()

    def _run(self, cmd: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a shell command relative to project root.
        Raises on failure so callers can handle.
        """
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=capture_output,
            text=True,
            check=True,
        )

    def _dvc(self, args: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run `dvc <args>`.
        """
        return self._run(["dvc"] + args, capture_output=capture_output)

    def _git(self, args: List[str], *, capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        """
        Run `git <args>` if git is available; return None on FileNotFoundError.
        """
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
        """
        Add and commit files; returns commit sha (short) if possible.
        """
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
            # It's fine if working tree clean, etc.
            return None

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def initialize_dvc(self, remote_url: Optional[str] = None) -> bool:
        """
        Initialize DVC for the project. Creates default data dirs and sets a remote.

        Returns:
            True on success.
        """
        if not DVC_AVAILABLE or not YAML_AVAILABLE:
            return False
        try:
            if self.is_initialized:
                print("✅ DVC already initialized")
                return True

            self._dvc(["init"])
            print("✅ DVC initialized")

            # Ensure data dirs exist
            for p in [self.data_dir, self.data_dir / "raw", self.data_dir / "processed", self.data_dir / "external"]:
                p.mkdir(parents=True, exist_ok=True)

            if remote_url:
                self.add_remote("origin", remote_url, default=True)

            # Create Repo handle
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

        Returns:
            Path to the `.dvc` file or None.
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

            if metadata and YAML_AVAILABLE:
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
        """
        Create a dataset version (tracks paths, writes manifest, optional git tag/push).
        """
        if not self.is_initialized:
            print("⚠️  DVC not initialized; returning a minimal marker.")
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

        # Track and accumulate sizes
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
                        total_size += fp.stat().st_size
                        file_count += 1
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
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "data_hash": self._calculate_dataset_hash(data_paths),
        }

        # Write manifest and track it
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self.data_dir / f"{version_tag}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(version_info, f, indent=2)
            self.track_dataset(str(manifest_path))
        except Exception as e:
            print(f"⚠️  Could not write/track manifest: {e}")

        # Optionally git tag & push data
        if git_tag:
            self._create_git_tag(version_tag, f"Dataset version: {dataset_name}")
        if push:
            try:
                self.push_data()
            except Exception:
                pass

        print(f"📦 Dataset version created: {version_tag} | Files: {file_count}, Size: {version_info['total_size_mb']} MB")
        return version_info

    def get_active_dataset_version(self, dataset_root: str = "data/") -> Dict[str, Any]:
        """
        Produce a *fingerprint* of the currently *checked out* dataset state.

        Strategy:
        - Prefer hashing `dvc.lock` contents (outs' checksums) for stability.
        - Fall back to hashing all tracked `.dvc` file entries under dataset_root.
        - Attach current Git commit (short) if available.

        Returns (example):
            {
              "version_tag": "lock-c3f9c1a",
              "lock_hash": "...",
              "git_commit": "c3f9c1a",
              "file_count": 1234,
              "total_size_mb": 456.78,
              "root": "data/"
            }
        """
        root = Path(dataset_root)
        info: Dict[str, Any] = {"root": str(root)}

        # Helper: current git commit short
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
                # dvc.lock structure may differ by DVC version; handle defensively
                for _, stage in (lock.get("stages", {}) or {}).items():
                    for out in stage.get("outs", []) or []:
                        # Prefer checksum fields (md5, etag, etc.), else path
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

        # If we didn’t get a lock hash, fall back to .dvc files under root
        if "version_tag" not in info:
            try:
                dvc_files = list((root if root.is_absolute() else (self.project_root / root)).rglob("*.dvc"))
                keys = []
                for df in dvc_files:
                    try:
                        with open(df, "r") as f:
                            d = yaml.safe_load(f) or {}
                        # Typical structure: {'outs': [{'md5': '...', 'path': '...'}, ...]}
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

        # Optional: size and count (best effort)
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
            info["file_count"] = file_count
            info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        except Exception:
            pass

        return info

    # ---------------------------------------------------------------------
    # Hash helpers
    # ---------------------------------------------------------------------
    def _calculate_dataset_hash(self, data_paths: List[str]) -> str:
        """
        Content hash across provided paths (file bytes + relative names).
        """
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
                            # include relative names for stability
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

    def get_remote_info(self) -> Dict[str, Any]:
        try:
            r = self._dvc(["remote", "list"])
            remotes: Dict[str, str] = {}
            for line in r.stdout.strip().splitlines():
                if line.strip():
                    name, _, url = line.partition("\t")
                    if name and url:
                        remotes[name.strip()] = url.strip()
            default_remote = None
            try:
                d = self._dvc(["remote", "default"])
                default_remote = d.stdout.strip()
            except Exception:
                pass
            return {"remotes": remotes, "default_remote": default_remote, "remote_count": len(remotes)}
        except Exception as e:
            print(f"❌ Failed to get remote info: {e}")
            return {}

    # ---------------------------------------------------------------------
    # Status / integrity / lineage
    # ---------------------------------------------------------------------
    def get_dataset_status(self) -> Dict[str, Any]:
        try:
            result = self._dvc(["status"])
            info = {
                "timestamp": datetime.now().isoformat(),
                "status_output": result.stdout,
                "is_clean": ("up to date" in result.stdout.lower()) or (result.stdout.strip() == ""),
                "tracked_files": [],
            }
            for dvc_file in self.project_root.rglob("*.dvc"):
                info["tracked_files"].append({
                    "dvc_file": str(dvc_file.relative_to(self.project_root)),
                    "data_file": str(dvc_file.with_suffix("").relative_to(self.project_root)),
                })
            return info
        except Exception as e:
            print(f"❌ Failed to get dataset status: {e}")
            return {}

    def validate_data_integrity(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            cmd = ["status"]
            if data_path:
                cmd.append(data_path)
            r = self._dvc(cmd)
            ok = ("up to date" in r.stdout.lower()) or (r.stdout.strip() == "")
            issues = []
            if not ok and r.stdout.strip():
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line and ("up to date" not in line.lower()):
                        issues.append(line)
            return {"timestamp": datetime.now().isoformat(), "is_valid": ok, "issues": issues, "status_output": r.stdout}
        except Exception as e:
            print(f"❌ Failed to validate data integrity: {e}")
            return {"is_valid": False, "error": str(e)}

    def get_data_lineage(self, data_path: str) -> Dict[str, Any]:
        try:
            dvc_file = f"{data_path}.dvc"
            if not Path(dvc_file).exists():
                return {"error": "Data not tracked by DVC"}
            d = {}
            if YAML_AVAILABLE:
                with open(dvc_file, "r") as f:
                    d = yaml.safe_load(f) or {}
            lineage = {
                "data_path": data_path,
                "dvc_file": dvc_file,
                "dvc_data": d,
                "tracked_since": datetime.fromtimestamp(Path(dvc_file).stat().st_ctime).isoformat(),
                "last_modified": datetime.fromtimestamp(Path(dvc_file).stat().st_mtime).isoformat(),
                "metadata": d.get("meta", {}),
            }
            try:
                if self._git_is_repo():
                    gl = self._git(["log", "--oneline", dvc_file])
                    if gl and gl.returncode == 0:
                        lineage["git_history"] = gl.stdout.strip().splitlines()
            except Exception:
                pass
            try:
                dag = self._dvc(["dag", dvc_file])
                lineage["pipeline_dependencies"] = dag.stdout
            except Exception:
                lineage["pipeline_dependencies"] = None
            return lineage
        except Exception as e:
            print(f"❌ Failed to get data lineage: {e}")
            return {"error": str(e)}

    # ---------------------------------------------------------------------
    # Version listings / compare / cleanup / export
    # ---------------------------------------------------------------------
    def list_dataset_versions(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            versions: List[Dict[str, Any]] = []
            pattern = f"{dataset_name}_v*_manifest.json" if dataset_name else "*_manifest.json"
            for mf in (self.data_dir.glob(pattern) if self.data_dir.exists() else []):
                try:
                    with open(mf, "r") as f:
                        versions.append(json.load(f))
                except Exception as e:
                    print(f"⚠️  Could not read {mf}: {e}")
            versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return versions
        except Exception as e:
            print(f"❌ Failed to list dataset versions: {e}")
            return []

    def compare_dataset_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        try:
            m1 = self.data_dir / f"{version1}_manifest.json"
            m2 = self.data_dir / f"{version2}_manifest.json"
            if not m1.exists():
                return {"error": f"Version {version1} not found"}
            if not m2.exists():
                return {"error": f"Version {version2} not found"}
            with open(m1, "r") as f:
                i1 = json.load(f)
            with open(m2, "r") as f:
                i2 = json.load(f)
            return {
                "version1": version1,
                "version2": version2,
                "timestamp1": i1.get("timestamp"),
                "timestamp2": i2.get("timestamp"),
                "size_diff_mb": i2.get("total_size_mb", 0) - i1.get("total_size_mb", 0),
                "file_count_diff": i2.get("file_count", 0) - i1.get("file_count", 0),
                "hash_changed": i1.get("data_hash") != i2.get("data_hash"),
                "paths_added": sorted(set(i2.get("data_paths", [])) - set(i1.get("data_paths", []))),
                "paths_removed": sorted(set(i1.get("data_paths", [])) - set(i2.get("data_paths", []))),
            }
        except Exception as e:
            print(f"❌ Failed to compare dataset versions: {e}")
            return {"error": str(e)}

    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        try:
            datasets: Dict[str, List[Dict[str, Any]]] = {}
            for mf in (self.data_dir.glob("*_manifest.json") if self.data_dir.exists() else []):
                try:
                    with open(mf, "r") as f:
                        vi = json.load(f)
                    datasets.setdefault(vi.get("dataset_name", "unknown"), []).append({"manifest_file": mf, "info": vi})
                except Exception:
                    continue
            cleaned = 0
            for _, rows in datasets.items():
                rows.sort(key=lambda x: x["info"].get("timestamp", ""), reverse=True)
                for row in rows[keep_versions:]:
                    try:
                        row["manifest_file"].unlink(missing_ok=True)
                        for dvc_file in row["info"].get("dvc_files", []):
                            Path(dvc_file).unlink(missing_ok=True)
                        cleaned += 1
                        print(f"🗑️  Removed old dataset version: {row['info'].get('version_tag')}")
                    except Exception as e:
                        print(f"⚠️  Failed to remove old version: {e}")
            if cleaned:
                print(f"🧹 Cleaned {cleaned} old dataset version(s)")
            return cleaned
        except Exception as e:
            print(f"❌ Failed to cleanup old versions: {e}")
            return 0

    def export_dataset_info(self, output_file: str) -> None:
        try:
            export = {
                "export_timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "dvc_initialized": self.is_initialized,
                "dataset_versions": self.list_dataset_versions(),
                "dataset_status": self.get_dataset_status(),
                "tracked_files": [],
            }
            for dvc_file in (self.project_root.rglob("*.dvc") if self.project_root.exists() else []):
                try:
                    with open(dvc_file, "r") as f:
                        d = yaml.safe_load(f) if YAML_AVAILABLE else {}
                except Exception:
                    d = {}
                export["tracked_files"].append({
                    "dvc_file": str(dvc_file.relative_to(self.project_root)),
                    "data_file": str(dvc_file.with_suffix("").relative_to(self.project_root)),
                    "dvc_data": d,
                })
            with open(output_file, "w") as f:
                json.dump(export, f, indent=2, default=str)
            print(f"📁 Exported dataset info to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to export dataset info: {e}")

    # ---------------------------------------------------------------------
    # Checkout
    # ---------------------------------------------------------------------
    def checkout_dataset_version(self, version_tag: str) -> bool:
        """
        Checkout files from a specific dataset version manifest and try git tag.
        """
        try:
            mf = self.data_dir / f"{version_tag}_manifest.json"
            if not mf.exists():
                print(f"❌ Version manifest not found: {version_tag}")
                return False
            with open(mf, "r") as f:
                info = json.load(f)
            for dvc_file in info.get("dvc_files", []):
                if Path(dvc_file).exists():
                    self._dvc(["checkout", dvc_file])
            # Try git tag checkout (optional)
            if self._git_is_repo():
                try:
                    self._git(["checkout", version_tag])
                    print(f"🏷️  Checked out Git tag: {version_tag}")
                except Exception:
                    print(f"⚠️  Git tag {version_tag} not found; continuing with DVC only")
            print(f"✅ Checked out dataset version: {version_tag}")
            return True
        except Exception as e:
            print(f"❌ Failed to checkout dataset version: {e}")
            return False

    # ---------------------------------------------------------------------
    # Cleanup / context mgmt
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
