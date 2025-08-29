"""
DVC Backend for LeZeA MLOps System

This module provides a comprehensive DVC (Data Version Control) backend for dataset
versioning, data pipeline management, and reproducible machine learning workflows.
It integrates with Git for complete experiment tracking and supports remote storage
backends for scalable data management.

Features:
    - Dataset versioning with automated tracking
    - Data pipeline management and reproducibility
    - Remote storage integration (S3, )
    - Git integration for complete version control
    - Metadata management and dataset fingerprinting
    - Graceful degradation when dependencies are missing
    - Production-ready error handling and logging

"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# DVC dependencies with graceful fallback
try:
    import dvc.api  # noqa: F401
    import dvc.repo
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    dvc = None  # type: ignore

# YAML support for DVC file manipulation
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore


class DVCBackend:
    """
    Professional DVC backend for dataset versioning and data pipeline management.
    
    This class provides a robust interface for DVC operations with comprehensive
    error handling, graceful degradation, and integration with Git workflows.
    
    Features:
        - Automated dataset tracking and versioning
        - Remote storage management and synchronization
        - Git integration for complete version control
        - Metadata injection and dataset fingerprinting
        - Health monitoring and connectivity testing
        - CLI tool integration with proper error handling
        
    Attributes:
        config: Configuration object containing project settings
        project_root: Root directory of the project
        dvc_dir: DVC configuration directory (.dvc)
        data_dir: Primary data storage directory
        repo: DVC repository instance
        available: Boolean indicating backend availability
        is_initialized: Boolean indicating DVC initialization status
    """

    def __init__(self, config):
        """
        Initialize DVC backend with dependency validation and setup.
        
        Args:
            config: Configuration object with project settings
        """
        self.config = config
        self.project_root: Path = Path(getattr(config, "project_root", Path.cwd()))
        self.dvc_dir = self.project_root / ".dvc"
        self.data_dir = self.project_root / "data"

        # Initialize state
        self.repo = None
        self.available: bool = False
        self.is_initialized: bool = False
        
        # Check CLI tool availability
        self._dvc_cli_available: bool = shutil.which("dvc") is not None
        self._git_cli_available: bool = shutil.which("git") is not None

        # Validate dependencies and initialize
        if not self._validate_dependencies():
            return

        self.is_initialized = self._check_dvc_initialization()
        
        if self.is_initialized:
            self._initialize_repository()
        else:
            print("DVC not initialized in this project (run `dvc init`)")

    def _validate_dependencies(self) -> bool:
        """
        Validate required dependencies are available.
        
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        if not DVC_AVAILABLE:
            print("DVC python package not available. Install: pip install 'dvc[s3]'")
            return False
            
        if not YAML_AVAILABLE:
            print("PyYAML not available. Install: pip install PyYAML")
            return False
            
        return True

    def _initialize_repository(self):
        """Initialize DVC repository connection."""
        try:
            self.repo = dvc.repo.Repo(str(self.project_root))
            self.available = True
            print("DVC backend ready")
        except Exception as e:
            print(f"DVC repository initialization failed: {e}")

    # System Health and Connectivity

    def ping(self) -> bool:
        """
        Test DVC backend health and availability.
        
        Used by health check systems to verify backend status.
        
        Returns:
            True if backend is healthy and operational, False otherwise
        """
        return bool(self.available and self.is_initialized)

    # Command Line Interface Integration

    def _run_command(self, command: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Execute system command with proper error handling.
        
        Args:
            command: List of command components
            capture_output: Whether to capture command output
            
        Returns:
            Completed process result
            
        Raises:
            subprocess.CalledProcessError: If command execution fails
        """
        return subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=capture_output,
            text=True,
            check=True,
        )

    def _execute_dvc_command(self, args: List[str], *, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Execute DVC command with availability checking.
        
        Args:
            args: DVC command arguments
            capture_output: Whether to capture command output
            
        Returns:
            Completed process result
            
        Raises:
            FileNotFoundError: If DVC CLI is not available
        """
        if not self._dvc_cli_available:
            raise FileNotFoundError("DVC CLI tool not found in PATH")
        return self._run_command(["dvc"] + args, capture_output=capture_output)

    def _execute_git_command(self, args: List[str], *, capture_output: bool = True) -> Optional[subprocess.CompletedProcess]:
        """
        Execute Git command with graceful failure handling.
        
        Args:
            args: Git command arguments
            capture_output: Whether to capture command output
            
        Returns:
            Completed process result or None if Git is unavailable
        """
        if not self._git_cli_available:
            return None
            
        try:
            return self._run_command(["git"] + args, capture_output=capture_output)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    # Git Integration Methods

    def _is_git_repository(self) -> bool:
        """
        Check if current directory is within a Git repository.
        
        Returns:
            True if inside Git repository, False otherwise
        """
        try:
            result = self._execute_git_command(["rev-parse", "--is-inside-work-tree"])
            return bool(result and result.stdout.strip() == "true")
        except Exception:
            return False

    def _create_git_commit(self, message: str, add_paths: Optional[List[str]] = None) -> Optional[str]:
        """
        Create Git commit with specified files and message.
        
        Args:
            message: Commit message
            add_paths: Optional list of specific paths to add
            
        Returns:
            Short commit SHA if successful, None otherwise
        """
        if not self._is_git_repository():
            return None
            
        try:
            # Stage files for commit
            if add_paths:
                self._execute_git_command(["add"] + add_paths)
            else:
                self._execute_git_command(["add", "-A"])
                
            # Create commit
            self._execute_git_command(["commit", "-m", message])
            
            # Get commit SHA
            result = self._execute_git_command(["rev-parse", "--short", "HEAD"])
            return result.stdout.strip() if result else None
            
        except Exception:
            return None

    def _create_git_tag(self, tag_name: str, message: str) -> None:
        """
        Create annotated Git tag.
        
        Args:
            tag_name: Name for the Git tag
            message: Tag annotation message
        """
        try:
            result = self._execute_git_command(["tag", "-a", tag_name, "-m", message])
            if result is not None:
                print(f"Created Git tag: {tag_name}")
        except Exception as e:
            print(f"Could not create Git tag: {e}")

    # DVC Initialization and Configuration

    def _check_dvc_initialization(self) -> bool:
        """
        Check if DVC is properly initialized in the project.
        
        Returns:
            True if DVC is initialized, False otherwise
        """
        return self.dvc_dir.exists() and (self.dvc_dir / "config").exists()

    def initialize_dvc(self, remote_url: Optional[str] = None) -> bool:
        """
        Initialize DVC in the project with optional remote configuration.
        
        Creates necessary directory structure and configures remote storage
        if URL is provided.
        
        Args:
            remote_url: Optional URL for default remote storage
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not (DVC_AVAILABLE and YAML_AVAILABLE and self._dvc_cli_available):
            return False
            
        try:
            if self.is_initialized:
                print("DVC already initialized")
                return True

            # Initialize DVC
            self._execute_dvc_command(["init"])
            print("DVC initialized successfully")

            # Create standard data directory structure
            data_directories = [
                self.data_dir,
                self.data_dir / "raw",
                self.data_dir / "processed", 
                self.data_dir / "external"
            ]
            
            for directory in data_directories:
                directory.mkdir(parents=True, exist_ok=True)

            # Configure remote if provided
            if remote_url:
                self.add_remote("origin", remote_url, default=True)

            # Update instance state
            self.repo = dvc.repo.Repo(str(self.project_root))
            self.is_initialized = True
            self.available = True
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize DVC: {e}")
            return False

    def add_remote(self, name: str, url: str, default: bool = False) -> bool:
        """
        Add DVC remote storage configuration.
        
        Args:
            name: Remote name identifier
            url: Remote storage URL
            default: Whether to set as default remote
            
        Returns:
            True if remote added successfully, False otherwise
        """
        try:
            self._execute_dvc_command(["remote", "add", name, url])
            
            if default:
                self._execute_dvc_command(["remote", "default", name])
                
            print(f"Added DVC remote: {name} -> {url}")
            return True
            
        except Exception as e:
            print(f"Failed to add remote: {e}")
            return False

    # Dataset Tracking and Versioning

    def track_dataset(self, data_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Track dataset or file with DVC and inject optional metadata.
        
        Args:
            data_path: Path to data file or directory to track
            metadata: Optional metadata to embed in DVC file
            
        Returns:
            Path to generated DVC file if successful, None otherwise
        """
        if not self.is_initialized:
            print("DVC not initialized; skipping tracking")
            return None
            
        try:
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data path does not exist: {path}")
                
            # Track with DVC
            self._execute_dvc_command(["add", str(path)])
            dvc_file_path = f"{str(path)}.dvc"

            # Inject metadata if provided and YAML is available
            if metadata and YAML_AVAILABLE and Path(dvc_file_path).exists():
                self._inject_dataset_metadata(dvc_file_path, metadata)

            print(f"Dataset tracked: {path} -> {dvc_file_path}")
            return dvc_file_path
            
        except Exception as e:
            print(f"Failed to track dataset: {e}")
            return None

    def _inject_dataset_metadata(self, dvc_file: str, metadata: Dict[str, Any]) -> None:
        """
        Inject metadata into DVC file.
        
        Args:
            dvc_file: Path to DVC file
            metadata: Metadata dictionary to inject
        """
        try:
            # Read existing DVC file
            with open(dvc_file, "r", encoding="utf-8") as f:
                dvc_data = yaml.safe_load(f) or {}
                
            # Add metadata section
            dvc_data.setdefault("meta", {})
            dvc_data["meta"].update({
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
            })
            
            # Write updated DVC file
            with open(dvc_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(dvc_data, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Could not inject metadata into {dvc_file}: {e}")

    def version_dataset(
        self,
        experiment_id: str,
        dataset_name: str,
        data_paths: List[str],
        description: str = "",
        *,
        create_git_tag: bool = True,
        push_to_remote: bool = False,
    ) -> Dict[str, Any]:
        """
        Create comprehensive dataset version with manifest and tracking.
        
        Args:
            experiment_id: Unique experiment identifier
            dataset_name: Human-readable dataset name
            data_paths: List of paths to include in version
            description: Version description
            create_git_tag: Whether to create Git tag for version
            push_to_remote: Whether to push data to remote storage
            
        Returns:
            Dictionary containing version information and statistics
        """
        if not self.is_initialized:
            # Return minimal tracking data when DVC unavailable
            return self._create_fallback_version_info(
                experiment_id, dataset_name, data_paths, description
            )

        # Generate version identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"{dataset_name}_v{timestamp}"

        # Track all data paths and collect statistics
        tracked_files = []
        total_size = 0
        file_count = 0

        for path_str in data_paths:
            path = Path(path_str)
            if not path.exists():
                continue
                
            # Calculate statistics
            path_stats = self._calculate_path_statistics(path)
            total_size += path_stats["size"]
            file_count += path_stats["count"]
            
            # Track with DVC
            dvc_file = self.track_dataset(
                path_str,
                {
                    "experiment_id": experiment_id,
                    "dataset_name": dataset_name,
                    "version_tag": version_tag,
                    "description": description,
                },
            )
            
            if dvc_file:
                tracked_files.append(dvc_file)

        # Create version information
        version_info = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "version_tag": version_tag,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "data_paths": data_paths,
            "dvc_files": tracked_files,
            "total_size_bytes": int(total_size),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": int(file_count),
            "data_hash": self._calculate_dataset_hash(data_paths),
        }

        # Create and track version manifest
        self._create_version_manifest(version_tag, version_info)

        # Optional Git operations
        if create_git_tag:
            self._create_git_tag(version_tag, f"Dataset version: {dataset_name}")
            
        if push_to_remote:
            try:
                self.push_data()
            except Exception:
                pass

        print(f"Dataset version created: {version_tag} | files={file_count}, size={version_info['total_size_mb']} MB")
        return version_info

    def _create_fallback_version_info(self, experiment_id: str, dataset_name: str, 
                                    data_paths: List[str], description: str) -> Dict[str, Any]:
        """
        Create minimal version info when DVC is unavailable.
        
        Args:
            experiment_id: Experiment identifier
            dataset_name: Dataset name
            data_paths: Data paths list
            description: Version description
            
        Returns:
            Basic version information dictionary
        """
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

    def _calculate_path_statistics(self, path: Path) -> Dict[str, int]:
        """
        Calculate size and file count statistics for a path.
        
        Args:
            path: Path to analyze
            
        Returns:
            Dictionary with size and count statistics
        """
        stats = {"size": 0, "count": 0}
        
        try:
            if path.is_file():
                stats["size"] = path.stat().st_size
                stats["count"] = 1
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        stats["count"] += 1
                        try:
                            stats["size"] += file_path.stat().st_size
                        except Exception:
                            pass
        except Exception:
            pass
            
        return stats

    def _create_version_manifest(self, version_tag: str, version_info: Dict[str, Any]) -> None:
        """
        Create and track version manifest file.
        
        Args:
            version_tag: Version identifier
            version_info: Complete version information
        """
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self.data_dir / f"{version_tag}_manifest.json"
            
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(version_info, f, indent=2)
                
            self.track_dataset(str(manifest_path))
            
        except Exception as e:
            print(f"Could not create version manifest: {e}")

    # Dataset State and Fingerprinting

    def get_active_dataset_version(self, dataset_root: str = "data/") -> Dict[str, Any]:
        """
        Generate fingerprint of current dataset state.
        
        Creates a unique identifier for the current state of data based on
        DVC lock file or fallback hashing of DVC output files.
        
        Args:
            dataset_root: Root directory to analyze
            
        Returns:
            Dictionary containing dataset state information
        """
        root_path = Path(dataset_root)
        state_info: Dict[str, Any] = {"root": str(root_path)}

        # Get current Git commit
        git_commit = self._get_current_git_commit()
        state_info["git_commit"] = git_commit

        # Try to use DVC lock file for stable versioning
        lock_hash = self._extract_lock_file_hash()
        if lock_hash:
            state_info.update({
                "version_tag": f"lock-{lock_hash[:7]}",
                "lock_hash": lock_hash,
            })
        else:
            # Fallback to hashing DVC output files
            fallback_hash = self._calculate_dvc_outputs_hash(root_path)
            state_info.update({
                "version_tag": f"state-{fallback_hash[:7]}",
                "lock_hash": fallback_hash,
            })

        # Add size and count statistics
        stats = self._calculate_dataset_statistics(root_path)
        state_info.update(stats)

        return state_info

    def _get_current_git_commit(self) -> Optional[str]:
        """Get current Git commit SHA (short form)."""
        try:
            if self._is_git_repository():
                result = self._execute_git_command(["rev-parse", "--short", "HEAD"])
                return result.stdout.strip() if result else None
        except Exception:
            pass
        return None

    def _extract_lock_file_hash(self) -> Optional[str]:
        """Extract hash from DVC lock file."""
        if not YAML_AVAILABLE:
            return None
            
        lock_path = self.project_root / "dvc.lock"
        if not lock_path.exists():
            return None
            
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                lock_data = yaml.safe_load(f) or {}
                
            # Extract output hashes from all stages
            output_keys = []
            for stage_data in (lock_data.get("stages", {}) or {}).values():
                for output in (stage_data.get("outs", []) or []):
                    key = (output.get("md5") or output.get("etag") or 
                          output.get("hash") or output.get("path"))
                    if key:
                        output_keys.append(str(key))
                        
            # Generate combined hash
            if output_keys:
                hasher = hashlib.sha256()
                for key in sorted(output_keys):
                    hasher.update(key.encode("utf-8"))
                return hasher.hexdigest()
                
        except Exception as e:
            print(f"Could not read dvc.lock: {e}")
            
        return None

    def _calculate_dvc_outputs_hash(self, root_path: Path) -> str:
        """Calculate hash from DVC output files."""
        if not YAML_AVAILABLE:
            return "0" * 64
            
        try:
            # Find all DVC files in the root path
            search_path = root_path if root_path.is_absolute() else (self.project_root / root_path)
            dvc_files = list(search_path.rglob("*.dvc"))
            
            # Extract output keys from DVC files
            output_keys = []
            for dvc_file in dvc_files:
                try:
                    with open(dvc_file, "r", encoding="utf-8") as f:
                        dvc_data = yaml.safe_load(f) or {}
                        
                    for output in (dvc_data.get("outs", []) or []):
                        key = (output.get("md5") or output.get("etag") or 
                              output.get("hash") or output.get("path"))
                        if key:
                            output_keys.append(str(key))
                            
                except Exception:
                    continue
                    
            # Generate combined hash
            hasher = hashlib.sha256()
            for key in sorted(filter(None, output_keys)):
                hasher.update(key.encode("utf-8"))
                
            return hasher.hexdigest() if output_keys else "0" * 64
            
        except Exception:
            return "0" * 64

    def _calculate_dataset_statistics(self, root_path: Path) -> Dict[str, Any]:
        """Calculate file count and size statistics for dataset."""
        stats = {"file_count": 0, "total_size_mb": 0.0}
        
        try:
            search_path = root_path if root_path.is_absolute() else (self.project_root / root_path)
            if not search_path.exists():
                return stats
                
            total_size = 0
            file_count = 0
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    try:
                        total_size += file_path.stat().st_size
                    except Exception:
                        pass
                        
            stats["file_count"] = int(file_count)
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
        except Exception:
            pass
            
        return stats

    # Hash Calculation Methods

    def _calculate_dataset_hash(self, data_paths: List[str]) -> str:
        """
        Calculate content hash across multiple data paths.
        
        Creates a deterministic hash based on file contents and relative paths
        for dataset fingerprinting and change detection.
        
        Args:
            data_paths: List of data paths to include in hash
            
        Returns:
            SHA256 hash of dataset content
        """
        hasher = hashlib.sha256()
        
        try:
            for data_path in sorted(map(str, data_paths)):
                path = Path(data_path)
                if not path.exists():
                    continue
                    
                if path.is_file():
                    self._hash_single_file(hasher, path)
                else:
                    self._hash_directory_contents(hasher, path)
                    
        except Exception:
            pass
            
        return hasher.hexdigest()

    def _hash_single_file(self, hasher: hashlib.sha256, file_path: Path) -> None:
        """Hash a single file's name and contents."""
        # Include filename in hash
        hasher.update(str(file_path.name).encode("utf-8"))
        
        # Include file contents in hash
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except Exception:
            pass

    def _hash_directory_contents(self, hasher: hashlib.sha256, dir_path: Path) -> None:
        """Hash all files in a directory recursively."""
        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue
                
            # Include relative path in hash for structure consistency
            try:
                relative_path = str(file_path.relative_to(self.project_root))
            except Exception:
                relative_path = str(file_path)
                
            hasher.update(relative_path.encode("utf-8"))
            
            # Include file contents
            try:
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            except Exception:
                pass

    # Remote Storage Operations

    def push_data(self, remote_name: Optional[str] = None) -> bool:
        """
        Push tracked data to DVC remote storage.
        
        Args:
            remote_name: Optional specific remote to push to
            
        Returns:
            True if push successful, False otherwise
        """
        try:
            command = ["push"]
            if remote_name:
                command.extend(["-r", remote_name])
                
            self._execute_dvc_command(command)
            print("Data pushed to DVC remote")
            return True
            
        except Exception as e:
            print(f"Failed to push data: {e}")
            return False

    def pull_data(self, remote_name: Optional[str] = None) -> bool:
        """
        Pull tracked data from DVC remote storage.
        
        Args:
            remote_name: Optional specific remote to pull from
            
        Returns:
            True if pull successful, False otherwise
        """
        try:
            command = ["pull"]
            if remote_name:
                command.extend(["-r", remote_name])
                
            self._execute_dvc_command(command)
            print("Data pulled from DVC remote")
            return True
            
        except Exception as e:
            print(f"Failed to pull data: {e}")
            return False

    # Resource Management

    def close(self) -> None:
        """
        Close DVC repository and cleanup resources.
        
        Properly closes the DVC repository connection and releases
        any held resources for clean shutdown.
        """
        if self.repo:
            try:
                self.repo.close()
            except Exception:
                pass
        print("DVC backend cleaned up")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()