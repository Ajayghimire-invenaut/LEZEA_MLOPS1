"""
DVC Backend for LeZeA MLOps
==========================

Handles dataset versioning and data pipeline management using DVC:
- Dataset version control with automatic tagging
- Data lineage tracking and provenance
- Data pipeline management and execution
- Remote storage synchronization (S3, GCS, etc.)
- Data integrity verification
- Reproducible data processing workflows

This backend provides a comprehensive interface to DVC while handling
Git integration, remote storage, and LeZeA-specific data requirements.
"""

import os
import json
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import dvc.api
    import dvc.repo
    from dvc.exceptions import DvcException
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    dvc = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class DVCBackend:
    """
    DVC backend for dataset versioning and data pipeline management
    
    This class provides:
    - Automated dataset versioning with metadata
    - Data lineage tracking and provenance
    - Pipeline management for reproducible workflows
    - Remote storage synchronization
    - Data integrity verification
    - Integration with Git for complete reproducibility
    """
    
    def __init__(self, config):
        """
        Initialize DVC backend
        
        Args:
            config: Configuration object with DVC settings
        """
        if not DVC_AVAILABLE:
            print("⚠️ DVC is not available. Install with: pip install dvc[s3]")
            self.available = False
            return
        
        if not YAML_AVAILABLE:
            print("⚠️ PyYAML is not available. Install with: pip install PyYAML")
            self.available = False
            return
        
        self.config = config
        self.project_root = Path.cwd()
        self.dvc_dir = self.project_root / '.dvc'
        self.data_dir = self.project_root / 'data'
        
        # Initialize DVC repository
        self.repo = None
        self.is_initialized = self._check_dvc_initialized()
        
        if self.is_initialized:
            try:
                self.repo = dvc.repo.Repo(str(self.project_root))
                self.available = True
                print("✅ DVC backend ready")
            except Exception as e:
                print(f"⚠️ DVC repo initialization failed: {e}")
                self.available = False
        else:
            self.available = False
            print("⚠️ DVC not initialized. Run dvc init first.")
    
    def _check_dvc_initialized(self) -> bool:
        """Check if DVC is initialized in the project"""
        return self.dvc_dir.exists() and (self.dvc_dir / 'config').exists()
    
    def _run_dvc_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a DVC command and return the result"""
        try:
            result = subprocess.run(
                ['dvc'] + command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ DVC command failed: {' '.join(command)}")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("DVC is not installed or not in PATH")
    
    def initialize_dvc(self, remote_url: str = None) -> bool:
        """
        Initialize DVC in the project
        
        Args:
            remote_url: Optional remote storage URL (S3, GCS, etc.)
        
        Returns:
            True if successful
        """
        try:
            if self.is_initialized:
                print("✅ DVC already initialized")
                return True
            
            # Initialize DVC
            self._run_dvc_command(['init'])
            print("✅ DVC initialized")
            
            # Add remote storage if provided
            if remote_url:
                self.add_remote('origin', remote_url, default=True)
            
            # Create data directories
            self.data_dir.mkdir(exist_ok=True)
            (self.data_dir / 'raw').mkdir(exist_ok=True)
            (self.data_dir / 'processed').mkdir(exist_ok=True)
            (self.data_dir / 'external').mkdir(exist_ok=True)
            
            # Initialize repo object
            self.repo = dvc.repo.Repo(str(self.project_root))
            self.is_initialized = True
            self.available = True
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize DVC: {e}")
            return False
    
    def add_remote(self, name: str, url: str, default: bool = False) -> bool:
        """
        Add remote storage to DVC
        
        Args:
            name: Remote name
            url: Remote URL (s3://, gs://, etc.)
            default: Set as default remote
        
        Returns:
            True if successful
        """
        try:
            # Add remote
            self._run_dvc_command(['remote', 'add', name, url])
            
            # Set as default if requested
            if default:
                self._run_dvc_command(['remote', 'default', name])
            
            print(f"✅ Added DVC remote: {name} -> {url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add remote: {e}")
            return False
    
    def track_dataset(self, data_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Track a dataset with DVC
        
        Args:
            data_path: Path to dataset file or directory
            metadata: Additional metadata about the dataset
        
        Returns:
            DVC file path (.dvc file)
        """
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
            # Add to DVC tracking
            self._run_dvc_command(['add', str(data_path)])
            dvc_file = f"{data_path}.dvc"
            
            # Add metadata if provided
            if metadata:
                self._add_dataset_metadata(dvc_file, metadata)
            
            print(f"📁 Dataset tracked: {data_path} -> {dvc_file}")
            return dvc_file
            
        except Exception as e:
            print(f"❌ Failed to track dataset: {e}")
            return None
    
    def _add_dataset_metadata(self, dvc_file: str, metadata: Dict[str, Any]):
        """Add metadata to DVC file"""
        try:
            # Read existing DVC file
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
            
            # Add metadata section
            dvc_data['meta'] = {
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            # Write back to file
            with open(dvc_file, 'w') as f:
                yaml.safe_dump(dvc_data, f, default_flow_style=False)
                
        except Exception as e:
            print(f"⚠️ Could not add metadata to DVC file: {e}")
    
    def version_dataset(self, experiment_id: str, dataset_name: str,
                       data_paths: List[str], description: str = "") -> Dict[str, Any]:
        """
        Create a versioned dataset for an experiment
        
        Args:
            experiment_id: Experiment identifier
            dataset_name: Name of the dataset
            data_paths: List of data file/directory paths
            description: Description of the dataset version
        
        Returns:
            Dataset version information
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_tag = f"{dataset_name}_v{timestamp}"
            
            # Track all data paths
            dvc_files = []
            total_size = 0
            file_count = 0
            
            for data_path in data_paths:
                path = Path(data_path)
                if path.exists():
                    # Calculate size
                    if path.is_file():
                        total_size += path.stat().st_size
                        file_count += 1
                    elif path.is_dir():
                        for file_path in path.rglob('*'):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                                file_count += 1
                    
                    # Track with DVC
                    dvc_file = self.track_dataset(data_path, {
                        'experiment_id': experiment_id,
                        'dataset_name': dataset_name,
                        'version_tag': version_tag,
                        'description': description
                    })
                    
                    if dvc_file:
                        dvc_files.append(dvc_file)
            
            # Create version manifest
            version_info = {
                'experiment_id': experiment_id,
                'dataset_name': dataset_name,
                'version_tag': version_tag,
                'timestamp': datetime.now().isoformat(),
                'description': description,
                'data_paths': data_paths,
                'dvc_files': dvc_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024*1024), 2),
                'file_count': file_count,
                'data_hash': self._calculate_dataset_hash(data_paths)
            }
            
            # Save manifest
            manifest_path = self.data_dir / f"{version_tag}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # Track manifest with DVC
            self.track_dataset(str(manifest_path))
            
            # Create Git tag for this version
            self._create_git_tag(version_tag, f"Dataset version: {dataset_name}")
            
            print(f"📦 Dataset version created: {version_tag}")
            print(f"   Files: {file_count}, Size: {version_info['total_size_mb']} MB")
            
            return version_info
            
        except Exception as e:
            print(f"❌ Failed to version dataset: {e}")
            return {}
    
    def _calculate_dataset_hash(self, data_paths: List[str]) -> str:
        """Calculate hash of dataset contents"""
        hasher = hashlib.sha256()
        
        for data_path in sorted(data_paths):
            path = Path(data_path)
            if path.exists():
                if path.is_file():
                    with open(path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                elif path.is_dir():
                    for file_path in sorted(path.rglob('*')):
                        if file_path.is_file():
                            hasher.update(str(file_path).encode())
                            with open(file_path, 'rb') as f:
                                for chunk in iter(lambda: f.read(4096), b""):
                                    hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _create_git_tag(self, tag_name: str, message: str):
        """Create a Git tag for the dataset version"""
        try:
            subprocess.run(['git', 'tag', '-a', tag_name, '-m', message], 
                         cwd=self.project_root, check=True, capture_output=True)
            print(f"🏷️ Created Git tag: {tag_name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Could not create Git tag: {e}")
        except FileNotFoundError:
            print("⚠️ Git not available for tagging")
    
    def push_data(self, remote_name: str = None) -> bool:
        """
        Push data to remote storage
        
        Args:
            remote_name: Name of remote (uses default if None)
        
        Returns:
            True if successful
        """
        try:
            command = ['push']
            if remote_name:
                command.extend(['-r', remote_name])
            
            self._run_dvc_command(command)
            print("📤 Data pushed to remote storage")
            return True
            
        except Exception as e:
            print(f"❌ Failed to push data: {e}")
            return False
    
    def pull_data(self, remote_name: str = None) -> bool:
        """
        Pull data from remote storage
        
        Args:
            remote_name: Name of remote (uses default if None)
        
        Returns:
            True if successful
        """
        try:
            command = ['pull']
            if remote_name:
                command.extend(['-r', remote_name])
            
            self._run_dvc_command(command)
            print("📥 Data pulled from remote storage")
            return True
            
        except Exception as e:
            print(f"❌ Failed to pull data: {e}")
            return False
    
    def get_dataset_status(self) -> Dict[str, Any]:
        """Get status of tracked datasets"""
        try:
            result = self._run_dvc_command(['status'])
            
            # Parse DVC status output
            status_info = {
                'timestamp': datetime.now().isoformat(),
                'status_output': result.stdout,
                'is_clean': 'Data and pipelines are up to date' in result.stdout or result.stdout.strip() == '',
                'tracked_files': []
            }
            
            # Find all .dvc files
            for dvc_file in self.project_root.rglob('*.dvc'):
                status_info['tracked_files'].append({
                    'dvc_file': str(dvc_file.relative_to(self.project_root)),
                    'data_file': str(dvc_file.with_suffix('').relative_to(self.project_root))
                })
            
            return status_info
            
        except Exception as e:
            print(f"❌ Failed to get dataset status: {e}")
            return {}
    
    def list_dataset_versions(self, dataset_name: str = None) -> List[Dict[str, Any]]:
        """
        List all dataset versions
        
        Args:
            dataset_name: Filter by dataset name (optional)
        
        Returns:
            List of dataset version information
        """
        try:
            versions = []
            
            # Find all manifest files
            pattern = f"{dataset_name}_v*_manifest.json" if dataset_name else "*_manifest.json"
            for manifest_file in self.data_dir.glob(pattern):
                try:
                    with open(manifest_file, 'r') as f:
                        version_info = json.load(f)
                    versions.append(version_info)
                except Exception as e:
                    print(f"⚠️ Could not read manifest {manifest_file}: {e}")
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return versions
            
        except Exception as e:
            print(f"❌ Failed to list dataset versions: {e}")
            return []
    
    def checkout_dataset_version(self, version_tag: str) -> bool:
        """
        Checkout a specific dataset version
        
        Args:
            version_tag: Version tag to checkout
        
        Returns:
            True if successful
        """
        try:
            # Find manifest file
            manifest_file = self.data_dir / f"{version_tag}_manifest.json"
            if not manifest_file.exists():
                print(f"❌ Version manifest not found: {version_tag}")
                return False
            
            # Load version info
            with open(manifest_file, 'r') as f:
                version_info = json.load(f)
            
            # Checkout each DVC file
            for dvc_file in version_info.get('dvc_files', []):
                if Path(dvc_file).exists():
                    self._run_dvc_command(['checkout', dvc_file])
            
            # Checkout Git tag if exists
            try:
                subprocess.run(['git', 'checkout', version_tag], 
                             cwd=self.project_root, check=True, capture_output=True)
                print(f"🏷️ Checked out Git tag: {version_tag}")
            except subprocess.CalledProcessError:
                print(f"⚠️ Git tag {version_tag} not found, continuing with DVC checkout only")
            
            print(f"✅ Checked out dataset version: {version_tag}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to checkout dataset version: {e}")
            return False
    
    def create_data_pipeline(self, pipeline_name: str, stages: List[Dict[str, Any]]) -> str:
        """
        Create a DVC pipeline for data processing
        
        Args:
            pipeline_name: Name of the pipeline
            stages: List of pipeline stages
        
        Returns:
            Path to created pipeline file
        """
        try:
            pipeline_file = self.project_root / f"dvc_{pipeline_name}.yaml"
            
            # Create DVC pipeline YAML
            pipeline_config = {
                'stages': {}
            }
            
            for stage in stages:
                stage_name = stage['name']
                pipeline_config['stages'][stage_name] = {
                    'cmd': stage['command'],
                    'deps': stage.get('dependencies', []),
                    'outs': stage.get('outputs', [])
                }
                
                if 'parameters' in stage:
                    pipeline_config['stages'][stage_name]['params'] = stage['parameters']
                
                if 'metrics' in stage:
                    pipeline_config['stages'][stage_name]['metrics'] = stage['metrics']
                
                if 'plots' in stage:
                    pipeline_config['stages'][stage_name]['plots'] = stage['plots']
            
            # Write pipeline file
            with open(pipeline_file, 'w') as f:
                yaml.safe_dump(pipeline_config, f, default_flow_style=False)
            
            print(f"📋 Created DVC pipeline: {pipeline_file}")
            return str(pipeline_file)
            
        except Exception as e:
            print(f"❌ Failed to create pipeline: {e}")
            return None
    
    def run_pipeline(self, pipeline_file: str = None, stage: str = None) -> bool:
        """
        Run DVC pipeline
        
        Args:
            pipeline_file: Path to pipeline file (optional)
            stage: Specific stage to run (optional)
        
        Returns:
            True if successful
        """
        try:
            command = ['repro']
            
            if stage:
                command.extend(['-s', stage])
            
            if pipeline_file:
                command.append(pipeline_file)
            
            self._run_dvc_command(command, capture_output=False)
            print("🔄 DVC pipeline completed")
            return True
            
        except Exception as e:
            print(f"❌ Failed to run pipeline: {e}")
            return False
    
    def get_data_lineage(self, data_path: str) -> Dict[str, Any]:
        """
        Get lineage information for a dataset
        
        Args:
            data_path: Path to data file/directory
        
        Returns:
            Lineage information
        """
        try:
            # Get DVC file info
            dvc_file = f"{data_path}.dvc"
            if not Path(dvc_file).exists():
                return {'error': 'Data not tracked by DVC'}
            
            # Read DVC file
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
            
            # Get file metadata
            lineage_info = {
                'data_path': data_path,
                'dvc_file': dvc_file,
                'dvc_data': dvc_data,
                'tracked_since': datetime.fromtimestamp(Path(dvc_file).stat().st_ctime).isoformat(),
                'last_modified': datetime.fromtimestamp(Path(dvc_file).stat().st_mtime).isoformat(),
                'metadata': dvc_data.get('meta', {})
            }
            
            # Try to get Git history of DVC file
            try:
                git_log = subprocess.run(
                    ['git', 'log', '--oneline', dvc_file],
                    capture_output=True, text=True, cwd=self.project_root
                )
                if git_log.returncode == 0:
                    lineage_info['git_history'] = git_log.stdout.strip().split('\n')
            except:
                pass
            
            # Get pipeline dependencies if any
            try:
                result = self._run_dvc_command(['dag', dvc_file])
                lineage_info['pipeline_dependencies'] = result.stdout
            except:
                lineage_info['pipeline_dependencies'] = None
            
            return lineage_info
            
        except Exception as e:
            print(f"❌ Failed to get data lineage: {e}")
            return {'error': str(e)}
    
    def compare_dataset_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two dataset versions
        
        Args:
            version1: First version tag
            version2: Second version tag
        
        Returns:
            Comparison results
        """
        try:
            # Load both version manifests
            manifest1_path = self.data_dir / f"{version1}_manifest.json"
            manifest2_path = self.data_dir / f"{version2}_manifest.json"
            
            if not manifest1_path.exists():
                return {'error': f'Version {version1} not found'}
            if not manifest2_path.exists():
                return {'error': f'Version {version2} not found'}
            
            with open(manifest1_path, 'r') as f:
                info1 = json.load(f)
            with open(manifest2_path, 'r') as f:
                info2 = json.load(f)
            
            comparison = {
                'version1': version1,
                'version2': version2,
                'timestamp1': info1.get('timestamp'),
                'timestamp2': info2.get('timestamp'),
                'size_diff_mb': info2.get('total_size_mb', 0) - info1.get('total_size_mb', 0),
                'file_count_diff': info2.get('file_count', 0) - info1.get('file_count', 0),
                'hash_changed': info1.get('data_hash') != info2.get('data_hash'),
                'paths_added': list(set(info2.get('data_paths', [])) - set(info1.get('data_paths', []))),
                'paths_removed': list(set(info1.get('data_paths', [])) - set(info2.get('data_paths', [])))
            }
            
            return comparison
            
        except Exception as e:
            print(f"❌ Failed to compare dataset versions: {e}")
            return {'error': str(e)}
    
    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        """
        Clean up old dataset versions
        
        Args:
            keep_versions: Number of versions to keep per dataset
        
        Returns:
            Number of versions cleaned up
        """
        try:
            # Group versions by dataset name
            datasets = {}
            for manifest_file in self.data_dir.glob('*_manifest.json'):
                try:
                    with open(manifest_file, 'r') as f:
                        version_info = json.load(f)
                    
                    dataset_name = version_info.get('dataset_name', 'unknown')
                    if dataset_name not in datasets:
                        datasets[dataset_name] = []
                    
                    datasets[dataset_name].append({
                        'manifest_file': manifest_file,
                        'version_info': version_info
                    })
                except:
                    continue
            
            cleaned_count = 0
            
            # Clean up old versions for each dataset
            for dataset_name, versions in datasets.items():
                # Sort by timestamp (newest first)
                versions.sort(key=lambda x: x['version_info'].get('timestamp', ''), reverse=True)
                
                # Remove old versions
                for version in versions[keep_versions:]:
                    try:
                        # Remove manifest file
                        version['manifest_file'].unlink()
                        
                        # Remove DVC files if they exist
                        for dvc_file in version['version_info'].get('dvc_files', []):
                            dvc_path = Path(dvc_file)
                            if dvc_path.exists():
                                dvc_path.unlink()
                        
                        cleaned_count += 1
                        print(f"🗑️ Removed old version: {version['version_info'].get('version_tag')}")
                        
                    except Exception as e:
                        print(f"⚠️ Could not remove version: {e}")
            
            print(f"🧹 Cleaned up {cleaned_count} old dataset versions")
            return cleaned_count
            
        except Exception as e:
            print(f"❌ Failed to cleanup old versions: {e}")
            return 0
    
    def export_dataset_info(self, output_file: str):
        """
        Export comprehensive dataset information
        
        Args:
            output_file: Path to save dataset information
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'project_root': str(self.project_root),
                'dvc_initialized': self.is_initialized,
                'dataset_versions': self.list_dataset_versions(),
                'dataset_status': self.get_dataset_status(),
                'tracked_files': []
            }
            
            # Add tracked file information
            for dvc_file in self.project_root.rglob('*.dvc'):
                try:
                    with open(dvc_file, 'r') as f:
                        dvc_data = yaml.safe_load(f)
                    
                    export_data['tracked_files'].append({
                        'dvc_file': str(dvc_file.relative_to(self.project_root)),
                        'data_file': str(dvc_file.with_suffix('').relative_to(self.project_root)),
                        'dvc_data': dvc_data
                    })
                except Exception as e:
                    print(f"⚠️ Could not read DVC file {dvc_file}: {e}")
            
            # Save export
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"📁 Exported dataset info to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to export dataset info: {e}")
    
    def get_remote_info(self) -> Dict[str, Any]:
        """Get information about configured remotes"""
        try:
            result = self._run_dvc_command(['remote', 'list'])
            
            remotes = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        remotes[parts[0]] = parts[1]
            
            # Get default remote
            try:
                default_result = self._run_dvc_command(['remote', 'default'])
                default_remote = default_result.stdout.strip()
            except:
                default_remote = None
            
            return {
                'remotes': remotes,
                'default_remote': default_remote,
                'remote_count': len(remotes)
            }
            
        except Exception as e:
            print(f"❌ Failed to get remote info: {e}")
            return {}
    
    def validate_data_integrity(self, data_path: str = None) -> Dict[str, Any]:
        """
        Validate data integrity for tracked files
        
        Args:
            data_path: Specific data path to validate (None for all)
        
        Returns:
            Validation results
        """
        try:
            if data_path:
                command = ['status', data_path]
            else:
                command = ['status']
            
            result = self._run_dvc_command(command)
            
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'is_valid': 'Data and pipelines are up to date' in result.stdout or result.stdout.strip() == '',
                'status_output': result.stdout,
                'issues': []
            }
            
            # Parse status output for issues
            if result.stdout.strip() and 'up to date' not in result.stdout:
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('Data and pipelines'):
                        validation_results['issues'].append(line)
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Failed to validate data integrity: {e}")
            return {'is_valid': False, 'error': str(e)}
    
    def close(self):
        """Clean up DVC backend resources"""
        if self.repo:
            try:
                self.repo.close()
            except:
                pass
        print("🧹 DVC backend cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()