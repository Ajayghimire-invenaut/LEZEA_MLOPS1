"""
MLflow Backend for LeZeA MLOps
=============================

Handles all MLflow operations including:
- Experiment creation and management
- Run lifecycle management
- Metrics, parameters, and tags logging
- Artifact management
- Model registration and versioning

This backend provides a clean interface to MLflow while handling
connection management, error handling, and LeZeA-specific requirements.
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    import mlflow
    import mlflow.tracking
    from mlflow.entities import ViewType
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class MLflowBackend:
    """
    MLflow backend for experiment tracking and model management
    
    This class provides:
    - Clean interface to MLflow operations
    - Automatic connection management
    - Error handling and retry logic
    - LeZeA-specific experiment organization
    - Artifact management utilities
    """
    
    def __init__(self, config):
        """
        Initialize MLflow backend
        
        Args:
            config: Configuration object with MLflow settings
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError(
                "MLflow is not available. Install with: pip install mlflow"
            )
        
        self.config = config
        self.mlflow_config = config.get_mlflow_config()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        
        # Current experiment and run tracking
        self.current_experiment = None
        self.current_experiment_id = None
        self.current_run = None
        self.current_run_id = None
        
        # Connection verification
        self._verify_connection()
        
        print(f"✅ MLflow backend connected: {self.mlflow_config['tracking_uri']}")
    
    def _verify_connection(self):
        """Verify MLflow connection is working"""
        try:
            # Try to list experiments to test connection
            mlflow.search_experiments(max_results=1)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MLflow: {e}")
    
    def create_experiment(self, experiment_name: str, experiment_id: str, 
                         artifact_location: str = None, tags: Dict[str, str] = None) -> str:
        """
        Create or get existing MLflow experiment
        
        Args:
            experiment_name: Name of the experiment
            experiment_id: Unique experiment identifier
            artifact_location: Optional S3 or local path for artifacts
            tags: Optional experiment tags
        
        Returns:
            MLflow experiment ID
        """
        try:
            # Check if experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # Create new experiment
                mlflow_experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location or self.mlflow_config.get('default_artifact_root'),
                    tags=tags
                )
                print(f"📁 Created MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")
            else:
                mlflow_experiment_id = experiment.experiment_id
                print(f"📂 Using existing MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")
            
            # Set as active experiment
            mlflow.set_experiment(experiment_name)
            self.current_experiment = experiment_name
            self.current_experiment_id = mlflow_experiment_id
            
            # Add LeZeA-specific tags
            if experiment:
                self._add_experiment_tags(mlflow_experiment_id, experiment_id, tags)
            
            return mlflow_experiment_id
            
        except Exception as e:
            print(f"❌ Failed to create/get experiment: {e}")
            raise
    
    def _add_experiment_tags(self, mlflow_experiment_id: str, lezea_experiment_id: str, 
                           additional_tags: Dict[str, str] = None):
        """Add LeZeA-specific tags to experiment"""
        try:
            tags = {
                'lezea_experiment_id': lezea_experiment_id,
                'lezea_version': '1.0.0',
                'created_by': 'lezea_mlops',
                'creation_timestamp': datetime.now().isoformat()
            }
            
            if additional_tags:
                tags.update(additional_tags)
            
            # MLflow doesn't have direct experiment tag setting in some versions
            # We'll set these as run tags instead when runs are created
            self._pending_experiment_tags = tags
            
        except Exception as e:
            print(f"⚠️ Failed to add experiment tags: {e}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None, 
                  nested: bool = False) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to set on the run
            nested: Whether this is a nested run
        
        Returns:
            MLflow run ID
        """
        try:
            # Prepare run tags
            run_tags = {
                'mlflow.runName': run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'lezea.created_by': 'lezea_mlops',
                'lezea.start_time': datetime.now().isoformat()
            }
            
            # Add experiment tags if available
            if hasattr(self, '_pending_experiment_tags'):
                run_tags.update(self._pending_experiment_tags)
            
            # Add user-provided tags
            if tags:
                run_tags.update(tags)
            
            # Start the run
            run = mlflow.start_run(
                run_name=run_name,
                tags=run_tags,
                nested=nested
            )
            
            self.current_run = run
            self.current_run_id = run.info.run_id
            
            print(f"🏃 Started MLflow run: {run_name or 'unnamed'} (ID: {self.current_run_id[:8]}...)")
            
            return self.current_run_id
            
        except Exception as e:
            print(f"❌ Failed to start run: {e}")
            raise
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            if self.current_run:
                # Log end time
                self.log_metric("run_end_timestamp", datetime.now().timestamp())
                
                # End the run
                mlflow.end_run(status=status)
                print(f"🏁 Ended MLflow run with status: {status}")
                
                # Clear current run tracking
                self.current_run = None
                self.current_run_id = None
            else:
                print("⚠️ No active run to end")
                
        except Exception as e:
            print(f"❌ Failed to end run: {e}")
    
    def log_param(self, key: str, value: Any):
        """
        Log a single parameter
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        try:
            # Convert complex types to strings
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif value is None:
                value = "null"
            else:
                value = str(value)
            
            # MLflow has parameter length limits
            if len(value) > 500:
                value = value[:497] + "..."
            
            mlflow.log_param(key, value)
            
        except Exception as e:
            print(f"❌ Failed to log param {key}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Process parameters to handle MLflow limitations
            processed_params = {}
            
            for key, value in params.items():
                # Convert complex types to strings
                if isinstance(value, (dict, list)):
                    processed_value = json.dumps(value)
                elif value is None:
                    processed_value = "null"
                else:
                    processed_value = str(value)
                
                # Truncate if too long
                if len(processed_value) > 500:
                    processed_value = processed_value[:497] + "..."
                
                processed_params[key] = processed_value
            
            mlflow.log_params(processed_params)
            print(f"📝 Logged {len(params)} parameters")
            
        except Exception as e:
            print(f"❌ Failed to log params: {e}")
    
    def log_metric(self, key: str, value: Union[int, float], step: int = None):
        """
        Log a single metric
        
        Args:
            key: Metric name
            value: Metric value (must be numeric)
            step: Optional step number
        """
        try:
            # Ensure value is numeric
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    print(f"⚠️ Skipping non-numeric metric {key}: {value}")
                    return
            
            mlflow.log_metric(key, value, step=step)
            
        except Exception as e:
            print(f"❌ Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: int = None):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for all metrics
        """
        try:
            # Filter to only numeric values
            numeric_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = value
                else:
                    try:
                        numeric_metrics[key] = float(value)
                    except (ValueError, TypeError):
                        print(f"⚠️ Skipping non-numeric metric {key}: {value}")
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                
        except Exception as e:
            print(f"❌ Failed to log metrics: {e}")
    
    def set_tag(self, key: str, value: str):
        """
        Set a single tag
        
        Args:
            key: Tag name
            value: Tag value
        """
        try:
            mlflow.set_tag(key, str(value))
        except Exception as e:
            print(f"❌ Failed to set tag {key}: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set multiple tags
        
        Args:
            tags: Dictionary of tags to set
        """
        try:
            # Convert all values to strings
            processed_tags = {k: str(v) for k, v in tags.items()}
            mlflow.set_tags(processed_tags)
            print(f"🏷️ Set {len(tags)} tags")
            
        except Exception as e:
            print(f"❌ Failed to set tags: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact (file)
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifacts directory
        """
        try:
            if not os.path.exists(local_path):
                print(f"⚠️ Artifact file not found: {local_path}")
                return
            
            mlflow.log_artifact(local_path, artifact_path)
            print(f"📎 Logged artifact: {os.path.basename(local_path)}")
            
        except Exception as e:
            print(f"❌ Failed to log artifact: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        Log multiple artifacts from a directory
        
        Args:
            local_dir: Path to local directory
            artifact_path: Optional path within artifacts directory
        """
        try:
            if not os.path.exists(local_dir):
                print(f"⚠️ Artifact directory not found: {local_dir}")
                return
            
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"📁 Logged artifacts from: {local_dir}")
            
        except Exception as e:
            print(f"❌ Failed to log artifacts: {e}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as a JSON artifact
        
        Args:
            dictionary: Dictionary to save
            filename: Name of the JSON file
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dictionary, f, indent=2, default=str)
                temp_path = f.name
            
            # Log as artifact
            mlflow.log_artifact(temp_path, filename)
            
            # Clean up
            os.unlink(temp_path)
            
            print(f"📄 Logged dictionary as: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to log dictionary: {e}")
    
    def log_text(self, text: str, filename: str):
        """
        Log text content as an artifact
        
        Args:
            text: Text content to save
            filename: Name of the text file
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_path = f.name
            
            # Log as artifact
            mlflow.log_artifact(temp_path, filename)
            
            # Clean up
            os.unlink(temp_path)
            
            print(f"📝 Logged text as: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to log text: {e}")
    
    def register_model(self, model_name: str, model_path: str = None, 
                      description: str = None, tags: Dict[str, str] = None) -> str:
        """
        Register a model in MLflow Model Registry
        
        Args:
            model_name: Name for the registered model
            model_path: Path to model artifacts (if None, uses current run)
            description: Model description
            tags: Model tags
        
        Returns:
            Model version
        """
        try:
            if model_path is None:
                model_uri = f"runs:/{self.current_run_id}/model"
            else:
                model_uri = model_path
            
            result = mlflow.register_model(
                model_uri, 
                model_name,
                tags=tags
            )
            
            model_version = result.version
            
            # Add description if provided
            if description:
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.update_model_version(
                        name=model_name,
                        version=model_version,
                        description=description
                    )
                except Exception as e:
                    print(f"⚠️ Failed to add model description: {e}")
            
            print(f"🏆 Registered model: {model_name} v{model_version}")
            return model_version
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return None
    
    def search_runs(self, experiment_ids: List[str] = None, filter_string: str = "",
                   order_by: List[str] = None, max_results: int = 1000) -> List[Dict]:
        """
        Search for runs in MLflow
        
        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: Filter string for search
            order_by: List of order criteria
            max_results: Maximum number of results
        
        Returns:
            List of run information dictionaries
        """
        try:
            if experiment_ids is None and self.current_experiment_id:
                experiment_ids = [self.current_experiment_id]
            
            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
                output_format="list"
            )
            
            return runs
            
        except Exception as e:
            print(f"❌ Failed to search runs: {e}")
            return []
    
    def get_run_info(self, run_id: str = None) -> Dict:
        """
        Get information about a specific run
        
        Args:
            run_id: Run ID (if None, uses current run)
        
        Returns:
            Dictionary with run information
        """
        try:
            target_run_id = run_id or self.current_run_id
            if not target_run_id:
                return {}
            
            run = mlflow.get_run(target_run_id)
            
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags),
                'artifact_uri': run.info.artifact_uri
            }
            
        except Exception as e:
            print(f"❌ Failed to get run info: {e}")
            return {}
    
    def get_experiment_url(self) -> str:
        """Get the MLflow UI URL for the current experiment"""
        if self.current_experiment_id:
            base_url = self.mlflow_config['tracking_uri'].rstrip('/')
            return f"{base_url}/#/experiments/{self.current_experiment_id}"
        return self.mlflow_config['tracking_uri']
    
    def get_run_url(self) -> str:
        """Get the MLflow UI URL for the current run"""
        if self.current_run_id and self.current_experiment_id:
            base_url = self.mlflow_config['tracking_uri'].rstrip('/')
            return f"{base_url}/#/experiments/{self.current_experiment_id}/runs/{self.current_run_id}"
        return self.get_experiment_url()
    
    def delete_run(self, run_id: str):
        """
        Delete a run
        
        Args:
            run_id: ID of the run to delete
        """
        try:
            mlflow.delete_run(run_id)
            print(f"🗑️ Deleted run: {run_id}")
        except Exception as e:
            print(f"❌ Failed to delete run: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current experiment"""
        try:
            if not self.current_experiment_id:
                return {}
            
            # Get all runs for the experiment
            runs = self.search_runs()
            
            if not runs:
                return {'experiment_id': self.current_experiment_id, 'run_count': 0}
            
            # Calculate summary statistics
            summary = {
                'experiment_id': self.current_experiment_id,
                'experiment_name': self.current_experiment,
                'run_count': len(runs),
                'completed_runs': len([r for r in runs if r.get('status') == 'FINISHED']),
                'failed_runs': len([r for r in runs if r.get('status') == 'FAILED']),
                'running_runs': len([r for r in runs if r.get('status') == 'RUNNING']),
                'total_duration': sum([(r.get('end_time', 0) or 0) - (r.get('start_time', 0) or 0) 
                                     for r in runs if r.get('start_time') and r.get('end_time')]),
                'mlflow_ui_url': self.get_experiment_url()
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Failed to get experiment summary: {e}")
            return {}
    
    def cleanup(self):
        """Clean up any active runs and connections"""
        try:
            if self.current_run:
                self.end_run("KILLED")
            print("🧹 Cleaned up MLflow backend")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()