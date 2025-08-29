"""
MLflow Backend for LeZeA Experiment Tracking System


"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MLflow dependencies with graceful fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False


def _stringify(value: Any, max_length: int = 500) -> str:
    """
    Convert any value to a string representation suitable for MLflow.
    
    Handles complex data types by converting to JSON and applies length
    truncation to prevent oversized parameter/tag values.
    
    Args:
        value: The value to convert to string
        max_length: Maximum allowed string length before truncation
        
    Returns:
        String representation of the value, truncated if necessary
    """
    if isinstance(value, (dict, list)):
        serialized = json.dumps(value)
    elif value is None:
        serialized = "null"
    else:
        serialized = str(value)
        
    # Apply length limit with ellipsis indication
    if len(serialized) <= max_length:
        return serialized
    else:
        return serialized[:max_length - 3] + "..."


class MLflowBackend:
    """
    Professional MLflow backend for experiment tracking and model management.
    
    This class provides a comprehensive interface for MLflow operations with
    enhanced error handling, logging, and production-ready features.
    
    Features:
        - Robust connection management with early validation
        - Idempotent experiment and run management
        - Comprehensive logging capabilities (params, metrics, artifacts)
        - Model registry integration
        - Context managers for nested runs
        - URL generation for web UI navigation
        - Graceful error handling and recovery
        
    Attributes:
        config: Configuration object containing MLflow settings
        current_experiment: Name of the currently active experiment
        current_experiment_id: MLflow ID of the current experiment
        current_run: Active MLflow run object
        current_run_id: ID of the currently active run
        available: Boolean indicating backend availability
    """
    
    def __init__(self, config) -> None:
        """
        Initialize MLflow backend with configuration and connection validation.
        
        Args:
            config: Configuration object with get_mlflow_config() method
            
        Raises:
            RuntimeError: If MLflow is not available
            ConnectionError: If connection to MLflow tracking server fails
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError(
                "MLflow is not available. Install with: pip install mlflow"
            )

        self.config = config
        self.mlflow_config = config.get_mlflow_config()

        # Configure MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])

        # Initialize state tracking
        self.current_experiment: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        self.current_run = None
        self.current_run_id: Optional[str] = None
        self._pending_experiment_tags: Dict[str, str] = {}
        self.available: bool = False

        # Verify connection and mark as available
        self._verify_connection()
        self.available = True
        
        print(f"MLflow backend connected: {self.mlflow_config['tracking_uri']}")

    # Connection and Experiment Management
    
    def _verify_connection(self) -> None:
        """
        Verify connectivity to MLflow tracking server.
        
        Performs a lightweight operation to ensure the tracking server
        is reachable and responding correctly.
        
        Raises:
            ConnectionError: If connection verification fails
        """
        try:
            # Attempt to search experiments as a connectivity test
            _ = mlflow.search_experiments(max_results=1)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MLflow: {e}")

    def ping(self) -> bool:
        """
        Test connectivity to MLflow tracking server.
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            _ = mlflow.search_experiments(max_results=1)
            return True
        except Exception:
            return False

    def create_experiment(
        self,
        experiment_name: str,
        lezea_experiment_id: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create or retrieve an MLflow experiment and set it as current.
        
        This method is idempotent - if an experiment with the given name
        already exists, it will be reused rather than creating a duplicate.
        
        Args:
            experiment_name: Human-readable name for the experiment
            lezea_experiment_id: LeZeA system experiment identifier
            artifact_location: Optional custom artifact storage location
            tags: Optional experiment-level tags
            
        Returns:
            MLflow experiment ID
            
        Raises:
            Exception: If experiment creation or retrieval fails
        """
        try:
            # Check for existing experiment
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if existing_experiment is None:
                # Create new experiment
                processed_tags = None
                if tags:
                    processed_tags = {k: _stringify(v) for k, v in tags.items()}
                    
                artifact_loc = (artifact_location or 
                               self.mlflow_config.get("default_artifact_root"))
                
                mlflow_experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_loc,
                    tags=processed_tags,
                )
                print(f"Created MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")
            else:
                # Use existing experiment
                mlflow_experiment_id = existing_experiment.experiment_id
                print(f"Using existing MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")

            # Set as current experiment
            mlflow.set_experiment(experiment_name)
            self.current_experiment = experiment_name
            self.current_experiment_id = mlflow_experiment_id

            # Apply experiment-level tags
            self._add_experiment_tags(mlflow_experiment_id, lezea_experiment_id, tags)
            
            return mlflow_experiment_id
            
        except Exception as e:
            print(f"Failed to create/get experiment: {e}")
            raise

    def _add_experiment_tags(
        self, 
        mlflow_experiment_id: str, 
        lezea_experiment_id: str, 
        additional_tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add standardized tags to experiment and cache for run inheritance.
        
        Args:
            mlflow_experiment_id: MLflow experiment ID
            lezea_experiment_id: LeZeA experiment ID
            additional_tags: Optional additional tags to apply
        """
        try:
            # Create base tag set with system metadata
            base_tags = {
                "lezea_experiment_id": lezea_experiment_id,
                "lezea_version": "1.0.0",
                "created_by": "lezea_mlops",
                "creation_timestamp": datetime.now().isoformat(),
            }
            
            # Merge additional tags if provided
            if additional_tags:
                base_tags.update({str(k): _stringify(v) for k, v in additional_tags.items()})
            
            # Cache tags for run inheritance
            self._pending_experiment_tags.update(base_tags)

            # Attempt to set experiment-level tags (best effort)
            try:
                client = MlflowClient()
                for key, value in base_tags.items():
                    client.set_experiment_tag(mlflow_experiment_id, key, str(value))
            except Exception:
                # Experiment tags may not be supported in all MLflow versions
                pass
                
        except Exception as e:
            print(f"Failed to add experiment tags: {e}")

    # Run Lifecycle Management
    
    def start_run(
        self, 
        run_name: Optional[str] = None, 
        tags: Optional[Dict[str, str]] = None, 
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run with proper lifecycle management.
        
        Automatically ends any existing active run to prevent stacking.
        Applies experiment-level tags and system metadata to the new run.
        
        Args:
            run_name: Optional human-readable name for the run
            tags: Optional run-specific tags
            nested: Whether this is a nested child run
            
        Returns:
            MLflow run ID
            
        Raises:
            Exception: If run creation fails
        """
        try:
            # End any existing active run to prevent conflicts
            if self.current_run is not None:
                print("Ending previous active run before starting new one")
                self.end_run(status="FINISHED")

            # Prepare run tags with system metadata
            run_tags = {
                "mlflow.runName": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "lezea.created_by": "lezea_mlops",
                "lezea.start_time": datetime.now().isoformat(),
            }
            
            # Inherit experiment-level tags
            if self._pending_experiment_tags:
                run_tags.update(self._pending_experiment_tags)
                
            # Apply custom tags
            if tags:
                run_tags.update({str(k): _stringify(v) for k, v in tags.items()})

            # Start the MLflow run
            run = mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested)
            self.current_run = run
            self.current_run_id = run.info.run_id
            
            run_display_name = run_name or 'unnamed'
            run_id_short = self.current_run_id[:8] if self.current_run_id else 'unknown'
            print(f"Started MLflow run: {run_display_name} (ID: {run_id_short}...)")
            
            return self.current_run_id
            
        except Exception as e:
            print(f"Failed to start run: {e}")
            raise

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the currently active MLflow run with proper cleanup.
        
        Logs an end timestamp metric and performs cleanup of internal state.
        Safe to call even if no run is active.
        
        Args:
            status: Run completion status (FINISHED, FAILED, KILLED)
        """
        try:
            if self.current_run is None:
                print("No active run to end")
                return
                
            # Log end timestamp for tracking purposes
            try:
                self.log_metric("run_end_timestamp", datetime.now().timestamp())
            except Exception:
                # Don't fail run ending if timestamp logging fails
                pass
                
            # End the MLflow run
            mlflow.end_run(status=status)
            print(f"Ended MLflow run with status: {status}")
            
        finally:
            # Always clean up state regardless of errors
            self.current_run = None
            self.current_run_id = None

    @contextmanager
    def child_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for creating nested child runs.
        
        Automatically handles run lifecycle - starts a nested run on entry
        and ensures proper cleanup on exit.
        
        Args:
            run_name: Optional name for the child run
            tags: Optional tags specific to the child run
            
        Yields:
            str: The child run ID
        """
        self.start_run(run_name=run_name, tags=tags, nested=True)
        try:
            yield self.current_run_id
        finally:
            self.end_run(status="FINISHED")

    # Logging Interface
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter with error handling.
        
        Args:
            key: Parameter name
            value: Parameter value (will be stringified)
        """
        try:
            mlflow.log_param(key, _stringify(value))
        except Exception as e:
            print(f"Failed to log param {key}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters in batch with error handling.
        
        Args:
            params: Dictionary of parameter key-value pairs
        """
        try:
            processed_params = {k: _stringify(v) for k, v in params.items()}
            if processed_params:
                mlflow.log_params(processed_params)
                print(f"Logged {len(processed_params)} parameters")
        except Exception as e:
            print(f"Failed to log params: {e}")

    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None) -> None:
        """
        Log a single metric with type validation and error handling.
        
        Args:
            key: Metric name
            value: Numeric metric value
            step: Optional step number for time series metrics
        """
        try:
            # Ensure numeric type
            if not isinstance(value, (int, float)):
                value = float(value)
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """
        Log multiple metrics in batch with type validation.
        
        Filters out non-numeric values and logs warnings for skipped metrics.
        
        Args:
            metrics: Dictionary of metric key-value pairs
            step: Optional step number for time series metrics
        """
        try:
            numeric_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = value
                else:
                    try:
                        numeric_metrics[key] = float(value)
                    except (ValueError, TypeError):
                        print(f"Skipping non-numeric metric {key}: {value}")
                        
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                
        except Exception as e:
            print(f"Failed to log metrics: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a single tag with error handling.
        
        Args:
            key: Tag name
            value: Tag value (will be stringified)
        """
        try:
            mlflow.set_tag(key, _stringify(value))
        except Exception as e:
            print(f"Failed to set tag {key}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set multiple tags in batch with error handling.
        
        Args:
            tags: Dictionary of tag key-value pairs
        """
        try:
            processed_tags = {k: _stringify(v) for k, v in tags.items()}
            mlflow.set_tags(processed_tags)
            print(f"Set {len(tags)} tags")
        except Exception as e:
            print(f"Failed to set tags: {e}")

    # Artifact Management
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a single file artifact with path validation.
        
        Args:
            local_path: Path to the local file to upload
            artifact_path: Optional path within the artifact store
        """
        try:
            if not os.path.exists(local_path):
                print(f"Artifact file not found: {local_path}")
                return
                
            mlflow.log_artifact(local_path, artifact_path)
            print(f"Logged artifact: {os.path.basename(local_path)}")
            
        except Exception as e:
            print(f"Failed to log artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log all files from a directory as artifacts.
        
        Args:
            local_dir: Path to the local directory to upload
            artifact_path: Optional path within the artifact store
        """
        try:
            if not os.path.isdir(local_dir):
                print(f"Artifact directory not found: {local_dir}")
                return
                
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"Logged artifacts from: {local_dir}")
            
        except Exception as e:
            print(f"Failed to log artifacts: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """
        Log a dictionary as a JSON artifact with version compatibility.
        
        Uses native MLflow functionality when available, falls back to
        manual file creation for older MLflow versions.
        
        Args:
            dictionary: Dictionary to serialize as JSON
            artifact_file: Target filename within artifact store
        """
        try:
            if hasattr(mlflow, "log_dict"):
                # Use native MLflow method if available
                mlflow.log_dict(dictionary, artifact_file=artifact_file)
            else:
                # Fallback for older MLflow versions
                with tempfile.TemporaryDirectory() as tmp_dir:
                    target_path = Path(tmp_dir) / artifact_file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(target_path, 'w', encoding='utf-8') as f:
                        json.dump(dictionary, f, indent=2, default=str)
                    
                    mlflow.log_artifacts(tmp_dir)
                    
            print(f"Logged dictionary as: {artifact_file}")
            
        except Exception as e:
            print(f"Failed to log dictionary: {e}")

    def log_text(self, text: str, artifact_file: str) -> None:
        """
        Log text content as an artifact with version compatibility.
        
        Args:
            text: Text content to save
            artifact_file: Target filename within artifact store
        """
        try:
            if hasattr(mlflow, "log_text"):
                # Use native MLflow method if available
                mlflow.log_text(text, artifact_file=artifact_file)
            else:
                # Fallback for older MLflow versions
                with tempfile.TemporaryDirectory() as tmp_dir:
                    target_path = Path(tmp_dir) / artifact_file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    mlflow.log_artifacts(tmp_dir)
                    
            print(f"Logged text as: {artifact_file}")
            
        except Exception as e:
            print(f"Failed to log text: {e}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure as an artifact with version compatibility.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Target filename within artifact store
        """
        try:
            if hasattr(mlflow, "log_figure"):
                # Use native MLflow method if available
                mlflow.log_figure(figure, artifact_file)
                print(f"Logged figure as: {artifact_file}")
                return
                
            # Fallback: save PNG to temp directory and upload
            with tempfile.TemporaryDirectory() as tmp_dir:
                target_path = Path(tmp_dir) / artifact_file
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Ensure PNG extension for matplotlib compatibility
                if target_path.suffix == "":
                    target_path = target_path.with_suffix(".png")
                
                figure.savefig(str(target_path), bbox_inches="tight")
                mlflow.log_artifacts(tmp_dir)
                
            print(f"Logged figure as: {artifact_file}")
            
        except Exception as e:
            print(f"Failed to log figure: {e}")

    def log_artifact_with_meta(
        self, 
        local_path: str, 
        metadata: Dict[str, Any], 
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a file artifact along with associated metadata.
        
        Uploads the specified file and creates a companion .meta.json file
        containing the provided metadata.
        
        Args:
            local_path: Path to the file to upload
            metadata: Dictionary of metadata to associate with the file
            artifact_path: Optional path within artifact store
        """
        try:
            # Upload the primary artifact
            self.log_artifact(local_path, artifact_path)
            
            # Create metadata filename
            filename = os.path.basename(local_path)
            if artifact_path:
                meta_filename = f"{artifact_path}/{filename}.meta.json"
            else:
                meta_filename = f"{filename}.meta.json"
                
            # Upload metadata as JSON
            self.log_dict(metadata, meta_filename)
            
        except Exception as e:
            print(f"Failed to log artifact metadata: {e}")

    # Model Registry Integration
    
    def register_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_path: Optional custom model path (defaults to current run)
            description: Optional model description
            tags: Optional model version tags
            
        Returns:
            Model version number if successful, None if failed
        """
        try:
            # Determine model URI
            if model_path is None:
                model_uri = f"runs:/{self.current_run_id}/model"
            else:
                model_uri = model_path
            
            # Process tags for MLflow compatibility
            processed_tags = None
            if tags:
                processed_tags = {k: _stringify(v) for k, v in tags.items()}
            
            # Register the model
            result = mlflow.register_model(
                model_uri, 
                model_name, 
                tags=processed_tags
            )
            version = result.version
            
            # Add description if provided
            if description:
                try:
                    client = MlflowClient()
                    client.update_model_version(
                        name=model_name, 
                        version=version, 
                        description=description
                    )
                except Exception:
                    # Description update is optional
                    pass
                    
            print(f"Registered model: {model_name} v{version}")
            return version
            
        except Exception as e:
            print(f"Failed to register model: {e}")
            return None

    # Query and Search Interface
    
    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Search for runs across experiments with flexible filtering.
        
        Args:
            experiment_ids: Optional list of experiment IDs to search
            filter_string: MLflow filter string for run selection
            order_by: Optional list of sort criteria
            max_results: Maximum number of results to return
            
        Returns:
            List of run data dictionaries
        """
        try:
            # Use current experiment if none specified
            if experiment_ids is None and self.current_experiment_id:
                experiment_ids = [self.current_experiment_id]
                
            # Execute search
            runs_df = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
            
            # Convert to list of dictionaries for portability
            return [
                row._asdict() if hasattr(row, "_asdict") else row.to_dict()
                for _, row in runs_df.iterrows()
            ]
            
        except Exception as e:
            print(f"Failed to search runs: {e}")
            return []

    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a specific run.
        
        Args:
            run_id: Optional run ID (defaults to current run)
            
        Returns:
            Dictionary containing run metadata, metrics, params, and tags
        """
        try:
            target_run_id = run_id or self.current_run_id
            if not target_run_id:
                return {}
                
            run = mlflow.get_run(target_run_id)
            
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
                "artifact_uri": run.info.artifact_uri,
            }
            
        except Exception as e:
            print(f"Failed to get run info: {e}")
            return {}

    # Utility Methods
    
    def get_experiment_url(self) -> str:
        """
        Generate URL for the current experiment in MLflow UI.
        
        Returns:
            Web URL for accessing the experiment
        """
        base_url = self.mlflow_config["tracking_uri"].rstrip("/")
        if self.current_experiment_id:
            return f"{base_url}/#/experiments/{self.current_experiment_id}"
        return base_url

    def get_run_url(self) -> str:
        """
        Generate URL for the current run in MLflow UI.
        
        Returns:
            Web URL for accessing the run, falls back to experiment URL
        """
        base_url = self.mlflow_config["tracking_uri"].rstrip("/")
        if self.current_run_id and self.current_experiment_id:
            return f"{base_url}/#/experiments/{self.current_experiment_id}/runs/{self.current_run_id}"
        return self.get_experiment_url()

    # Context Management
    
    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.current_run is not None:
            self.end_run("KILLED")
        print("Cleaned up MLflow backend")