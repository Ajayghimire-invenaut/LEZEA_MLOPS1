"""
LeZeA MLOps Experiment Tracker
=============================

Main interface for Marcus's AGI experiments. This is the primary class that
orchestrates all backends and provides a unified API for experiment tracking.

Usage:
    # Context manager (recommended)
    with ExperimentTracker("my_experiment", "Testing LeZeA") as tracker:
        tracker.log_lezea_config(tasker_pop=100, builder_pop=50)
        tracker.log_training_step(1, loss=0.5, accuracy=0.8)
        tracker.log_checkpoint("model.pth", step=1000)
    
    # Manual usage
    tracker = ExperimentTracker("my_experiment", "Testing LeZeA")
    tracker.start()
    # ... training code ...
    tracker.end()

Features:
- Unified interface to all storage backends
- Automatic environment detection and logging
- Resource monitoring and optimization recommendations
- Business metrics and cost tracking
- Complete experiment lifecycle management
"""

import time
import uuid
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from contextlib import contextmanager

# Import configuration
from .config import config

# Import backends
from .backends.mlflow_backend import MLflowBackend
from .backends.mongodb_backend import MongoBackend
from .backends.s3_backend import S3Backend
from .backends.postgres_backend import PostgresBackend
from .backends.dvc_backend import DVCBackend

# Import monitoring
from .monitoring.gpu_monitor import GPUMonitor
from .monitoring.env_tags import EnvironmentTagger

# Import utilities
from .utils.logging import setup_logging, get_logger
from .utils.validation import validate_experiment_name, validate_metrics


class ExperimentTracker:
    """
    Main experiment tracker for LeZeA MLOps system
    
    This class orchestrates all backends and provides a unified interface
    for tracking LeZeA AGI experiments with comprehensive monitoring,
    storage, and analytics capabilities.
    """
    
    def __init__(self, experiment_name: str, purpose: str = "", 
                 tags: Dict[str, str] = None, auto_start: bool = False):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            purpose: Description/purpose of the experiment  
            tags: Additional tags for the experiment
            auto_start: Whether to automatically start the experiment
        """
        # Validate inputs
        validate_experiment_name(experiment_name)
        
        # Core experiment info
        self.experiment_id = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.purpose = purpose
        self.tags = tags or {}
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = False
        
        # Setup logging
        self.logger = get_logger(f"experiment.{experiment_name}")
        self.logger.info(f"Initializing experiment: {experiment_name}")
        
        # Backend availability tracking
        self.backends = {}
        self.backend_errors = {}
        
        # Initialize backends
        self._init_backends()
        
        # Initialize monitoring
        self._init_monitoring()
        
        # Experiment state tracking
        self.lezea_config = {}
        self.constraints = {}
        self.training_steps = 0
        self.checkpoints_saved = 0
        self.total_cost = 0.0
        
        # Performance tracking
        self.step_times = []
        self.resource_warnings = []
        
        self.logger.info(f"Experiment tracker initialized - ID: {self.experiment_id[:8]}...")
        print(f"🚀 LeZeA MLOps Tracker Ready")
        print(f"   Experiment: {experiment_name}")
        print(f"   ID: {self.experiment_id[:8]}...")
        print(f"   Purpose: {purpose}")
        print(f"   Backends: {len([b for b in self.backends.values() if b is not None])}/{len(self.backends)}")
        
        # Auto-start if requested
        if auto_start:
            self.start()
    
    def _init_backends(self):
        """Initialize all storage backends with error handling"""
        backend_configs = [
            ('mlflow', MLflowBackend, "Experiment tracking"),
            ('mongodb', MongoBackend, "Complex data storage"),
            ('s3', S3Backend, "Artifact storage"),
            ('postgres', PostgresBackend, "Metadata storage"),
            ('dvc', DVCBackend, "Dataset versioning")
        ]
        
        for backend_name, backend_class, description in backend_configs:
            try:
                self.backends[backend_name] = backend_class(config)
                self.logger.info(f"✅ {description} backend ready")
            except Exception as e:
                self.backends[backend_name] = None
                self.backend_errors[backend_name] = str(e)
                self.logger.warning(f"❌ {description} backend failed: {e}")
                print(f"⚠️ {description} unavailable: {e}")
    
    def _init_monitoring(self):
        """Initialize monitoring components"""
        try:
            self.gpu_monitor = GPUMonitor()
            self.logger.info(f"✅ GPU monitoring ready ({self.gpu_monitor.device_count} devices)")
        except Exception as e:
            self.gpu_monitor = None
            self.logger.warning(f"❌ GPU monitoring failed: {e}")
        
        try:
            self.env_tagger = EnvironmentTagger()
            self.logger.info("✅ Environment detection ready")
        except Exception as e:
            self.env_tagger = None
            self.logger.warning(f"❌ Environment detection failed: {e}")
    
    def start(self):
        """Start the experiment tracking"""
        if self.is_active:
            self.logger.warning("Experiment already active")
            print("⚠️ Experiment already active")
            return self
        
        try:
            self.logger.info("Starting experiment tracking")
            
            # Start MLflow run
            if self.backends['mlflow']:
                self.backends['mlflow'].create_experiment(
                    self.experiment_name, 
                    self.experiment_id
                )
                self.backends['mlflow'].start_run(
                    f"{self.experiment_name}_{self.experiment_id[:8]}",
                    tags=self.tags
                )
                
                # Log experiment metadata
                self._log_experiment_metadata()
                
                # Log environment information
                self._log_environment()
            
            # Store experiment in MongoDB
            if self.backends['mongodb']:
                self.backends['mongodb'].store_experiment_metadata(
                    self.experiment_id,
                    {
                        'name': self.experiment_name,
                        'purpose': self.purpose,
                        'tags': self.tags,
                        'start_time': self.start_time.isoformat(),
                        'backends_available': [k for k, v in self.backends.items() if v is not None]
                    }
                )
            
            # Start resource monitoring
            if self.gpu_monitor:
                self.gpu_monitor.start_monitoring(self.experiment_id)
            
            self.is_active = True
            self.logger.info(f"Experiment started: {self.experiment_name}")
            print(f"🎯 Started experiment: {self.experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment: {e}")
            self.logger.error(traceback.format_exc())
            print(f"❌ Failed to start experiment: {e}")
            raise
        
        return self
    
    def log_lezea_config(self, 
                        tasker_pop_size: int,
                        builder_pop_size: int,
                        algorithm_type: str,
                        start_network_id: str = None,
                        hyperparameters: Dict[str, Any] = None,
                        seeds: Dict[str, int] = None):
        """
        Log LeZeA-specific configuration
        
        Args:
            tasker_pop_size: Size of tasker population
            builder_pop_size: Size of builder population
            algorithm_type: Algorithm type (DQN, SAC, etc.)
            start_network_id: Starting network identifier
            hyperparameters: Hyperparameter dictionary
            seeds: Seed dictionary for reproducibility
        """
        if not self.is_active:
            self.logger.warning("Experiment not active. Call start() first.")
            print("⚠️ Experiment not active. Call start() first.")
            return
        
        self.lezea_config = {
            'tasker_population_size': tasker_pop_size,
            'builder_population_size': builder_pop_size,
            'algorithm_type': algorithm_type,
            'start_network_id': start_network_id,
            'hyperparameters': hyperparameters or {},
            'seeds': seeds or {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Log to MLflow as parameters
            if self.backends['mlflow']:
                params = {
                    'tasker_pop_size': tasker_pop_size,
                    'builder_pop_size': builder_pop_size,
                    'algorithm_type': algorithm_type,
                    'start_network_id': start_network_id or 'none'
                }
                
                # Add hyperparameters with prefix
                if hyperparameters:
                    params.update({f'hp_{k}': v for k, v in hyperparameters.items()})
                
                # Add seeds with prefix
                if seeds:
                    params.update({f'seed_{k}': v for k, v in seeds.items()})
                
                self.backends['mlflow'].log_params(params)
            
            # Store in MongoDB for complex queries
            if self.backends['mongodb']:
                self.backends['mongodb'].store_lezea_config(
                    self.experiment_id, 
                    self.lezea_config
                )
            
            self.logger.info(f"Logged LeZeA config: {tasker_pop_size} taskers, {builder_pop_size} builders")
            print(f"📝 Logged LeZeA config: {tasker_pop_size} taskers, {builder_pop_size} builders")
            
        except Exception as e:
            self.logger.error(f"Failed to log LeZeA config: {e}")
            print(f"❌ Failed to log LeZeA config: {e}")
    
    def log_constraints(self, 
                       max_runtime: int = None,
                       max_steps: int = None,
                       max_episodes: int = None):
        """Log experiment constraints"""
        if not self.is_active:
            self.logger.warning("Experiment not active")
            return
        
        self.constraints = {
            'max_runtime_seconds': max_runtime,
            'max_steps': max_steps,
            'max_episodes': max_episodes,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Log to MLflow
            if self.backends['mlflow']:
                constraint_params = {k: v for k, v in self.constraints.items() 
                                   if v is not None and k != 'timestamp'}
                self.backends['mlflow'].log_params(constraint_params)
            
            self.logger.info(f"Logged constraints: {constraint_params}")
            print(f"⏱️ Logged constraints: {constraint_params}")
            
        except Exception as e:
            self.logger.error(f"Failed to log constraints: {e}")
            print(f"❌ Failed to log constraints: {e}")
    
    def log_training_step(self, step: int, **metrics):
        """
        Log training metrics for a step
        
        Args:
            step: Training step number
            **metrics: Arbitrary metrics (loss, accuracy, reward, etc.)
        """
        if not self.is_active:
            self.logger.warning("Experiment not active")
            return
        
        step_start_time = time.time()
        
        try:
            # Validate metrics
            validated_metrics = validate_metrics(metrics)
            
            # Track step timing
            if self.step_times:
                step_duration = step_start_time - self.step_times[-1]
                validated_metrics['step_duration_seconds'] = step_duration
            
            self.step_times.append(step_start_time)
            self.training_steps += 1
            
            # Log to MLflow
            if self.backends['mlflow']:
                self.backends['mlflow'].log_metrics(validated_metrics, step=step)
            
            # Store in MongoDB for complex analysis
            if self.backends['mongodb']:
                step_data = {
                    'step': step,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': validated_metrics,
                    'experiment_id': self.experiment_id
                }
                self.backends['mongodb'].store_training_step(self.experiment_id, step_data)
            
            # Check for performance issues
            self._check_performance_warnings(step, validated_metrics)
            
            # Progress logging
            if step % 100 == 0 or step < 10:
                metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                       for k, v in validated_metrics.items()])
                self.logger.info(f"Step {step}: {metrics_str}")
                print(f"📊 Step {step}: {metrics_str}")
            
        except Exception as e:
            self.logger.error(f"Failed to log training step {step}: {e}")
            print(f"❌ Failed to log step {step}: {e}")
    
    def log_modification_tree(self, step: int, modifications: List[Dict], 
                            statistics: Dict[str, Any]):
        """
        Log model modification trees and statistics
        
        Args:
            step: Training step
            modifications: List of modifications made
            statistics: Modification statistics
        """
        if not self.is_active:
            return
        
        try:
            modification_data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'modifications': modifications,
                'statistics': statistics
            }
            
            # Store in MongoDB (perfect for hierarchical data)
            if self.backends['mongodb']:
                self.backends['mongodb'].store_modification_tree(
                    self.experiment_id, 
                    modification_data
                )
            
            # Log summary statistics to MLflow
            if self.backends['mlflow'] and statistics:
                summary_metrics = {f"mod_{k}": v for k, v in statistics.items() 
                                 if isinstance(v, (int, float))}
                self.backends['mlflow'].log_metrics(summary_metrics, step=step)
            
            self.logger.info(f"Logged modification tree: {len(modifications)} changes at step {step}")
            print(f"🌳 Logged modification tree: {len(modifications)} changes at step {step}")
            
        except Exception as e:
            self.logger.error(f"Failed to log modification tree: {e}")
            print(f"❌ Failed to log modification tree: {e}")
    
    def log_checkpoint(self, checkpoint_path: str, step: int = None, 
                      metadata: Dict[str, Any] = None):
        """
        Log model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step when saved
            metadata: Additional checkpoint metadata
        """
        if not self.is_active:
            return
        
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return
        
        try:
            # Upload to S3
            s3_key = None
            if self.backends['s3']:
                s3_key = self.backends['s3'].upload_checkpoint(
                    checkpoint_path,
                    self.experiment_id,
                    step,
                    metadata
                )
            
            # Log to MLflow
            if self.backends['mlflow']:
                self.backends['mlflow'].log_artifact(checkpoint_path)
                if step is not None:
                    self.backends['mlflow'].log_metric("checkpoint_step", step)
            
            self.checkpoints_saved += 1
            
            self.logger.info(f"Logged checkpoint: {os.path.basename(checkpoint_path)} -> S3: {s3_key}")
            print(f"💾 Logged checkpoint: {os.path.basename(checkpoint_path)} -> S3: {s3_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to log checkpoint: {e}")
            print(f"❌ Failed to log checkpoint: {e}")
    
    def log_final_models(self, tasker_model_path: str = None, 
                        builder_model_path: str = None, metadata: Dict = None):
        """
        Log final trained models
        
        Args:
            tasker_model_path: Path to final tasker model
            builder_model_path: Path to final builder model
            metadata: Additional model metadata
        """
        if not self.is_active:
            return
        
        models_logged = []
        
        try:
            # Upload tasker model
            if tasker_model_path and os.path.exists(tasker_model_path):
                if self.backends['s3']:
                    s3_key = self.backends['s3'].upload_final_model(
                        tasker_model_path,
                        self.experiment_id,
                        'tasker',
                        f"final_tasker_{self.experiment_id[:8]}",
                        metadata
                    )
                    models_logged.append(f"tasker -> {s3_key}")
            
            # Upload builder model
            if builder_model_path and os.path.exists(builder_model_path):
                if self.backends['s3']:
                    s3_key = self.backends['s3'].upload_final_model(
                        builder_model_path,
                        self.experiment_id,
                        'builder',
                        f"final_builder_{self.experiment_id[:8]}",
                        metadata
                    )
                    models_logged.append(f"builder -> {s3_key}")
            
            if models_logged:
                self.logger.info(f"Logged final models: {', '.join(models_logged)}")
                print(f"🏆 Logged final models: {', '.join(models_logged)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log final models: {e}")
            print(f"❌ Failed to log final models: {e}")
    
    def log_business_metrics(self, cost: float, comments: str = "", conclusion: str = ""):
        """
        Log business metrics and conclusions
        
        Args:
            cost: Total cost of resources used
            comments: Additional comments
            conclusion: Experiment conclusion
        """
        if not self.is_active:
            return
        
        try:
            business_data = {
                'cost': cost,
                'comments': comments,
                'conclusion': conclusion,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in MongoDB
            if self.backends['mongodb']:
                self.backends['mongodb'].store_business_metrics(
                    self.experiment_id, 
                    business_data
                )
            
            # Log cost to MLflow
            if self.backends['mlflow']:
                self.backends['mlflow'].log_metric("total_cost", cost)
                self.backends['mlflow'].log_text(
                    f"{comments}\n\nConclusion: {conclusion}", 
                    "experiment_notes.txt"
                )
            
            self.total_cost += cost
            
            self.logger.info(f"Logged business metrics: ${cost:.2f} cost")
            print(f"💰 Logged business metrics: ${cost:.2f} cost")
            
        except Exception as e:
            self.logger.error(f"Failed to log business metrics: {e}")
            print(f"❌ Failed to log business metrics: {e}")
    
    def version_dataset(self, dataset_name: str, data_paths: List[str], 
                       description: str = ""):
        """
        Version dataset using DVC
        
        Args:
            dataset_name: Name of the dataset
            data_paths: List of data file paths
            description: Dataset description
        """
        if not self.backends['dvc'] or not self.backends['dvc'].available:
            self.logger.warning("DVC not available for dataset versioning")
            print("❌ DVC not available for dataset versioning")
            return None
        
        try:
            version_info = self.backends['dvc'].version_dataset(
                self.experiment_id,
                dataset_name,
                data_paths,
                description
            )
            
            # Log dataset info to MLflow
            if self.backends['mlflow'] and version_info:
                self.backends['mlflow'].log_params({
                    f"dataset_{dataset_name}_version": version_info.get('version_tag'),
                    f"dataset_{dataset_name}_size_mb": version_info.get('total_size_mb'),
                    f"dataset_{dataset_name}_files": version_info.get('file_count')
                })
            
            self.logger.info(f"Versioned dataset: {dataset_name} -> {version_info.get('version_tag')}")
            print(f"📦 Versioned dataset: {dataset_name} -> {version_info.get('version_tag')}")
            
            return version_info
            
        except Exception as e:
            self.logger.error(f"Failed to version dataset: {e}")
            print(f"❌ Failed to version dataset: {e}")
            return None
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        try:
            duration = (datetime.now() - self.start_time).total_seconds() if self.is_active else \
                      (self.end_time - self.start_time).total_seconds() if self.end_time else 0
            
            summary = {
                'experiment_id': self.experiment_id,
                'name': self.experiment_name,
                'purpose': self.purpose,
                'status': 'Running' if self.is_active else 'Completed',
                'duration_seconds': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'training_steps': self.training_steps,
                'checkpoints_saved': self.checkpoints_saved,
                'total_cost': self.total_cost,
                'lezea_config': self.lezea_config,
                'constraints': self.constraints,
                'backends_available': [k for k, v in self.backends.items() if v is not None],
                'backend_errors': self.backend_errors
            }
            
            # Add performance metrics
            if self.step_times:
                summary['avg_step_time'] = sum(
                    self.step_times[i] - self.step_times[i-1] 
                    for i in range(1, len(self.step_times))
                ) / max(1, len(self.step_times) - 1)
            
            # Add resource warnings
            if self.resource_warnings:
                summary['resource_warnings'] = self.resource_warnings[-5:]  # Last 5 warnings
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment summary: {e}")
            return {'error': str(e)}
    
    def get_recommendations(self) -> List[str]:
        """Get performance and optimization recommendations"""
        recommendations = []
        
        try:
            # Check experiment duration
            duration = (datetime.now() - self.start_time).total_seconds()
            if duration > 3600:  # > 1 hour
                recommendations.append("Long experiment duration - consider checkpointing more frequently")
            
            # Check step timing
            if len(self.step_times) > 10:
                recent_times = self.step_times[-10:]
                step_durations = [recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))]
                avg_duration = sum(step_durations) / len(step_durations)
                
                if avg_duration > 10:  # > 10 seconds per step
                    recommendations.append("Slow training steps - consider batch size optimization")
            
            # Check backend availability
            failed_backends = [k for k, v in self.backends.items() if v is None]
            if failed_backends:
                recommendations.append(f"Some backends unavailable: {', '.join(failed_backends)} - consider fixing for full functionality")
            
            # Check GPU monitoring
            if self.gpu_monitor and hasattr(self.gpu_monitor, 'get_current_usage'):
                try:
                    usage = self.gpu_monitor.get_current_usage()
                    if usage and usage.get('memory_percent', 0) > 90:
                        recommendations.append("High GPU memory usage - consider reducing batch size")
                except:
                    pass
            
            # Resource warnings
            if len(self.resource_warnings) > 5:
                recommendations.append("Multiple resource warnings - review system capacity")
            
            if not recommendations:
                recommendations.append("Experiment running efficiently - no recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def _log_experiment_metadata(self):
        """Log experiment metadata to MLflow"""
        try:
            if self.backends['mlflow']:
                metadata = {
                    'experiment_id': self.experiment_id,
                    'experiment_name': self.experiment_name,
                    'purpose': self.purpose,
                    'start_timestamp': self.start_time.isoformat()
                }
                metadata.update(self.tags)
                self.backends['mlflow'].set_tags(metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to log experiment metadata: {e}")
    
    def _log_environment(self):
        """Log environment information"""
        try:
            if self.env_tagger:
                env_info = self.env_tagger.get_environment_info()
                
                # Log to MLflow as tags
                if self.backends['mlflow']:
                    env_tags = self.env_tagger.get_mlflow_tags(env_info)
                    self.backends['mlflow'].set_tags(env_tags)
                    
                    # Log detailed info as artifact
                    self.backends['mlflow'].log_dict(env_info, "environment_info.json")
                
                self.logger.info(f"Logged environment: {env_info.get('system', {}).get('python_version', 'unknown')}")
                
        except Exception as e:
            self.logger.error(f"Failed to log environment: {e}")
    
    def _check_performance_warnings(self, step: int, metrics: Dict[str, Any]):
        """Check for performance issues and generate warnings"""
        try:
            warnings = []
            
            # Check for NaN or infinite values
            for key, value in metrics.items():
                if isinstance(value, float):
                    if value != value:  # NaN check
                        warnings.append(f"NaN value detected in {key} at step {step}")
                    elif abs(value) == float('inf'):
                        warnings.append(f"Infinite value detected in {key} at step {step}")
            
            # Check loss trends
            if 'loss' in metrics and len(self.step_times) > 10:
                # Could add loss trend analysis here
                pass
            
            # Add to warnings list
            if warnings:
                self.resource_warnings.extend(warnings)
                # Keep only last 20 warnings
                self.resource_warnings = self.resource_warnings[-20:]
                
                for warning in warnings:
                    self.logger.warning(warning)
                    
        except Exception as e:
            self.logger.error(f"Failed to check performance warnings: {e}")
    
    def end(self):
        """End the experiment and clean up"""
        if not self.is_active:
            self.logger.warning("Experiment not active")
            print("⚠️ Experiment not active")
            return
        
        try:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            self.logger.info("Ending experiment")
            
            # Stop resource monitoring
            if self.gpu_monitor:
                self.gpu_monitor.stop_monitoring()
            
            # Log final metrics
            if self.backends['mlflow']:
                final_metrics = {
                    "experiment_duration_seconds": duration,
                    "total_training_steps": self.training_steps,
                    "total_checkpoints": self.checkpoints_saved,
                    "total_cost": self.total_cost
                }
                self.backends['mlflow'].log_metrics(final_metrics)
                
                # End MLflow run
                self.backends['mlflow'].end_run()
            
            # Store final experiment summary
            if self.backends['mongodb']:
                summary = self.get_experiment_summary()
                self.backends['mongodb'].store_experiment_summary(
                    self.experiment_id, 
                    summary
                )
            
            self.is_active = False
            
            self.logger.info(f"Experiment completed - Duration: {duration:.1f}s")
            print(f"🏁 Experiment completed!")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Training steps: {self.training_steps}")
            print(f"   Checkpoints: {self.checkpoints_saved}")
            print(f"   Total cost: ${self.total_cost:.2f}")
            
            # Show MLflow URL
            if self.backends['mlflow']:
                print(f"   View results: {self.backends['mlflow'].get_run_url()}")
            
            # Show recommendations
            recommendations = self.get_recommendations()
            if recommendations:
                print("\n💡 Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"   • {rec}")
                    
        except Exception as e:
            self.logger.error(f"Error ending experiment: {e}")
            self.logger.error(traceback.format_exc())
            print(f"❌ Error ending experiment: {e}")
    
    def __enter__(self):
        """Context manager entry - automatically start experiment"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically end experiment"""
        if exc_type is not None:
            self.logger.error(f"Experiment failed with {exc_type.__name__}: {exc_val}")
            print(f"❌ Experiment failed: {exc_val}")
        
        self.end()
        
        # Don't suppress exceptions