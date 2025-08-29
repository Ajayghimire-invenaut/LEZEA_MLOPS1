"""
LeZeA MLOps Experiment Tracker

A comprehensive tracking system for LeZeA (Learning-Zero-Alpha) experiments that integrates
with multiple storage backends and provides detailed monitoring of network evolution, population
dynamics, reward flows, and resource utilization.

This module provides unified experiment tracking aligned with MLOps best practices, supporting
distributed backends including MLflow, MongoDB, S3, PostgreSQL, and DVC for comprehensive
data management and experiment reproducibility.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import traceback
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping
from contextlib import contextmanager
from collections import defaultdict, Counter
from threading import Thread, Event
from queue import Queue, Empty
import tempfile
from dataclasses import dataclass, field
from enum import Enum
import platform
import subprocess

# Configuration management
from .config import config

# Backend integrations
from .backends.mlflow_backend import MLflowBackend
from .backends.mongodb_backend import MongoBackend
from .backends.s3_backend import S3Backend
from .backends.postgres_backend import PostgresBackend
from .backends.dvc_backend import DVCBackend

# System monitoring components
from .monitoring.gpu_monitor import GPUMonitor
from .monitoring.env_tags import EnvironmentTagger

# Data usage tracking (optional component)
try:
    from .monitoring.data_usage import DataUsageLogger
except Exception:
    DataUsageLogger = None

# Core utilities
from .utils.logging import get_logger
from .utils.validation import validate_experiment_name, validate_metrics

# Business cost modeling (optional)
try:
    from .business.cost_model import CostModel
except Exception:
    CostModel = None

# LeZeA-specific modification tracking
try:
    from .modification.trees import ModificationTree
except Exception:
    # Compatibility fallback for repositories using ModTree naming
    from .modification.trees import ModTree as ModificationTree


class NetworkType(Enum):
    """Network types in the LeZeA ecosystem."""
    TASKER = "tasker"
    BUILDER = "builder"
    HYBRID = "hybrid"


class PopulationStatus(Enum):
    """Population lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EVOLVING = "evolving"
    CONVERGED = "converged"
    TERMINATED = "terminated"


@dataclass
class NetworkLineage:
    """
    Tracks genealogy and inheritance relationships between networks.
    
    Maintains creation history, parent-child relationships, and fitness evolution
    for network populations in evolutionary algorithms.
    """
    network_id: str
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    modification_count: int = 0
    fitness_score: Optional[float] = None


@dataclass
class PopulationSnapshot:
    """
    Point-in-time capture of population state.
    
    Records demographic and fitness statistics for analyzing population
    evolution patterns and convergence behavior.
    """
    timestamp: datetime
    tasker_count: int
    builder_count: int
    generation: int
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    diversity_metric: float


@dataclass
class LayerSeedConfig:
    """
    Layer-level initialization configuration.
    
    Manages reproducible initialization of neural network layers with
    specific seeds and parameter configurations.
    """
    layer_id: str
    layer_type: str
    seed: int
    initialization_method: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardFlow:
    """
    Reward transfer between network components.
    
    Tracks incentive flows between taskers and builders, capturing performance
    attribution and task-specific reward distributions.
    """
    source_id: str
    target_id: str
    source_type: NetworkType
    target_type: NetworkType
    reward_value: float
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


def _compute_file_hash(filepath: str) -> str:
    """
    Compute SHA256 hash of a file for integrity verification.
    
    Args:
        filepath: Path to the file to hash
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    hash_obj = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _get_git_commit() -> Optional[str]:
    """
    Retrieve current Git commit hash for version tracking.
    
    Returns:
        Git commit hash if available, None otherwise
    """
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return result
    except Exception:
        return None


def _flatten_nested_params(params: Dict[str, Any], *, max_length: int = 400) -> Dict[str, Any]:
    """
    Flatten nested parameter dictionaries for backend storage.
    
    Converts complex parameter structures into simple key-value pairs suitable
    for MLflow parameter logging with size constraints.
    
    Args:
        params: Parameter dictionary to flatten
        max_length: Maximum string length for parameter values
        
    Returns:
        Flattened parameter dictionary
    """
    flattened: Dict[str, Any] = {}
    
    for key, value in (params or {}).items():
        if isinstance(value, (int, float, bool)) or value is None:
            flattened[key] = value
        elif isinstance(value, str):
            flattened[key] = value if len(value) <= max_length else (value[:max_length - 3] + "...")
        else:
            try:
                serialized = json.dumps(value, separators=(",", ":"))
                flattened[key] = serialized if len(serialized) <= max_length else (serialized[:max_length - 3] + "...")
            except Exception:
                flattened[key] = str(value)[:max_length]
                
    return flattened


def _extract_numeric_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """
    Extract numeric values from metrics dictionary, discarding non-numeric entries.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Dictionary containing only successfully converted numeric values
    """
    numeric_metrics: Dict[str, float] = {}
    
    for key, value in metrics.items():
        try:
            numeric_metrics[key] = float(value)
        except (ValueError, TypeError):
            continue  # Skip non-numeric values
            
    return numeric_metrics


class ExperimentTracker:
    """
    Comprehensive experiment tracking system for LeZeA research.
    
    Provides unified interface for logging experiments across multiple storage backends,
    with specialized support for evolutionary algorithms, network genealogy tracking,
    population dynamics monitoring, and resource attribution.
    
    Key Features:
    - Multi-backend storage (MLflow, MongoDB, S3, PostgreSQL, DVC)
    - Asynchronous logging for performance
    - Network lineage and genealogy tracking
    - Population evolution monitoring
    - Reward flow analysis
    - Component-level resource attribution
    - Business cost tracking
    - Data usage and learning relevance analysis
    """

    def __init__(
        self,
        experiment_name: str,
        purpose: str = "",
        tags: Optional[Dict[str, str]] = None,
        auto_start: bool = False,
        *,
        async_mode: bool = False,
        strict_validation: bool = False,
        local_fallback_dir: str = "artifacts/local",
    ) -> None:
        """
        Initialize experiment tracker with configuration.

        Args:
            experiment_name: Unique name identifying this experiment
            purpose: Brief description of experiment objectives
            tags: Optional metadata tags for categorization
            auto_start: Whether to automatically start experiment tracking
            async_mode: Enable non-blocking asynchronous logging
            strict_validation: Require all critical backends to be available
            local_fallback_dir: Directory for local artifact storage fallback
        """
        validate_experiment_name(experiment_name)
        
        # Core experiment metadata
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.purpose = purpose
        self.tags = tags or {}
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.is_active = False
        self.run_id: str = self.experiment_id
        
        # Logging infrastructure
        self.logger = get_logger(f"experiment.{experiment_name}")
        self.logger.info(f"Initializing experiment tracker: {experiment_name}")
        
        # Backend management
        self.backends: Dict[str, Any] = {}
        self.backend_errors: Dict[str, str] = {}
        self._initialize_backends()
        
        # System monitoring
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.env_tagger: Optional[EnvironmentTagger] = None
        self._initialize_monitoring()
        
        # Experiment state tracking
        self.lezea_config: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
        self.training_steps: int = 0
        self.checkpoints_saved: int = 0
        self.total_cost: float = 0.0
        self.strict_validation = strict_validation
        
        # Performance and resource tracking
        self._step_execution_times: List[float] = []
        self._resource_warnings: List[str] = []
        self._last_recorded_loss: Optional[float] = None
        self._scope_stack: List[Dict[str, str]] = []
        self._resource_accumulator: Dict[str, Dict[str, float]] = {}
        self._constraint_violation_signaled: bool = False
        
        # Results aggregation (per scope)
        self._tasker_reward_sums: Dict[str, Counter] = defaultdict(Counter)
        self._tasker_reward_counts: Dict[str, int] = defaultdict(int)
        self._builder_reward_sums: Dict[str, Counter] = defaultdict(Counter)
        self._builder_reward_counts: Dict[str, int] = defaultdict(int)
        
        # Reinforcement learning metrics
        self._rl_total_rewards: Dict[str, float] = defaultdict(float)
        self._rl_total_steps: Dict[str, int] = defaultdict(int)
        self._rl_episode_counts: Dict[str, int] = defaultdict(int)
        self._rl_action_distributions: Dict[str, Counter] = defaultdict(Counter)
        
        # Classification metrics
        self._classification_sample_counts: Dict[str, int] = defaultdict(int)
        self._classification_accuracy_sums: Dict[str, float] = defaultdict(float)
        self._classification_f1_sums: Dict[str, float] = defaultdict(float)
        
        # Generation/evolutionary metrics
        self._generation_counts: Dict[str, int] = defaultdict(int)
        self._generation_score_sums: Dict[str, float] = defaultdict(float)
        
        # LeZeA-specific tracking structures
        self.network_lineages: Dict[str, NetworkLineage] = {}
        self.population_history: List[PopulationSnapshot] = []
        self.layer_seeds: Dict[str, LayerSeedConfig] = {}
        self.reward_flows: List[RewardFlow] = []
        self.modification_trees: Dict[int, ModificationTree] = {}
        self.challenge_usage_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.sample_importance_weights: Dict[str, float] = {}
        self.component_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.tasker_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.builder_evaluations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.population_fitness: Dict[str, List[float]] = defaultdict(list)
        
        # Runtime context for data retrieval
        self._model_metadata: Dict[str, Any] = {}
        self._artifact_registry: List[str] = []
        
        # Optional component initialization
        self.cost_model = CostModel.from_env() if CostModel else None
        self.data_usage_logger = DataUsageLogger() if DataUsageLogger else None
        
        # Asynchronous processing infrastructure
        self.async_mode = bool(async_mode)
        self.local_fallback_dir = local_fallback_dir
        self._async_queue: Queue[Tuple[str, str, tuple, dict]] = Queue()
        self._stop_event: Event = Event()
        self._worker_thread: Optional[Thread] = None
        
        # User feedback
        self._log_initialization_summary()
        
        if auto_start:
            self.start()

    def _log_initialization_summary(self) -> None:
        """Log initialization summary to console."""
        print("LeZeA MLOps Experiment Tracker - Ready")
        print(f"Experiment: {self.experiment_name}")
        print(f"ID: {self.experiment_id[:8]}...")
        print(f"Purpose: {self.purpose}")
        
        if self.async_mode:
            print("Logging Mode: Asynchronous (non-blocking)")
        else:
            print("Logging Mode: Synchronous")

    def _execute_backend_method(self, backend_name: str, method: str, *args, **kwargs):
        """
        Execute backend method with retry logic and error handling.
        
        Args:
            backend_name: Name of the backend to execute on
            method: Method name to invoke
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Method execution result
            
        Raises:
            RuntimeError: If backend is unavailable or all retries fail
        """
        backend = self.backends.get(backend_name)
        if backend is None:
            raise RuntimeError(f"Backend '{backend_name}' is not available")
            
        max_attempts = 3
        base_delay = 0.2
        
        for attempt in range(max_attempts):
            try:
                return getattr(backend, method)(*args, **kwargs)
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise
                    
                self.logger.warning(
                    f"Retry {attempt + 1}/{max_attempts - 1} for {backend_name}.{method} failed: {exc}"
                )
                
                delay = base_delay + random.random() * base_delay
                time.sleep(delay)
                base_delay *= 2

    def _submit_backend_operation(
        self, 
        backend_name: str, 
        method: str, 
        *args, 
        async_allowed: bool = True, 
        **kwargs
    ):
        """
        Submit backend operation, using async queue if enabled.
        
        Args:
            backend_name: Backend to execute operation on
            method: Method to invoke
            *args: Method arguments
            async_allowed: Whether this operation can be executed asynchronously
            **kwargs: Method keyword arguments
            
        Returns:
            Result if executed synchronously, None if queued for async execution
        """
        if self.async_mode and async_allowed:
            self._async_queue.put((backend_name, method, args, kwargs))
            return None
        else:
            return self._execute_backend_method(backend_name, method, *args, **kwargs)

    def _process_async_queue(self):
        """Process asynchronous backend operations from queue."""
        while not self._stop_event.is_set() or not self._async_queue.empty():
            try:
                backend_name, method, args, kwargs = self._async_queue.get(timeout=0.2)
            except Empty:
                continue
                
            try:
                self._execute_backend_method(backend_name, method, *args, **kwargs)
            except Exception as exc:
                # Attempt fallback for dictionary payloads
                payload = kwargs.get("payload") or (args[0] if args else None)
                if isinstance(payload, dict) and method == "log_dict":
                    artifact_path = kwargs.get("artifact_file", "failed_async.json")
                    self._write_fallback_artifact(artifact_path, payload)
                    
                self.logger.warning(
                    f"Async operation failed {backend_name}.{method}: {exc}"
                )

    def _start_async_worker(self):
        """Start asynchronous processing worker thread."""
        if self.async_mode and self._worker_thread is None:
            self._worker_thread = Thread(target=self._process_async_queue, daemon=True)
            self._worker_thread.start()

    def _stop_async_worker(self):
        """Stop asynchronous processing worker thread."""
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
            self._stop_event.clear()

    def _write_fallback_artifact(self, relative_path: str, data: Dict[str, Any]) -> None:
        """
        Write artifact to local fallback directory.
        
        Args:
            relative_path: Relative path within fallback directory
            data: Data to write as JSON
        """
        try:
            base_dir = Path(self.local_fallback_dir) / self.experiment_id
            artifact_path = base_dir / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(artifact_path, "w") as file:
                json.dump(data, file, indent=2)
        except Exception:
            # Silent failure for fallback mechanism
            pass
    
    def _initialize_backends(self) -> None:
        """Initialize all configured storage backends with graceful degradation."""
        backend_registry: List[Tuple[str, Any, str]] = [
            ("mlflow", MLflowBackend, "Experiment tracking and UI"),
            ("mongodb", MongoBackend, "Document storage and complex queries"),
            ("s3", S3Backend, "Artifact storage and distribution"),
            ("postgres", PostgresBackend, "Structured metadata storage"),
            ("dvc", DVCBackend, "Dataset version control"),
        ]
        
        for name, backend_class, description in backend_registry:
            try:
                self.backends[name] = backend_class(config)
                self.logger.info(f"Backend initialized: {description}")
            except Exception as exc:
                self.backends[name] = None
                self.backend_errors[name] = str(exc)
                self.logger.warning(f"Backend initialization failed - {description}: {exc}")
                print(f"Warning: {description} unavailable: {exc}")

    def _initialize_monitoring(self) -> None:
        """Initialize system monitoring components."""
        try:
            self.gpu_monitor = GPUMonitor()
            self.logger.info("GPU monitoring system initialized")
        except Exception as exc:
            self.gpu_monitor = None
            self.logger.warning(f"GPU monitoring initialization failed: {exc}")
            
        try:
            self.env_tagger = EnvironmentTagger()
            self.logger.info("Environment detection system initialized")
        except Exception as exc:
            self.env_tagger = None
            self.logger.warning(f"Environment detection initialization failed: {exc}")

    def start(self, *, prometheus_port: Optional[int] = 8000, strict: Optional[bool] = None) -> "ExperimentTracker":
        """
        Start experiment tracking and initialize backends.

        Args:
            prometheus_port: Port for Prometheus metrics exporter
            strict: Override strict validation setting
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If strict validation is enabled and critical backends fail
        """
        if self.is_active:
            self.logger.warning("Experiment tracking is already active")
            print("Warning: Experiment tracking already active")
            return self
            
        strict_mode = self.strict_validation if strict is None else bool(strict)
        
        # Initialize MLflow experiment and run
        if self.backends.get("mlflow"):
            self._execute_backend_method("mlflow", "create_experiment", self.experiment_name, self.experiment_id)
            run_id = self._execute_backend_method(
                "mlflow",
                "start_run",
                f"{self.experiment_name}_{self.experiment_id[:8]}",
                tags=self.tags,
            )
            if isinstance(run_id, str):
                self.run_id = run_id
            self._log_experiment_metadata()
            self._log_environment_info()
            
        # Initialize MongoDB experiment record
        if self.backends.get("mongodb"):
            self._execute_backend_method(
                "mongodb",
                "store_experiment_metadata",
                self.experiment_id,
                {
                    "name": self.experiment_name,
                    "purpose": self.purpose,
                    "tags": self.tags,
                    "start_time": self.start_time.isoformat(),
                    "backends_available": [name for name, backend in self.backends.items() if backend is not None],
                },
            )
            
        # Start GPU monitoring with Prometheus exporter
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring(self.experiment_id, prometheus_port=prometheus_port)
            
        # Perform backend health check
        health_check_passed = self._perform_health_check()
        if strict_mode and not health_check_passed:
            raise RuntimeError("Critical backend health check failed in strict mode")
            
        self.is_active = True
        self._start_async_worker()
        
        print(f"Experiment tracking started: {self.experiment_name}")
        return self

    def end(self) -> None:
        """
        End experiment tracking and finalize all logging.
        
        Performs comprehensive cleanup including:
        - Finalizing result summaries
        - Persisting LeZeA-specific data
        - Stopping monitoring systems
        - Calculating business costs
        - Generating recommendations
        """
        if not self.is_active:
            self.logger.warning("Experiment tracking is not currently active")
            print("Warning: Experiment tracking not active")
            return
            
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Stop monitoring systems
        if self.gpu_monitor:
            try:
                self.gpu_monitor.stop_monitoring()
            except Exception:
                pass
                
        # Finalize and persist result summaries
        self._finalize_result_summaries(log_to_backends=True)
        self._finalize_lezea_summaries()
        
        # Process data usage analytics
        if self.data_usage_logger:
            try:
                self._persist_data_usage_analytics()
            except Exception as exc:
                self.logger.warning(f"Data usage analytics persistence failed: {exc}")
                
        # Process business cost analysis
        if self.cost_model:
            try:
                self._persist_cost_analysis()
            except Exception as exc:
                self.logger.warning(f"Cost analysis persistence failed: {exc}")
                
        # Generate and log final metrics
        final_metrics = {
            "experiment_duration_seconds": duration,
            "total_training_steps": self.training_steps,
            "total_checkpoints": self.checkpoints_saved,
            "total_cost": self.total_cost,
        }
        
        # Add resource utilization summary
        resource_summary = self._compute_final_resource_summary()
        
        if self.backends.get("mlflow"):
            # Log numeric resource metrics
            numeric_resource_metrics: Dict[str, float] = {}
            for scope_key, metrics in resource_summary.get("scopes", {}).items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        numeric_resource_metrics[f"resource/{scope_key}/{metric_name}"] = float(metric_value)
                        
            for metric_name, metric_value in resource_summary.items():
                if isinstance(metric_value, (int, float)):
                    numeric_resource_metrics[metric_name] = float(metric_value)
                    
            if numeric_resource_metrics:
                self._submit_backend_operation("mlflow", "log_metrics", numeric_resource_metrics, step=None)
                
            self._submit_backend_operation("mlflow", "log_dict", resource_summary, artifact_file="resources/summary.json")
            
            try:
                self._execute_backend_method("mlflow", "end_run")
            except Exception:
                pass
                
        # Store comprehensive summary in MongoDB
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_experiment_summary", self.experiment_id, self.get_experiment_summary())
            
        # Ensure all async operations complete before finishing
        self._stop_async_worker()
        
        self.is_active = False
        
        # Display completion summary
        print("Experiment completed successfully!")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Training steps: {self.training_steps}")
        print(f"Checkpoints saved: {self.checkpoints_saved}")
        print(f"Total cost: €{self.total_cost:.2f}")
        
        if self.backends.get("mlflow"):
            try:
                print(f"Results available at: {self.backends['mlflow'].get_run_url()}")
            except Exception:
                pass
                
        # Display actionable recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            print("\nRecommendations for future experiments:")
            for recommendation in recommendations[:3]:
                print(f"  • {recommendation}")

    def _finalize_lezea_summaries(self) -> None:
        """Persist comprehensive LeZeA-specific tracking data."""
        try:
            # Network lineage analysis
            lineage_summary = {
                "total_networks": len(self.network_lineages),
                "max_generation": max((lineage.generation for lineage in self.network_lineages.values()), default=0),
                "avg_modifications": (
                    sum(lineage.modification_count for lineage in self.network_lineages.values()) / 
                    max(len(self.network_lineages), 1)
                ),
                "lineages": {
                    network_id: {
                        "parent_ids": lineage.parent_ids,
                        "generation": lineage.generation,
                        "modification_count": lineage.modification_count,
                        "fitness_score": lineage.fitness_score,
                    }
                    for network_id, lineage in self.network_lineages.items()
                },
            }
            
            # Population evolution analysis
            population_summary = {
                "snapshots_count": len(self.population_history),
                "final_population": self.population_history[-1].__dict__ if self.population_history else None,
                "fitness_progression": [snapshot.avg_fitness for snapshot in self.population_history],
                "diversity_progression": [snapshot.diversity_metric for snapshot in self.population_history],
            }
            
            # Reward flow analysis
            reward_summary = {
                "total_flows": len(self.reward_flows),
                "tasker_to_builder_flows": len([flow for flow in self.reward_flows if flow.source_type == NetworkType.TASKER]),
                "builder_to_tasker_flows": len([flow for flow in self.reward_flows if flow.source_type == NetworkType.BUILDER]),
                "avg_reward_value": sum((flow.reward_value for flow in self.reward_flows), 0.0) / max(len(self.reward_flows), 1),
            }
            
            # Challenge usage analysis
            challenge_summary = {
                "challenges_tracked": len(self.challenge_usage_rates),
                "challenge_statistics": {
                    challenge: {
                        "avg_rate": sum(rates.values()) / max(len(rates), 1),
                        "max_rate": max(rates.values()) if rates else 0.0,
                        "min_rate": min(rates.values()) if rates else 0.0,
                    }
                    for challenge, rates in self.challenge_usage_rates.items()
                },
            }
            
            # Component resource attribution
            resource_attribution_summary = {
                "components_tracked": len(self.component_resources),
                "total_cpu_usage": sum(resources.get("cpu_percent", 0.0) for resources in self.component_resources.values()),
                "total_memory_usage": sum(resources.get("memory_mb", 0.0) for resources in self.component_resources.values()),
            }
            
            # Persist summaries to backends
            if self.backends.get("mlflow"):
                self._submit_backend_operation("mlflow", "log_dict", lineage_summary, artifact_file="lezea/network_lineage_summary.json")
                self._submit_backend_operation("mlflow", "log_dict", population_summary, artifact_file="lezea/population_summary.json")
                self._submit_backend_operation("mlflow", "log_dict", reward_summary, artifact_file="lezea/reward_flow_summary.json")
                self._submit_backend_operation("mlflow", "log_dict", challenge_summary, artifact_file="lezea/challenge_usage_summary.json")
                self._submit_backend_operation("mlflow", "log_dict", resource_attribution_summary, artifact_file="lezea/resource_attribution_summary.json")
                
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "lezea_lineage_summary", **lineage_summary})
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "lezea_population_summary", **population_summary})
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "lezea_reward_summary", **reward_summary})
                
        except Exception as exc:
            self.logger.error(f"Failed to finalize LeZeA summaries: {exc}")

    def _ping_backend(self, name: str, backend_instance: Any) -> Tuple[bool, str]:
        """
        Test backend connectivity and availability.
        
        Args:
            name: Backend name for identification
            backend_instance: Backend instance to test
            
        Returns:
            Tuple of (success_flag, status_message)
        """
        try:
            if backend_instance is None:
                return False, "backend not initialized"
                
            if hasattr(backend_instance, "ping"):
                ping_result = backend_instance.ping()
                return (bool(ping_result), "ping successful")
                
            if hasattr(backend_instance, "available"):
                availability = getattr(backend_instance, "available")
                return (bool(availability), "availability check passed")
                
            # Default assumption for backends without explicit health checks
            return True, "assumed operational"
            
        except Exception as exc:
            return False, str(exc)

    def _perform_health_check(self) -> bool:
        """
        Perform comprehensive health check on all configured backends.
        
        Returns:
            True if all backends are healthy, False if any critical backends fail
        """
        health_results = {}
        
        for backend_name, backend_instance in self.backends.items():
            is_healthy, status_message = self._ping_backend(backend_name, backend_instance)
            health_results[backend_name] = (is_healthy, status_message)
        
        # Display health check matrix
        print("\nBackend Health Check:")
        all_healthy = True
        
        for backend_name, (is_healthy, status_message) in health_results.items():
            all_healthy &= is_healthy
            status_icon = "✓" if is_healthy else "✗"
            print(f"  {status_icon} {backend_name:10s} - {status_message}")
            
        return all_healthy

    def log_lezea_config(
        self,
        tasker_population_size: int,
        builder_population_size: int,
        algorithm_type: str,
        start_network_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        seeds: Optional[Dict[str, int]] = None,
        initialization_scheme: Optional[str] = None,
    ) -> None:
        """
        Log LeZeA-specific experiment configuration.

        Args:
            tasker_population_size: Number of tasker networks in population
            builder_population_size: Number of builder networks in population
            algorithm_type: Type of evolutionary algorithm being used
            start_network_id: Initial network ID if continuing from checkpoint
            hyperparameters: Algorithm-specific hyperparameters
            seeds: Random seeds for reproducibility
            initialization_scheme: Network initialization strategy
        """
        if not self.is_active:
            print("Warning: Experiment not active. Call start() first.")
            return
            
        self.lezea_config = {
            "tasker_population_size": tasker_population_size,
            "builder_population_size": builder_population_size,
            "algorithm_type": algorithm_type,
            "start_network_id": start_network_id,
            "hyperparameters": hyperparameters or {},
            "seeds": seeds or {},
            "initialization_scheme": initialization_scheme or "",
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            if self.backends.get("mlflow"):
                # Log basic parameters for MLflow UI
                mlflow_params = {
                    "tasker_pop_size": tasker_population_size,
                    "builder_pop_size": builder_population_size,
                    "algorithm_type": algorithm_type,
                    "start_network_id": start_network_id or "none",
                    "init_scheme": initialization_scheme or "",
                }
                
                # Add hyperparameters with prefix
                if hyperparameters:
                    mlflow_params.update({f"hp_{key}": value for key, value in hyperparameters.items()})
                    
                # Add seeds with prefix
                if seeds:
                    mlflow_params.update({f"seed_{key}": value for key, value in seeds.items()})
                    
                # Log parameters synchronously for immediate availability
                self._submit_backend_operation("mlflow", "log_params", mlflow_params, async_allowed=False)
                
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_lezea_config", self.experiment_id, self.lezea_config)
                
            print(f"LeZeA configuration logged: {tasker_population_size} taskers, {builder_population_size} builders")
            
        except Exception as exc:
            self.logger.error(f"Failed to log LeZeA configuration: {exc}")
            print(f"Error: Failed to log LeZeA configuration: {exc}")

    def register_network(
        self,
        network_id: str,
        network_type: NetworkType,
        parent_ids: Optional[List[str]] = None,
        generation: int = 0,
        layer_configs: Optional[List[LayerSeedConfig]] = None
    ) -> None:
        """
        Register a new network in the genealogy tracking system.

        Args:
            network_id: Unique identifier for the network
            network_type: Type of network (TASKER, BUILDER, or HYBRID)
            parent_ids: IDs of parent networks for inheritance tracking
            generation: Generation number in evolutionary process
            layer_configs: Layer-specific initialization configurations
        """
        if not self.is_active:
            return

        lineage = NetworkLineage(
            network_id=network_id,
            parent_ids=parent_ids or [],
            generation=generation,
            creation_time=datetime.now(),
            modification_count=0
        )
        self.network_lineages[network_id] = lineage

        # Store layer-level seed configurations
        if layer_configs:
            for layer_config in layer_configs:
                config_key = f"{network_id}_{layer_config.layer_id}"
                self.layer_seeds[config_key] = layer_config

        try:
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "network_registration",
                    "network_id": network_id,
                    "network_type": network_type.value,
                    "parent_ids": parent_ids or [],
                    "generation": generation,
                    "timestamp": datetime.now().isoformat(),
                    "layer_configs": [
                        {
                            "layer_id": config.layer_id,
                            "layer_type": config.layer_type,
                            "seed": config.seed,
                            "initialization_method": config.initialization_method,
                            "parameters": config.parameters
                        } for config in (layer_configs or [])
                    ]
                })

            if self.backends.get("mlflow"):
                registration_data = {
                    "network_id": network_id,
                    "network_type": network_type.value,
                    "parent_ids": parent_ids or [],
                    "generation": generation,
                    "layer_count": len(layer_configs) if layer_configs else 0
                }
                self._submit_backend_operation(
                    "mlflow", 
                    "log_dict", 
                    registration_data, 
                    artifact_file=f"networks/{network_id}_registration.json"
                )

            print(f"Network registered: {network_id} (generation {generation})")
            
        except Exception as exc:
            self.logger.error(f"Failed to register network: {exc}")

    def update_network_fitness(self, network_id: str, fitness_score: float) -> None:
        """
        Update fitness score for a registered network.

        Args:
            network_id: ID of the network to update
            fitness_score: New fitness score
        """
        if network_id in self.network_lineages:
            self.network_lineages[network_id].fitness_score = fitness_score

    def track_network_modification(self, network_id: str, modification_type: str, details: Dict[str, Any]) -> None:
        """
        Track modifications made to a network during evolution.

        Args:
            network_id: ID of the modified network
            modification_type: Type of modification (e.g., 'mutation', 'crossover')
            details: Detailed information about the modification
        """
        if network_id in self.network_lineages:
            self.network_lineages[network_id].modification_count += 1

        try:
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "network_modification",
                    "network_id": network_id,
                    "modification_type": modification_type,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as exc:
            self.logger.error(f"Failed to track network modification: {exc}")

    def log_population_snapshot(
        self,
        tasker_count: int,
        builder_count: int,
        generation: int,
        fitness_scores: List[float],
        diversity_metric: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a snapshot of the current population state.

        Args:
            tasker_count: Current number of tasker networks
            builder_count: Current number of builder networks
            generation: Current generation number
            fitness_scores: List of fitness scores for all networks
            diversity_metric: Measure of population diversity
            step: Optional training step number
        """
        if not fitness_scores:
            avg_fitness = best_fitness = worst_fitness = 0.0
        else:
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness = max(fitness_scores)
            worst_fitness = min(fitness_scores)

        snapshot = PopulationSnapshot(
            timestamp=datetime.now(),
            tasker_count=tasker_count,
            builder_count=builder_count,
            generation=generation,
            avg_fitness=avg_fitness,
            best_fitness=best_fitness,
            worst_fitness=worst_fitness,
            diversity_metric=diversity_metric
        )
        self.population_history.append(snapshot)

        try:
            if self.backends.get("mlflow"):
                population_metrics = {
                    "population/tasker_count": tasker_count,
                    "population/builder_count": builder_count,
                    "population/generation": generation,
                    "population/avg_fitness": avg_fitness,
                    "population/best_fitness": best_fitness,
                    "population/worst_fitness": worst_fitness,
                    "population/diversity": diversity_metric
                }
                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(population_metrics), step=step)

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "population_snapshot",
                    **snapshot.__dict__,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "fitness_distribution": {
                        "scores": fitness_scores,
                        "count": len(fitness_scores)
                    }
                })

            print(f"Population snapshot logged: Generation {generation}, {tasker_count}T/{builder_count}B, avg_fitness={avg_fitness:.3f}")
            
        except Exception as exc:
            self.logger.error(f"Failed to log population snapshot: {exc}")

    def log_reward_flow(
        self,
        source_id: str,
        target_id: str,
        source_type: NetworkType,
        target_type: NetworkType,
        reward_value: float,
        task_id: str,
        performance_metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Log reward transfer between network components.

        Args:
            source_id: ID of the network providing the reward
            target_id: ID of the network receiving the reward
            source_type: Type of source network
            target_type: Type of target network
            reward_value: Amount of reward transferred
            task_id: ID of the task associated with this reward
            performance_metrics: Additional performance metrics
            step: Optional training step number
        """
        reward_flow = RewardFlow(
            source_id=source_id,
            target_id=target_id,
            source_type=source_type,
            target_type=target_type,
            reward_value=reward_value,
            task_id=task_id,
            performance_metrics=performance_metrics or {}
        )
        self.reward_flows.append(reward_flow)

        # Update performance tracking for fitness analysis
        if source_type == NetworkType.TASKER:
            self.tasker_performance[source_id][task_id] = reward_value
            if target_id not in self.builder_evaluations:
                self.builder_evaluations[target_id] = {}
            self.builder_evaluations[target_id][source_id] = reward_value

        # Track population fitness evolution
        self.population_fitness[source_id].append(reward_value)

        try:
            if self.backends.get("mlflow"):
                reward_metrics = {
                    f"reward_flow/{source_type.value}_to_{target_type.value}": reward_value
                }
                if performance_metrics:
                    for metric_name, metric_value in performance_metrics.items():
                        reward_metrics[f"reward_flow/performance/{metric_name}"] = metric_value
                        
                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(reward_metrics), step=step)

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "reward_flow",
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_type": source_type.value,
                    "target_type": target_type.value,
                    "reward_value": reward_value,
                    "task_id": task_id,
                    "performance_metrics": performance_metrics or {},
                    "timestamp": datetime.now().isoformat()
                })

            print(f"Reward flow logged: {source_id} → {target_id} ({reward_value:.3f})")
            
        except Exception as exc:
            self.logger.error(f"Failed to log reward flow: {exc}")

    def log_modification_tree(self, step: int, modifications: List[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
        """
        Log modification tree for network evolution analysis.

        Args:
            step: Training step when modifications occurred
            modifications: List of modification operations
            statistics: Statistical summary of modifications
        """
        if not self.is_active:
            return
            
        try:
            # Create modification tree instance
            modification_tree = ModificationTree(step, modifications, statistics)
            self.modification_trees[step] = modification_tree

            # Calculate acceptance statistics
            accepted_count = len([mod for mod in modifications if mod.get("accepted", False)])
            rejected_count = len(modifications) - accepted_count
            acceptance_rate = accepted_count / len(modifications) if modifications else 0.0

            # Enhanced modification statistics
            enhanced_statistics = {
                **statistics,
                "total_modifications": len(modifications),
                "accepted_modifications": accepted_count,
                "rejected_modifications": rejected_count,
                "acceptance_rate": acceptance_rate,
                "modification_types": Counter(mod.get("type", "unknown") for mod in modifications)
            }

            modification_data = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "modifications": modifications,
                "statistics": enhanced_statistics,
                "scope": self._get_current_scope() or {"level": "global", "entity_id": "root"},
            }

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_modification_tree", self.experiment_id, modification_data)

            if self.backends.get("mlflow"):
                # Log numeric statistics as metrics
                numeric_statistics = {
                    f"modifications_{key}": value 
                    for key, value in enhanced_statistics.items() 
                    if isinstance(value, (int, float))
                }
                
                if numeric_statistics:
                    self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(numeric_statistics), step=step)
                    
                self._submit_backend_operation("mlflow", "log_dict", modification_data, artifact_file=f"modifications/step_{step}.json")

            print(f"Modification tree logged: {len(modifications)} changes, {acceptance_rate:.1%} accepted at step {step}")
            
        except Exception as exc:
            self.logger.error(f"Failed to log modification tree: {exc}")
            print(f"Error: Failed to log modification tree: {exc}")

    def log_challenge_usage_rate(
        self,
        challenge_id: str,
        difficulty_level: str,
        usage_rate: float,
        sample_count: int,
        importance_weights: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Log data usage patterns per challenge and difficulty level.

        Args:
            challenge_id: Unique identifier for the challenge
            difficulty_level: Difficulty classification
            usage_rate: Rate at which data is being used
            sample_count: Number of samples processed
            importance_weights: Sample importance weights for curriculum learning
            step: Optional training step number
        """
        self.challenge_usage_rates[challenge_id][difficulty_level] = usage_rate

        # Store sample importance weights for curriculum analysis
        if importance_weights:
            for sample_id, weight in importance_weights.items():
                weight_key = f"{challenge_id}_{sample_id}"
                self.sample_importance_weights[weight_key] = weight

        try:
            if self.backends.get("mlflow"):
                usage_metrics = {
                    f"data_usage/challenge/{challenge_id}/{difficulty_level}/rate": usage_rate,
                    f"data_usage/challenge/{challenge_id}/{difficulty_level}/count": float(sample_count)
                }
                
                if importance_weights:
                    avg_importance = sum(importance_weights.values()) / len(importance_weights)
                    usage_metrics[f"data_usage/challenge/{challenge_id}/{difficulty_level}/avg_importance"] = avg_importance

                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(usage_metrics), step=step)

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "challenge_usage_rate",
                    "challenge_id": challenge_id,
                    "difficulty_level": difficulty_level,
                    "usage_rate": usage_rate,
                    "sample_count": sample_count,
                    "importance_weights": importance_weights or {},
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as exc:
            self.logger.error(f"Failed to log challenge usage rate: {exc}")

    def log_learning_relevance(
        self,
        sample_ids: List[str],
        relevance_scores: Dict[str, float],
        challenge_rankings: Dict[str, int],
        step: Optional[int] = None
    ) -> None:
        """
        Log learning relevance metrics for curriculum learning analysis.

        Args:
            sample_ids: List of sample identifiers
            relevance_scores: Relevance scores per sample
            challenge_rankings: Challenge difficulty rankings
            step: Optional training step number
        """
        try:
            avg_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0.0

            if self.backends.get("mlflow"):
                relevance_metrics = {
                    "learning_relevance/avg_score": avg_relevance,
                    "learning_relevance/sample_count": float(len(sample_ids)),
                    "learning_relevance/high_relevance_count": float(
                        len([score for score in relevance_scores.values() if score > 0.7])
                    )
                }
                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(relevance_metrics), step=step)

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "learning_relevance",
                    "sample_ids": sample_ids,
                    "relevance_scores": relevance_scores,
                    "challenge_rankings": challenge_rankings,
                    "avg_relevance": avg_relevance,
                    "timestamp": datetime.now().isoformat()
                })

            # Update data usage logger with relevance information
            if self.data_usage_logger:
                for sample_id, score in relevance_scores.items():
                    try:
                        self.data_usage_logger.update_relevance(sample_id, score)
                    except Exception:
                        pass

        except Exception as exc:
            self.logger.error(f"Failed to log learning relevance: {exc}")

    def log_component_resources(
        self,
        component_id: str,
        component_type: str,
        cpu_percent: float,
        memory_mb: float,
        gpu_util_percent: Optional[float] = None,
        io_operations: Optional[int] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Log resource usage at the component level for performance analysis.

        Args:
            component_id: Unique identifier for the component
            component_type: Type of component ("builder", "tasker", "algorithm", "network", "layer")
            cpu_percent: CPU utilization percentage
            memory_mb: Memory usage in megabytes
            gpu_util_percent: GPU utilization percentage (optional)
            io_operations: Number of I/O operations (optional)
            step: Optional training step number
        """
        resource_data = {
            "cpu_percent": float(cpu_percent),
            "memory_mb": float(memory_mb),
            "gpu_util_percent": float(gpu_util_percent or 0.0),
            "io_operations": int(io_operations or 0),
            "timestamp": datetime.now().isoformat()
        }

        resource_key = f"{component_type}_{component_id}"
        self.component_resources[resource_key] = resource_data

        try:
            if self.backends.get("mlflow"):
                resource_metrics = {
                    f"resources/{component_type}/{component_id}/cpu_percent": float(cpu_percent),
                    f"resources/{component_type}/{component_id}/memory_mb": float(memory_mb)
                }
                
                if gpu_util_percent is not None:
                    resource_metrics[f"resources/{component_type}/{component_id}/gpu_util"] = float(gpu_util_percent)
                    
                if io_operations is not None:
                    resource_metrics[f"resources/{component_type}/{component_id}/io_ops"] = float(io_operations)

                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(resource_metrics), step=step)

            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "component_resources",
                    "component_id": component_id,
                    "component_type": component_type,
                    **resource_data
                })

        except Exception as exc:
            self.logger.error(f"Failed to log component resources: {exc}")

    def _log_experiment_metadata(self) -> None:
        """Log core experiment metadata to backends."""
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "purpose": self.purpose,
            "start_timestamp": self.start_time.isoformat(),
            "tags": self.tags,
        }
        
        if self.backends.get("mlflow"):
            self._submit_backend_operation("mlflow", "log_params", _flatten_nested_params(metadata), async_allowed=False)
            self._submit_backend_operation("mlflow", "log_dict", metadata, artifact_file="metadata/experiment.json")
            
        if self.backends.get("postgres"):
            try:
                self._submit_backend_operation("postgres", "store_experiment_metadata", metadata)
            except Exception:
                pass

    def _log_environment_info(self) -> None:
        """Log comprehensive environment information."""
        environment_info: Dict[str, Any] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "git_commit": _get_git_commit(),
            "config_profile": getattr(config, "profile", "default"),
        }
        
        # Add environment-specific tags if available
        if self.env_tagger:
            try:
                environment_info.update(self.env_tagger.tags())
            except Exception:
                pass
                
        if self.backends.get("mlflow"):
            # Log key environment parameters
            env_params = {
                "env_platform": environment_info["platform"],
                "env_python": environment_info["python_version"],
                "env_git": environment_info.get("git_commit") or "unavailable",
            }
            self._submit_backend_operation("mlflow", "log_params", env_params, async_allowed=False)
            self._submit_backend_operation("mlflow", "log_dict", environment_info, artifact_file="environment/full_environment.json")
            
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_environment", self.experiment_id, environment_info)

    def log_constraints(
        self,
        runtime_limit_seconds: Optional[int] = None,
        step_episode_limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log experiment constraints and limits.

        Args:
            runtime_limit_seconds: Maximum runtime in seconds
            step_episode_limits: Dictionary of step/episode constraints
        """
        self.constraints = {
            "runtime_limit_seconds": runtime_limit_seconds,
            "step_episode_limits": step_episode_limits or {},
            "constraint_timestamp": datetime.now().isoformat(),
        }
        
        if self.backends.get("mlflow"):
            constraint_params: Dict[str, Any] = {}
            if runtime_limit_seconds is not None:
                constraint_params["runtime_limit_sec"] = int(runtime_limit_seconds)
                
            self._submit_backend_operation("mlflow", "log_params", constraint_params, async_allowed=False)
            self._submit_backend_operation("mlflow", "log_dict", self.constraints, artifact_file="constraints/experiment_constraints.json")
            
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_constraints", self.experiment_id, self.constraints)

    def _get_scope_key(self) -> str:
        """Generate hierarchical scope key from current scope stack."""
        if not self._scope_stack:
            return "global"
        current_scope = self._scope_stack[-1]
        return f"{current_scope['level']}:{current_scope['entity_id']}"

    def _get_current_scope(self) -> Optional[Dict[str, str]]:
        """Get the current scope context."""
        return self._scope_stack[-1] if self._scope_stack else None

    def _add_scope_prefix(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Add scope prefix to metrics for hierarchical organization."""
        numeric_metrics = _extract_numeric_metrics(metrics)
        scope_key = self._get_scope_key()
        
        if scope_key == "global":
            return numeric_metrics
        else:
            return {f"{scope_key}/{metric_name}": metric_value for metric_name, metric_value in numeric_metrics.items()}

    @contextmanager
    def scope(self, level: str, entity_id: str):
        """
        Context manager for hierarchical experiment scoping.
        
        Args:
            level: Scope level (e.g., 'builder', 'tasker', 'algorithm')
            entity_id: Entity identifier within the scope
        """
        self._scope_stack.append({"level": level, "entity_id": entity_id})
        try:
            yield
        finally:
            self._scope_stack.pop()

    def builder_scope(self, builder_id: str):
        """Convenience method for builder-scoped operations."""
        return self.scope("builder", builder_id)

    def tasker_scope(self, tasker_id: str):
        """Convenience method for tasker-scoped operations."""
        return self.scope("tasker", tasker_id)

    def algorithm_scope(self, algorithm_name: str):
        """Convenience method for algorithm-scoped operations."""
        return self.scope("algorithm", algorithm_name)

    def network_scope(self, network_id: str):
        """Convenience method for network-scoped operations."""
        return self.scope("network", network_id)

    def layer_scope(self, layer_id: str):
        """Convenience method for layer-scoped operations."""
        return self.scope("layer", layer_id)

    def log_training_step(
        self,
        step: int,
        metrics: Mapping[str, Any],
        sample_ids: Optional[List[str]] = None,
        data_split: Optional[str] = None,
        modification_path: Optional[List[str]] = None,
        modification_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log training step with comprehensive metrics.

        Args:
            step: Training step number
            metrics: Performance metrics to log
            sample_ids: Sample IDs used in this step
            data_split: Data split used ('train', 'val', 'test')
            modification_path: Path of modifications applied
            modification_statistics: Statistics about modifications
        """
        validate_metrics(metrics)
        self.training_steps = max(self.training_steps, int(step))
        
        # Calculate loss delta if loss metric is present
        step_metrics = dict(metrics)
        if "loss" in metrics:
            try:
                current_loss = float(metrics["loss"])
                if self._last_recorded_loss is not None:
                    step_metrics["delta_loss"] = float(self._last_recorded_loss - current_loss)
                self._last_recorded_loss = current_loss
            except (ValueError, TypeError):
                pass
                
        # Log numeric metrics to MLflow
        if self.backends.get("mlflow"):
            self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(step_metrics), step=step)
            
        # Log detailed record to MongoDB
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                "kind": "training_step",
                "step": int(step),
                "timestamp": datetime.now().isoformat(),
                "metrics": dict(step_metrics),
                "data_split": data_split,
                "sample_ids": sample_ids or [],
                "scope": self._get_current_scope(),
                "modification_path": modification_path or [],
                "modification_statistics": modification_statistics or {},
            })

    def log_data_splits(
        self,
        train_ids: Optional[List[str]] = None,
        validation_ids: Optional[List[str]] = None,
        test_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Log data split information for experiment reproducibility.

        Args:
            train_ids: Training sample identifiers
            validation_ids: Validation sample identifiers
            test_ids: Test sample identifiers
        """
        split_info = {
            "train_size": len(train_ids or []),
            "validation_size": len(validation_ids or []),
            "test_size": len(test_ids or []),
            "split_timestamp": datetime.now().isoformat(),
        }
        
        if self.backends.get("mlflow"):
            self._submit_backend_operation("mlflow", "log_dict", split_info, artifact_file="data/splits.json")
            
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "data_splits", **split_info})
            
        # Update data usage logger with split information
        if self.data_usage_logger:
            try:
                if train_ids:
                    for sample_id in train_ids:
                        self.data_usage_logger.mark_seen(sample_id, split="train")
                if validation_ids:
                    for sample_id in validation_ids:
                        self.data_usage_logger.mark_seen(sample_id, split="validation")
                if test_ids:
                    for sample_id in test_ids:
                        self.data_usage_logger.mark_seen(sample_id, split="test")
            except Exception:
                pass

    def log_data_exposure(
        self,
        sample_id: str,
        split: str,
        challenge_id: Optional[str] = None
    ) -> None:
        """
        Log individual data sample exposure for usage tracking.

        Args:
            sample_id: Unique identifier for the data sample
            split: Data split ('train', 'validation', 'test')
            challenge_id: Optional challenge identifier
        """
        if self.data_usage_logger:
            try:
                self.data_usage_logger.record(sample_id, split=split, challenge=challenge_id)
            except Exception:
                pass

    def _accumulate_resource_metrics(self, scope_key: str, metric_values: Mapping[str, Any]) -> None:
        """
        Accumulate resource metrics for final summary calculation.
        
        Args:
            scope_key: Scope identifier for metric grouping
            metric_values: Dictionary of metric values to accumulate
        """
        resource_dict = self._resource_accumulator.setdefault(scope_key, {})
        for metric_name, metric_value in _extract_numeric_metrics(metric_values).items():
            resource_dict[metric_name] = resource_dict.get(metric_name, 0.0) + float(metric_value)

    def _compute_final_resource_summary(self) -> Dict[str, Any]:
        """
        Compute final resource utilization summary across all components.
        
        Returns:
            Dictionary containing resource utilization summary
        """
        total_cpu_usage = sum(resources.get("cpu_percent", 0.0) for resources in self.component_resources.values())
        total_memory_usage = sum(resources.get("memory_mb", 0.0) for resources in self.component_resources.values())
        
        scope_summaries = {scope_key: dict(metrics) for scope_key, metrics in self._resource_accumulator.items()}
        
        return {
            "total_cpu_percent_accumulated": total_cpu_usage,
            "total_memory_mb_accumulated": total_memory_usage,
            "scope_summaries": scope_summaries,
        }

    def log_checkpoint(
        self,
        file_path: str,
        step: Optional[int] = None,
        checkpoint_role: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model checkpoint with versioning and metadata.

        Args:
            file_path: Path to the checkpoint file
            step: Training step when checkpoint was created
            checkpoint_role: Role of the checkpoint ('model', 'optimizer', 'scheduler')
            metadata: Additional metadata about the checkpoint
        """
        self.checkpoints_saved += 1
        
        checkpoint_info = {
            "file_hash": _compute_file_hash(file_path) if os.path.exists(file_path) else None,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else None,
            "checkpoint_role": checkpoint_role,
            "training_step": step,
            "creation_timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # Upload to S3 if available
        if self.backends.get("s3") and os.path.exists(file_path):
            try:
                s3_object_key = self._execute_backend_method("s3", "upload_file", file_path, prefix=f"{self.experiment_id}/checkpoints/")
                checkpoint_info["s3_object_key"] = s3_object_key
            except Exception as exc:
                self.logger.warning(f"S3 checkpoint upload failed: {exc}")
                
        # Log to MLflow
        if self.backends.get("mlflow"):
            try:
                self._submit_backend_operation("mlflow", "log_artifact", file_path, artifact_path="checkpoints")
            except Exception:
                pass
            self._submit_backend_operation("mlflow", "log_dict", checkpoint_info, artifact_file=f"checkpoints/metadata_{self.checkpoints_saved:04d}.json")
            
        # Record in MongoDB
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "checkpoint", **checkpoint_info})

    def log_results(
        self,
        *,
        tasker_rewards: Optional[Dict[str, float]] = None,
        builder_rewards: Optional[Dict[str, float]] = None,
        action_outputs: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Log experimental results including reward distributions and action outputs.

        Args:
            tasker_rewards: Dictionary of tasker ID to reward mappings
            builder_rewards: Dictionary of builder ID to reward mappings
            action_outputs: Dictionary of action/output data
            step: Optional training step number
        """
        current_scope = self._get_scope_key()
        
        # Accumulate reward statistics
        if tasker_rewards:
            self._tasker_reward_sums[current_scope].update(tasker_rewards)
            self._tasker_reward_counts[current_scope] += 1
            
        if builder_rewards:
            self._builder_reward_sums[current_scope].update(builder_rewards)
            self._builder_reward_counts[current_scope] += 1
            
        # Log to MLflow for real-time monitoring
        if self.backends.get("mlflow"):
            mlflow_metrics: Dict[str, float] = {}
            
            if tasker_rewards:
                for tasker_id, reward_value in tasker_rewards.items():
                    mlflow_metrics[f"results/tasker_rewards/{tasker_id}"] = float(reward_value)
                    
            if builder_rewards:
                for builder_id, reward_value in builder_rewards.items():
                    mlflow_metrics[f"results/builder_rewards/{builder_id}"] = float(reward_value)
                    
            if mlflow_metrics:
                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix(mlflow_metrics), step=step)
                
            if action_outputs:
                self._submit_backend_operation("mlflow", "log_dict", action_outputs, artifact_file=f"results/action_outputs_step_{step or 0}.json")
                
        # Log comprehensive data to MongoDB
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                "kind": "experiment_results",
                "timestamp": datetime.now().isoformat(),
                "scope": self._get_current_scope(),
                "tasker_rewards": tasker_rewards or {},
                "builder_rewards": builder_rewards or {},
                "action_outputs": action_outputs or {},
                "training_step": step,
            })

    def log_business_metrics(
        self,
        resource_cost_eur: Optional[float] = None,
        analysis_comments: Optional[str] = None,
        experiment_conclusion: Optional[str] = None,
        visualization_files: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Log business-related metrics and cost analysis.

        Args:
            resource_cost_eur: Cost of resources used in EUR
            analysis_comments: Commentary on business implications
            experiment_conclusion: Overall experiment conclusion
            visualization_files: Dictionary mapping names to local file paths
        """
        if resource_cost_eur is not None:
            try:
                self.total_cost += float(resource_cost_eur)
            except (ValueError, TypeError):
                pass
                
        business_data = {
            "resource_cost_eur": resource_cost_eur,
            "analysis_comments": analysis_comments,
            "experiment_conclusion": experiment_conclusion,
            "logging_timestamp": datetime.now().isoformat(),
        }
        
        if self.backends.get("mlflow"):
            if resource_cost_eur is not None:
                self._submit_backend_operation("mlflow", "log_metrics", self._add_scope_prefix({"business/cost_eur": resource_cost_eur}))
                
            # Upload visualization files
            if visualization_files:
                for visualization_name, file_path in visualization_files.items():
                    if os.path.exists(file_path):
                        try:
                            self._submit_backend_operation("mlflow", "log_artifact", file_path, artifact_path=f"business/visualizations/{visualization_name}")
                        except Exception:
                            pass
                            
            self._submit_backend_operation("mlflow", "log_dict", business_data, artifact_file="business/metrics_summary.json")
            
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_business_metrics", self.experiment_id, business_data)

    def log_dataset_version(
        self,
        dataset_name: str,
        version_tag: str,
        dataset_path: Optional[str] = None,
        version_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log dataset version information for experiment reproducibility.

        Args:
            dataset_name: Name of the dataset
            version_tag: Version identifier or tag
            dataset_path: Path to the dataset files
            version_metadata: Additional version metadata
        """
        version_data = {
            "dataset_name": dataset_name,
            "version_tag": version_tag,
            "dataset_path": dataset_path,
            "version_metadata": version_metadata or {},
            "version_timestamp": datetime.now().isoformat(),
        }
        
        # Track with DVC if available
        if self.backends.get("dvc") and dataset_path:
            try:
                dvc_tracking_info = self._execute_backend_method("dvc", "track", dataset_path, version=version_tag)
                version_data["dvc_tracking"] = dvc_tracking_info
            except Exception as exc:
                self.logger.warning(f"DVC dataset tracking failed: {exc}")
                
        if self.backends.get("mlflow"):
            self._submit_backend_operation("mlflow", "log_dict", version_data, artifact_file=f"datasets/{dataset_name}_{version_tag}.json")
            
        if self.backends.get("mongodb"):
            self._submit_backend_operation("mongodb", "store_dataset_version", self.experiment_id, version_data)

    def _finalize_result_summaries(self, *, log_to_backends: bool) -> Dict[str, Any]:
        """
        Finalize and aggregate all result summaries across scopes.
        
        Args:
            log_to_backends: Whether to persist summaries to storage backends
            
        Returns:
            Dictionary containing aggregated result summaries
        """
        aggregated_summary: Dict[str, Any] = {"scope_summaries": {}}
        
        # Process tasker reward summaries
        for scope_key, reward_sums in self._tasker_reward_sums.items():
            sample_count = self._tasker_reward_counts.get(scope_key, 0) or 1
            average_rewards = {tasker_id: reward_sum / float(sample_count) for tasker_id, reward_sum in reward_sums.items()}
            aggregated_summary["scope_summaries"].setdefault(scope_key, {})["avg_tasker_rewards"] = average_rewards
            
        # Process builder reward summaries
        for scope_key, reward_sums in self._builder_reward_sums.items():
            sample_count = self._builder_reward_counts.get(scope_key, 0) or 1
            average_rewards = {builder_id: reward_sum / float(sample_count) for builder_id, reward_sum in reward_sums.items()}
            aggregated_summary["scope_summaries"].setdefault(scope_key, {})["avg_builder_rewards"] = average_rewards
            
        if log_to_backends:
            if self.backends.get("mlflow"):
                self._submit_backend_operation("mlflow", "log_dict", aggregated_summary, artifact_file="results/final_aggregated_summary.json")
                
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "final_aggregated_summary", **aggregated_summary})
                
        return aggregated_summary

    def _persist_data_usage_analytics(self) -> None:
        """Persist comprehensive data usage analytics."""
        if not self.data_usage_logger:
            return
            
        try:
            usage_summary = self.data_usage_logger.summary()
            
            if self.backends.get("mlflow"):
                # Log compact summary
                self._submit_backend_operation("mlflow", "log_dict", usage_summary, artifact_file="data_usage/usage_summary.json")
                
                # Export and upload full state
                with tempfile.TemporaryDirectory() as temp_directory:
                    state_export_path = os.path.join(temp_directory, "data_usage_complete_state.json")
                    
                    try:
                        self.data_usage_logger.export_json(state_export_path)
                    except Exception:
                        # Fallback manual export
                        complete_state = {
                            "state": {
                                "split_totals": getattr(self.data_usage_logger, "split_totals", {}),
                                "seen_sample_ids": {key: list(value) for key, value in getattr(self.data_usage_logger, "seen_ids", {}).items()},
                                "exposure_counts": dict(getattr(self.data_usage_logger, "exposures", {})),
                                "relevance_scores": {key: dict(value) for key, value in getattr(self.data_usage_logger, "relevance", {}).items()},
                            },
                            "summary": usage_summary,
                            "export_timestamp": datetime.now().isoformat(),
                        }
                        
                        with open(state_export_path, "w") as export_file:
                            json.dump(complete_state, export_file, indent=2)
                            
                    self._submit_backend_operation("mlflow", "log_artifact", state_export_path, artifact_path="data_usage")
                    
            if self.backends.get("mongodb"):
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {"kind": "data_usage_summary", **usage_summary})
                self._submit_backend_operation("mongodb", "store_results", self.experiment_id, {
                    "kind": "data_usage_state_reference",
                    "artifact_location": "mlflow: data_usage/data_usage_complete_state.json",
                    "export_timestamp": datetime.now().isoformat(),
                })
                
        except Exception as exc:
            self.logger.warning(f"Data usage analytics persistence failed: {exc}")

    def _persist_cost_analysis(self) -> None:
        """Persist business cost analysis and modeling results."""
        if not self.cost_model:
            return
            
        try:
            cost_analysis_summary = self.cost_model.summary()
        except Exception:
            cost_analysis_summary = {}
            
        if self.backends.get("mlflow"):
            self._submit_backend_operation("mlflow", "log_dict", cost_analysis_summary, artifact_file="business/cost_analysis_summary.json")
            
        if self.backends.get("mongodb"):
            try:
                self._submit_backend_operation("mongodb", "store_business_metrics", self.experiment_id, {"cost_analysis": cost_analysis_summary})
            except Exception:
                pass
                
        # Update total cost from model if available
        try:
            model_total_cost = float(cost_analysis_summary.get("total_eur", self.total_cost))
            self.total_cost = model_total_cost
        except (ValueError, TypeError, KeyError):
            pass

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive experiment summary.
        
        Returns:
            Dictionary containing complete experiment summary
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "purpose": self.purpose,
            "start_time": self.start_time.isoformat(),
            "end_time": (self.end_time.isoformat() if self.end_time else None),
            "total_training_steps": self.training_steps,
            "checkpoints_saved": self.checkpoints_saved,
            "total_cost_eur": self.total_cost,
            "lezea_configuration": self.lezea_config,
            "experiment_constraints": self.constraints,
        }

    def get_experiment_data(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive experiment data for analysis.
        
        Returns:
            Dictionary containing all tracked experiment data
        """
        return {
            "experiment_metadata": self.get_experiment_summary(),
            "resource_utilization": self._compute_final_resource_summary(),
            "population_evolution": [snapshot.__dict__ for snapshot in self.population_history],
            "network_genealogy": {network_id: vars(lineage) for network_id, lineage in self.network_lineages.items()},
        }

    def get_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on experiment analysis.
        
        Returns:
            List of recommendation strings
        """
        recommendations: List[str] = []
        
        # Runtime optimization recommendations
        if self.constraints.get("runtime_limit_seconds"):
            recommendations.append("Consider implementing early stopping based on loss convergence to optimize runtime usage.")
            
        # Population tracking recommendations
        if not self.population_history:
            recommendations.append("Log population snapshots regularly to visualize evolutionary dynamics and convergence patterns.")
            
        # Modification tracking recommendations
        if not self.modification_trees:
            recommendations.append("Record modification trees to audit network evolution and identify successful adaptation patterns.")
            
        # Backend optimization recommendations
        if self.backends.get("mlflow") and "mlflow" in self.backend_errors:
            recommendations.append("Resolve MLflow backend issues to enable experiment UI and comprehensive artifact tracking.")
            
        # Cost optimization recommendations
        if self.total_cost > 100.0:  # Arbitrary threshold for demonstration
            recommendations.append("Consider cost optimization strategies given the significant resource expenditure.")
            
        # Data usage recommendations
        if self.data_usage_logger and len(self.challenge_usage_rates) < 3:
            recommendations.append("Increase challenge diversity to improve curriculum learning effectiveness.")
            
        return recommendations