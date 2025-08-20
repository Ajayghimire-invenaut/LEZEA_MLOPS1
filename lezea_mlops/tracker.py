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
from typing import Any, Dict, List, Optional, Tuple, Set
from contextlib import contextmanager
from collections import defaultdict, Counter
from threading import Thread, Event
from queue import Queue, Empty
import tempfile
from dataclasses import dataclass, field
from enum import Enum

# Config
from .config import config

# Backends
from .backends.mlflow_backend import MLflowBackend
from .backends.mongodb_backend import MongoBackend
from .backends.s3_backend import S3Backend
from .backends.postgres_backend import PostgresBackend
from .backends.dvc_backend import DVCBackend

# Monitoring
from .monitoring.gpu_monitor import GPUMonitor
from .monitoring.env_tags import EnvironmentTagger
try:
    from .monitoring.data_usage import DataUsageLogger
except Exception:
    DataUsageLogger = None  # type: ignore

# Utils
from .utils.logging import get_logger
from .utils.validation import validate_experiment_name, validate_metrics

# Optional cost model (if file exists in your repo)
try:
    from .business.cost_model import CostModel  # noqa
except Exception:  # cost model optional
    CostModel = None  # type: ignore

# New LeZeA-specific components
from .modification.trees import ModificationTree  # Make sure this exists


# ---------------------------
# LeZeA Network Configuration Classes (NEW)
# ---------------------------
class NetworkType(Enum):
    TASKER = "tasker"
    BUILDER = "builder"
    HYBRID = "hybrid"

class PopulationStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EVOLVING = "evolving"
    CONVERGED = "converged"
    TERMINATED = "terminated"

@dataclass
class NetworkLineage:
    """Track network genealogy and inheritance"""
    network_id: str
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    modification_count: int = 0
    fitness_score: Optional[float] = None
    
@dataclass
class PopulationSnapshot:
    """Population state at a point in time"""
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
    """Layer-level seed management"""
    layer_id: str
    layer_type: str
    seed: int
    initialization_method: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardFlow:
    """Tasker ↔ Builder reward tracking"""
    source_id: str
    target_id: str
    source_type: NetworkType
    target_type: NetworkType
    reward_value: float
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Helpers
# ---------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_params(obj: Dict[str, Any], *, max_str: int = 400) -> Dict[str, Any]:
    """
    Flatten nested params to simple key/value pairs for MLflow.
    Lists/dicts are JSON-encoded (truncated) to stay within UI-friendly limits.
    """
    flat: Dict[str, Any] = {}
    for k, v in (obj or {}).items():
        if isinstance(v, (int, float, bool)) or v is None:
            flat[k] = v
        elif isinstance(v, str):
            flat[k] = v if len(v) <= max_str else (v[: max_str - 3] + "...")
        else:
            try:
                s = json.dumps(v, separators=(",", ":"))
                flat[k] = s if len(s) <= max_str else (s[: max_str - 3] + "...")
            except Exception:
                flat[k] = str(v)[:max_str]
    return flat


class ExperimentTracker:
    """Unified tracker for LeZeA experiments (spec-aligned)."""

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def __init__(
        self,
        experiment_name: str,
        purpose: str = "",
        tags: Optional[Dict[str, str]] = None,
        auto_start: bool = False,
        *,
        async_mode: bool = False,
        strict_default: bool = False,
        local_fallback_dir: str = "artifacts/local",
    ) -> None:
        """
        async_mode: queue backend writes off the hot path (best for training loops)
        strict_default: default behavior for start(strict=...) if not provided
        local_fallback_dir: where to dump JSON artifacts if all backends are down
        """
        validate_experiment_name(experiment_name)

        # Core metadata
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.purpose = purpose
        self.tags = tags or {}
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.is_active = False
        self.run_id: str = self.experiment_id  # set to MLflow run_id after start()

        # Logging
        self.logger = get_logger(f"experiment.{experiment_name}")
        self.logger.info(f"Initializing experiment: {experiment_name}")

        # Backends
        self.backends: Dict[str, Any] = {}
        self.backend_errors: Dict[str, str] = {}
        self._init_backends()

        # Monitoring
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.env_tagger: Optional[EnvironmentTagger] = None
        self._init_monitoring()

        # State
        self.lezea_config: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
        self.training_steps: int = 0
        self.checkpoints_saved: int = 0
        self.total_cost: float = 0.0  # manual business cost; auto-cost uses CostModel if available
        self.strict_default = strict_default

        self._step_times: List[float] = []
        self._resource_warnings: List[str] = []
        self._last_loss: Optional[float] = None
        self._scope_stack: List[Dict[str, str]] = []  # each: {level, entity_id}
        self._resource_accum: Dict[str, Dict[str, float]] = {}  # key: scope_key -> metrics
        self._stop_signaled: bool = False  # constraint signal once

        # Results accumulators (compact running aggregates per scope_key)
        self._tasker_rewards_sum: Dict[str, Counter] = defaultdict(Counter)
        self._tasker_rewards_n: Dict[str, int] = defaultdict(int)
        self._builder_rewards_sum: Dict[str, Counter] = defaultdict(Counter)
        self._builder_rewards_n: Dict[str, int] = defaultdict(int)
        self._rl_total_reward: Dict[str, float] = defaultdict(float)
        self._rl_total_steps: Dict[str, int] = defaultdict(int)
        self._rl_episodes_n: Dict[str, int] = defaultdict(int)
        self._rl_action_dist: Dict[str, Counter] = defaultdict(Counter)
        self._cls_n: Dict[str, int] = defaultdict(int)
        self._cls_acc_sum: Dict[str, float] = defaultdict(float)
        self._cls_macro_f1_sum: Dict[str, float] = defaultdict(float)
        self._gen_n: Dict[str, int] = defaultdict(int)
        self._gen_score_sum: Dict[str, float] = defaultdict(float)

        # NEW: LeZeA-specific tracking
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

        # In-memory run context for get_run_data()
        self._model_info: Dict[str, Any] = {}
        self._artifacts_hint: List[str] = []

        # Optional cost model instance (if available)
        self.cost = CostModel.from_env() if CostModel else None  # type: ignore

        # Data usage / learning relevance (optional)
        self.data_usage = DataUsageLogger() if DataUsageLogger else None  # type: ignore

        # Async plumbing
        self.async_mode = bool(async_mode)
        self.local_fallback_dir = local_fallback_dir
        self._q: Queue[Tuple[str, str, tuple, dict]] = Queue()
        self._stop_evt: Event = Event()
        self._worker: Optional[Thread] = None

        print("🚀 LeZeA MLOps Tracker Ready")
        print(f"   Experiment: {experiment_name}")
        print(f"   ID: {self.experiment_id[:8]}...")
        print(f"   Purpose: {purpose}")
        if self.async_mode:
            print("   Mode: async (non-blocking logging)")

        if auto_start:
            self.start()

    # ---------------------------
    # Backend call helpers (retry + optional async)
    # ---------------------------
    def _exec_with_retry(self, backend_name: str, method: str, *args, **kwargs):
        backend = self.backends.get(backend_name)
        if backend is None:
            raise RuntimeError(f"backend '{backend_name}' unavailable")
        attempts = 3
        delay = 0.2
        for i in range(attempts):
            try:
                return getattr(backend, method)(*args, **kwargs)
            except Exception as e:
                if i == attempts - 1:
                    raise
                self.logger.warning(
                    f"Retry {i+1}/{attempts-1} {backend_name}.{method} failed: {e}"
                )
                time.sleep(delay + random.random() * delay)
                delay *= 2

    def _submit(self, backend_name: str, method: str, *args, async_ok: bool = True, **kwargs):
        """Queue or execute a backend call."""
        if self.async_mode and async_ok:
            self._q.put((backend_name, method, args, kwargs))
            return None
        return self._exec_with_retry(backend_name, method, *args, **kwargs)

    def _drain(self):
        while not self._stop_evt.is_set() or not self._q.empty():
            try:
                backend_name, method, args, kwargs = self._q.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._exec_with_retry(backend_name, method, *args, **kwargs)
            except Exception as e:
                # Best-effort fallback for dict-like payloads
                payload = kwargs.get("payload") or (args[0] if args else None)
                if isinstance(payload, dict) and method in {"log_dict"}:
                    rel = kwargs.get("artifact_file") or "failed_async.json"
                    self._fallback_write(rel, payload)
                self.logger.warning(f"Async call dropped {backend_name}.{method}: {e}")

    def _start_async_worker(self):
        if self.async_mode and self._worker is None:
            self._worker = Thread(target=self._drain, daemon=True)
            self._worker.start()

    def _stop_async_worker(self):
        if self._worker:
            self._stop_evt.set()
            self._worker.join(timeout=5)
            self._worker = None
            self._stop_evt.clear()

    def _fallback_write(self, rel_path: str, obj: Dict[str, Any]) -> None:
        try:
            base = Path(self.local_fallback_dir) / self.experiment_id
            p = base / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(obj, f, indent=2)
        except Exception:
            pass

    # ---------------------------
    # Init helpers
    # ---------------------------
    def _init_backends(self) -> None:
        """Initialize all available backends, degrade gracefully."""
        registry: List[Tuple[str, Any, str]] = [
            ("mlflow", MLflowBackend, "Experiment tracking"),
            ("mongodb", MongoBackend, "Complex data storage"),
            ("s3", S3Backend, "Artifact storage"),
            ("postgres", PostgresBackend, "Metadata storage"),
            ("dvc", DVCBackend, "Dataset versioning"),
        ]
        for name, cls, desc in registry:
            try:
                self.backends[name] = cls(config)
                self.logger.info(f"✅ {desc} backend ready")
            except Exception as e:  # pragma: no cover — defensive
                self.backends[name] = None
                self.backend_errors[name] = str(e)
                self.logger.warning(f"❌ {desc} backend failed: {e}")
                print(f"⚠️ {desc} unavailable: {e}")

    def _init_monitoring(self) -> None:
        try:
            self.gpu_monitor = GPUMonitor()
            self.logger.info("✅ GPU monitoring ready")
        except Exception as e:
            self.gpu_monitor = None
            self.logger.warning(f"❌ GPU monitoring failed: {e}")
        try:
            self.env_tagger = EnvironmentTagger()
            self.logger.info("✅ Environment detection ready")
        except Exception as e:
            self.env_tagger = None
            self.logger.warning(f"❌ Environment detection failed: {e}")

    # ---------------------------
    # Start / End
    # ---------------------------
    def start(self, *, prom_port: Optional[int] = 8000, strict: Optional[bool] = None) -> "ExperimentTracker":
        """Start a run. If strict=True, fail when critical backends are down."""
        if self.is_active:
            self.logger.warning("Experiment already active")
            print("⚠️ Experiment already active")
            return self
        strict = self.strict_default if strict is None else bool(strict)
        try:
            # MLflow experiment + run
            if self.backends.get("mlflow"):
                self._exec_with_retry("mlflow", "create_experiment", self.experiment_name, self.experiment_id)
                run_id = self._exec_with_retry(
                    "mlflow",
                    "start_run",
                    f"{self.experiment_name}_{self.experiment_id[:8]}",
                    tags=self.tags,
                )
                if isinstance(run_id, str):
                    self.run_id = run_id
                self._log_experiment_metadata()
                self._log_environment()

            # Mongo: experiment record
            if self.backends.get("mongodb"):
                self._exec_with_retry(
                    "mongodb",
                    "store_experiment_metadata",
                    self.experiment_id,
                    {
                        "name": self.experiment_name,
                        "purpose": self.purpose,
                        "tags": self.tags,
                        "start_time": self.start_time.isoformat(),
                        "backends_available": [k for k, v in self.backends.items() if v is not None],
                    },
                )

            # Start GPU monitor (and its Prometheus exporter) if present
            if self.gpu_monitor:
                self.gpu_monitor.start_monitoring(self.experiment_id, prometheus_port=prom_port)

            # Health check (soft or strict)
            ok = self._health_check()
            if strict and not ok:
                raise RuntimeError("Backend health check failed")

            self.is_active = True
            self._start_async_worker()
            print(f"🎯 Started experiment: {self.experiment_name}")
            return self
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Failed to start experiment: {e}")
            self.logger.error(traceback.format_exc())
            print(f"❌ Failed to start experiment: {e}")
            raise

    def end(self) -> None:
        if not self.is_active:
            self.logger.warning("Experiment not active")
            print("⚠️ Experiment not active")
            return
        try:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            # Stop monitors
            if self.gpu_monitor:
                try:
                    self.gpu_monitor.stop_monitoring()
                except Exception:
                    pass

            # Persist final results summary (per scope)
            _ = self._finalize_results_summary(log_to_backends=True)

            # NEW: Persist LeZeA-specific summaries
            self._finalize_lezea_summaries()

            # Persist data-usage summary + full-state artifact
            if self.data_usage:
                try:
                    summary = self.data_usage.summary()
                    if self.backends.get("mlflow"):
                        # compact summary
                        self._submit("mlflow", "log_dict", summary, artifact_file="data_usage/summary.json")

                        # full state export as an artifact
                        with tempfile.TemporaryDirectory() as tmp:
                            dump_path = os.path.join(tmp, "data_usage_state.json")
                            try:
                                # `export_json` exists in the updated DataUsageLogger
                                self.data_usage.export_json(dump_path)
                            except Exception:
                                # Fallback to embedding the internal state if export_json is unavailable
                                payload = {
                                    "state": {
                                        "split_totals": self.data_usage.split_totals,
                                        "seen_ids": {k: list(v) for k, v in self.data_usage.seen_ids.items()},
                                        "exposures": dict(self.data_usage.exposures),
                                        "relevance": {k: dict(v) for k, v in self.data_usage.relevance.items()},
                                    },
                                    "summary": summary,
                                    "exported_at": datetime.now().isoformat(),
                                }
                                with open(dump_path, "w") as f:
                                    json.dump(payload, f, indent=2)
                            self._submit("mlflow", "log_artifact", dump_path, artifact_path="data_usage")

                    if self.backends.get("mongodb"):
                        self._submit("mongodb", "store_results", self.experiment_id, {"kind": "data_usage_summary", **summary})
                        self._submit(
                            "mongodb",
                            "store_results",
                            self.experiment_id,
                            {
                                "kind": "data_usage_state_meta",
                                "artifact_hint": "mlflow: data_usage/data_usage_state.json",
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                except Exception:
                    pass

            # Persist cost summary (if model available)
            if self.cost:
                try:
                    cost_summary = self.cost.summary()
                except Exception:
                    cost_summary = {}
                if self.backends.get("mlflow"):
                    self._submit("mlflow", "log_dict", cost_summary, artifact_file="business/cost_summary.json")
                if self.backends.get("mongodb"):
                    try:
                        self._submit("mongodb", "store_business_metrics", self.experiment_id, {"cost_model": cost_summary})
                    except Exception:
                        pass
                try:
                    self.total_cost = float(cost_summary.get("total_eur", self.total_cost))
                except Exception:
                    pass

            # Final summaries
            final_metrics = {
                "experiment_duration_seconds": duration,
                "total_training_steps": self.training_steps,
                "total_checkpoints": self.checkpoints_saved,
                "total_cost": self.total_cost,
            }

            # Add resource summary metrics
            res_summary = self._final_resource_summary()
            if self.backends.get("mlflow"):
                # Log numeric resource summary as metrics + full JSON as artifact
                numeric_flat = {}
                for scope_key, vals in res_summary.get("scopes", {}).items():
                    for k, v in vals.items():
                        if isinstance(v, (int, float)):
                            numeric_flat[f"resource/{scope_key}/{k}"] = v
                numeric_flat.update({k: v for k, v in res_summary.items() if isinstance(v, (int, float))})
                self._submit("mlflow", "log_metrics", numeric_flat, step=None)
                self._submit("mlflow", "log_dict", res_summary, artifact_file="resources/summary.json")
                try:
                    self._exec_with_retry("mlflow", "end_run")
                except Exception:
                    pass

            # Store summary in Mongo
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_experiment_summary", self.experiment_id, self.get_experiment_summary())

            # Stop async worker AFTER queuing all submissions
            self._stop_async_worker()

            self.is_active = False
            print("🏁 Experiment completed!")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Training steps: {self.training_steps}")
            print(f"   Checkpoints: {self.checkpoints_saved}")
            print(f"   Total cost: €{self.total_cost:.2f}")

            if self.backends.get("mlflow"):
                try:
                    print(f"   View results: {self.backends['mlflow'].get_run_url()}")
                except Exception:
                    pass

            recs = self.get_recommendations()
            if recs:
                print("\n💡 Recommendations:")
                for r in recs[:3]:
                    print(f"   • {r}")
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Error ending experiment: {e}")
            self.logger.error(traceback.format_exc())
            print(f"❌ Error ending experiment: {e}")

    # ---------------------------
    # NEW: LeZeA-specific finalization
    # ---------------------------
    def _finalize_lezea_summaries(self) -> None:
        """Persist all LeZeA-specific tracking data"""
        try:
            # Network lineage summary
            lineage_summary = {
                "total_networks": len(self.network_lineages),
                "max_generation": max((l.generation for l in self.network_lineages.values()), default=0),
                "avg_modifications": sum(l.modification_count for l in self.network_lineages.values()) / max(len(self.network_lineages), 1),
                "lineages": {nid: {
                    "parent_ids": l.parent_ids,
                    "generation": l.generation,
                    "modification_count": l.modification_count,
                    "fitness_score": l.fitness_score
                } for nid, l in self.network_lineages.items()}
            }
            
            # Population evolution summary
            pop_summary = {
                "snapshots_count": len(self.population_history),
                "final_population": self.population_history[-1].__dict__ if self.population_history else None,
                "fitness_progression": [s.avg_fitness for s in self.population_history],
                "diversity_progression": [s.diversity_metric for s in self.population_history]
            }
            
            # Reward flow summary
            reward_summary = {
                "total_flows": len(self.reward_flows),
                "tasker_to_builder_flows": len([rf for rf in self.reward_flows if rf.source_type == NetworkType.TASKER]),
                "builder_to_tasker_flows": len([rf for rf in self.reward_flows if rf.source_type == NetworkType.BUILDER]),
                "avg_reward_value": sum(rf.reward_value for rf in self.reward_flows) / max(len(self.reward_flows), 1)
            }
            
            # Challenge usage rates summary
            usage_summary = {
                "challenges_tracked": len(self.challenge_usage_rates),
                "challenge_stats": {
                    challenge: {
                        "avg_rate": sum(rates.values()) / max(len(rates), 1),
                        "max_rate": max(rates.values()) if rates else 0,
                        "min_rate": min(rates.values()) if rates else 0
                    } for challenge, rates in self.challenge_usage_rates.items()
                }
            }
            
            # Component resource attribution summary
            resource_summary = {
                "components_tracked": len(self.component_resources),
                "total_cpu_usage": sum(res.get("cpu_percent", 0) for res in self.component_resources.values()),
                "total_memory_usage": sum(res.get("memory_mb", 0) for res in self.component_resources.values())
            }

            # Log to backends
            if self.backends.get("mlflow"):
                self._submit("mlflow", "log_dict", lineage_summary, artifact_file="lezea/network_lineage_summary.json")
                self._submit("mlflow", "log_dict", pop_summary, artifact_file="lezea/population_summary.json")
                self._submit("mlflow", "log_dict", reward_summary, artifact_file="lezea/reward_flow_summary.json")
                self._submit("mlflow", "log_dict", usage_summary, artifact_file="lezea/challenge_usage_summary.json")
                self._submit("mlflow", "log_dict", resource_summary, artifact_file="lezea/resource_attribution_summary.json")
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {"kind": "lezea_lineage_summary", **lineage_summary})
                self._submit("mongodb", "store_results", self.experiment_id, {"kind": "lezea_population_summary", **pop_summary})
                self._submit("mongodb", "store_results", self.experiment_id, {"kind": "lezea_reward_summary", **reward_summary})
                
        except Exception as e:
            self.logger.error(f"Failed to finalize LeZeA summaries: {e}")

    # ---------------------------
    # Health Check
    # ---------------------------
    def _ping_backend(self, name: str, obj: Any) -> Tuple[bool, str]:
        try:
            if obj is None:
                return False, "unavailable"
            if hasattr(obj, "ping"):
                ok = obj.ping()
                return (bool(ok), "ok")
            if hasattr(obj, "available"):
                return (bool(getattr(obj, "available")), "available flag")
            # Fallback: assume OK
            return True, "assumed ok"
        except Exception as e:  # pragma: no cover
            return False, str(e)

    def _health_check(self) -> bool:
        checks = {}
        for name, obj in self.backends.items():
            ok, msg = self._ping_backend(name, obj)
            checks[name] = (ok, msg)
        # Log nice matrix
        lines = ["\n🔎 Backend health:"]
        ok_all = True
        for n, (ok, msg) in checks.items():
            ok_all &= bool(ok)
            state = "✅" if ok else "❌"
            lines.append(f"  {state} {n:8s} - {msg}")
        print("\n".join(lines))
        return ok_all

    # ---------------------------
    # NEW: 1.4 LeZeA-specific configuration (ENHANCED)
    # ---------------------------
    def log_lezea_config(
        self,
        tasker_pop_size: int,
        builder_pop_size: int,
        algorithm_type: str,
        start_network_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        seeds: Optional[Dict[str, int]] = None,
        init_scheme: Optional[str] = None,
    ) -> None:
        if not self.is_active:
            print("⚠️ Experiment not active. Call start() first.")
            return
        self.lezea_config = {
            "tasker_population_size": tasker_pop_size,
            "builder_population_size": builder_pop_size,
            "algorithm_type": algorithm_type,
            "start_network_id": start_network_id,
            "hyperparameters": hyperparameters or {},
            "seeds": seeds or {},
            "init_scheme": init_scheme or "",
            "timestamp": datetime.now().isoformat(),
        }
        try:
            if self.backends.get("mlflow"):
                params = {
                    "tasker_pop_size": tasker_pop_size,
                    "builder_pop_size": builder_pop_size,
                    "algorithm_type": algorithm_type,
                    "start_network_id": start_network_id or "none",
                    "init_scheme": init_scheme or "",
                }
                if hyperparameters:
                    params.update({f"hp_{k}": v for k, v in hyperparameters.items()})
                if seeds:
                    params.update({f"seed_{k}": v for k, v in seeds.items()})
                self._submit("mlflow", "log_params", params, async_ok=False)  # params are small; do synchronously
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_lezea_config", self.experiment_id, self.lezea_config)
            print(f"📝 Logged LeZeA config: {tasker_pop_size} taskers, {builder_pop_size} builders")
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Failed to log LeZeA config: {e}")
            print(f"❌ Failed to log LeZeA config: {e}")

    # ---------------------------
    # NEW: Network genealogy & lineage tracking (1.4)
    # ---------------------------
    def register_network(
        self, 
        network_id: str, 
        network_type: NetworkType,
        parent_ids: Optional[List[str]] = None,
        generation: int = 0,
        layer_configs: Optional[List[LayerSeedConfig]] = None
    ) -> None:
        """Register a new network in the genealogy tree"""
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
        
        # Store layer-level seeds if provided
        if layer_configs:
            for layer_config in layer_configs:
                self.layer_seeds[f"{network_id}_{layer_config.layer_id}"] = layer_config
        
        try:
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "network_registration",
                    "network_id": network_id,
                    "network_type": network_type.value,
                    "parent_ids": parent_ids or [],
                    "generation": generation,
                    "timestamp": datetime.now().isoformat(),
                    "layer_configs": [
                        {
                            "layer_id": lc.layer_id,
                            "layer_type": lc.layer_type,
                            "seed": lc.seed,
                            "initialization_method": lc.initialization_method,
                            "parameters": lc.parameters
                        } for lc in (layer_configs or [])
                    ]
                })
            
            if self.backends.get("mlflow"):
                self._submit("mlflow", "log_dict", {
                    "network_id": network_id,
                    "network_type": network_type.value,
                    "parent_ids": parent_ids or [],
                    "generation": generation,
                    "layer_count": len(layer_configs) if layer_configs else 0
                }, artifact_file=f"networks/{network_id}_registration.json")
                
            print(f"🧬 Registered network: {network_id} (gen {generation})")
        except Exception as e:
            self.logger.error(f"Failed to register network: {e}")

    def update_network_fitness(self, network_id: str, fitness_score: float) -> None:
        """Update fitness score for a network"""
        if network_id in self.network_lineages:
            self.network_lineages[network_id].fitness_score = fitness_score
            
    def track_network_modification(self, network_id: str, modification_type: str, details: Dict[str, Any]) -> None:
        """Track modifications made to a network"""
        if network_id in self.network_lineages:
            self.network_lineages[network_id].modification_count += 1
            
        try:
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "network_modification",
                    "network_id": network_id,
                    "modification_type": modification_type,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"Failed to track network modification: {e}")

    # ---------------------------
    # NEW: Population tracking (1.4)
    # ---------------------------
    def log_population_snapshot(
        self,
        tasker_count: int,
        builder_count: int,
        generation: int,
        fitness_scores: List[float],
        diversity_metric: float,
        step: Optional[int] = None
    ) -> None:
        """Log current population state"""
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
                metrics = {
                    "population/tasker_count": tasker_count,
                    "population/builder_count": builder_count,
                    "population/generation": generation,
                    "population/avg_fitness": avg_fitness,
                    "population/best_fitness": best_fitness,
                    "population/worst_fitness": worst_fitness,
                    "population/diversity": diversity_metric
                }
                self._submit("mlflow", "log_metrics", self._prefix_metrics(metrics), step=step)
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "population_snapshot",
                    **snapshot.__dict__,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "fitness_distribution": {
                        "scores": fitness_scores,
                        "count": len(fitness_scores)
                    }
                })
                
            print(f"👥 Population snapshot: Gen {generation}, {tasker_count}T/{builder_count}B, avg_fit={avg_fitness:.3f}")
        except Exception as e:
            self.logger.error(f"Failed to log population snapshot: {e}")

    # ---------------------------
    # NEW: Reward flow tracking (1.6.1)
    # ---------------------------
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
        """Log reward transfer between taskers and builders"""
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
        
        # Track performance for fitness tracking
        if source_type == NetworkType.TASKER:
            self.tasker_performance[source_id][task_id] = reward_value
            if target_id not in self.builder_evaluations:
                self.builder_evaluations[target_id] = {}
            self.builder_evaluations[target_id][source_id] = reward_value
            
        # Update population fitness tracking
        self.population_fitness[source_id].append(reward_value)
        
        try:
            if self.backends.get("mlflow"):
                metrics = {
                    f"reward_flow/{source_type.value}_to_{target_type.value}": reward_value
                }
                if performance_metrics:
                    for k, v in performance_metrics.items():
                        metrics[f"reward_flow/performance/{k}"] = v
                self._submit("mlflow", "log_metrics", self._prefix_metrics(metrics), step=step)
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
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
                
            print(f"💰 Reward flow: {source_id} → {target_id} ({reward_value:.3f})")
        except Exception as e:
            self.logger.error(f"Failed to log reward flow: {e}")

    # ---------------------------
    # NEW: Enhanced modification trees (1.5.2-1.5.3)
    # ---------------------------
    def log_modification_tree(self, step: int, modifications: List[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
        if not self.is_active:
            return
        try:
            # Create modification tree object
            mod_tree = ModificationTree(step, modifications, statistics)
            self.modification_trees[step] = mod_tree
            
            # Calculate acceptance/rejection stats
            accepted = len([m for m in modifications if m.get("accepted", False)])
            rejected = len(modifications) - accepted
            acceptance_rate = accepted / len(modifications) if modifications else 0
            
            # Enhanced statistics
            enhanced_stats = {
                **statistics,
                "total_modifications": len(modifications),
                "accepted_modifications": accepted,
                "rejected_modifications": rejected,
                "acceptance_rate": acceptance_rate,
                "modification_types": Counter(m.get("type", "unknown") for m in modifications)
            }
            
            payload = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "modifications": modifications,
                "statistics": enhanced_stats,
                "scope": self._current_scope() or {"level": "global", "entity_id": "-"},
            }
            
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_modification_tree", self.experiment_id, payload)
                
            if self.backends.get("mlflow"):
                # Log numeric stats as metrics
                numeric_stats = {f"mod_{k}": v for k, v in enhanced_stats.items() if isinstance(v, (int, float))}
                if numeric_stats:
                    self._submit("mlflow", "log_metrics", self._prefix_metrics(numeric_stats), step=step)
                self._submit("mlflow", "log_dict", payload, artifact_file=f"modifications/step_{step}.json")
                
            print(f"🌳 Logged modification tree: {len(modifications)} changes, {acceptance_rate:.1%} accepted at step {step}")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log modification tree: {e}")
            print(f"❌ Failed to log modification tree: {e}")

    # ---------------------------
    # NEW: Challenge-specific data usage rates (1.5.6)
    # ---------------------------
    def log_challenge_usage_rate(
        self,
        challenge_id: str,
        difficulty_level: str,
        usage_rate: float,
        sample_count: int,
        importance_weights: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ) -> None:
        """Log data usage rates per challenge with difficulty analysis"""
        self.challenge_usage_rates[challenge_id][difficulty_level] = usage_rate
        
        if importance_weights:
            for sample_id, weight in importance_weights.items():
                self.sample_importance_weights[f"{challenge_id}_{sample_id}"] = weight
        
        try:
            if self.backends.get("mlflow"):
                metrics = {
                    f"data_usage/challenge/{challenge_id}/{difficulty_level}/rate": usage_rate,
                    f"data_usage/challenge/{challenge_id}/{difficulty_level}/count": sample_count
                }
                if importance_weights:
                    avg_importance = sum(importance_weights.values()) / len(importance_weights)
                    metrics[f"data_usage/challenge/{challenge_id}/{difficulty_level}/avg_importance"] = avg_importance
                    
                self._submit("mlflow", "log_metrics", self._prefix_metrics(metrics), step=step)
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "challenge_usage_rate",
                    "challenge_id": challenge_id,
                    "difficulty_level": difficulty_level,
                    "usage_rate": usage_rate,
                    "sample_count": sample_count,
                    "importance_weights": importance_weights or {},
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Failed to log challenge usage rate: {e}")

    # ---------------------------
    # NEW: Enhanced learning relevance (1.5.7)
    # ---------------------------
    def log_learning_relevance(
        self,
        sample_ids: List[str],
        relevance_scores: Dict[str, float],
        challenge_rankings: Dict[str, int],
        step: Optional[int] = None
    ) -> None:
        """Log learning relevance with automatic scoring and challenge ranking"""
        try:
            avg_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0
            
            if self.backends.get("mlflow"):
                metrics = {
                    "learning_relevance/avg_score": avg_relevance,
                    "learning_relevance/sample_count": len(sample_ids),
                    "learning_relevance/high_relevance_count": len([s for s in relevance_scores.values() if s > 0.7])
                }
                self._submit("mlflow", "log_metrics", self._prefix_metrics(metrics), step=step)
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "learning_relevance",
                    "sample_ids": sample_ids,
                    "relevance_scores": relevance_scores,
                    "challenge_rankings": challenge_rankings,
                    "avg_relevance": avg_relevance,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Update data usage logger if available
            if self.data_usage:
                for sample_id, score in relevance_scores.items():
                    self.data_usage.update_relevance(sample_id, score)
                    
        except Exception as e:
            self.logger.error(f"Failed to log learning relevance: {e}")

    # ---------------------------
    # NEW: Component-level resource attribution (1.5.4)
    # ---------------------------
    def log_component_resources(
        self,
        component_id: str,
        component_type: str,  # "builder", "tasker", "algorithm", "network", "layer"
        cpu_percent: float,
        memory_mb: float,
        gpu_util_percent: Optional[float] = None,
        io_operations: Optional[int] = None,
        step: Optional[int] = None
    ) -> None:
        """Log resource usage at component level"""
        resource_data = {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_util_percent": gpu_util_percent or 0,
            "io_operations": io_operations or 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.component_resources[f"{component_type}_{component_id}"] = resource_data
        
        try:
            if self.backends.get("mlflow"):
                metrics = {
                    f"resources/{component_type}/{component_id}/cpu_percent": cpu_percent,
                    f"resources/{component_type}/{component_id}/memory_mb": memory_mb
                }
                if gpu_util_percent is not None:
                    metrics[f"resources/{component_type}/{component_id}/gpu_util"] = gpu_util_percent
                if io_operations is not None:
                    metrics[f"resources/{component_type}/{component_id}/io_ops"] = io_operations
                    
                self._submit("mlflow", "log_metrics", self._prefix_metrics(metrics), step=step)
                
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {
                    "kind": "component_resources",
                    "component_id": component_id,
                    "component_type": component_type,
                    **resource_data
                })
                
        except Exception as e:
            self.logger.error(f"Failed to log component resources: {e}")