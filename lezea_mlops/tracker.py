# lezea_mlops/tracker.py
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
import io
import json
import math
import time
import uuid
import hashlib
import random
import platform
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union, Iterable
from contextlib import contextmanager
from collections import defaultdict, Counter
from threading import Thread, Event
from queue import Queue, Empty
from enum import Enum

# Configuration management
try:
    from .config import config  # type: ignore
except Exception:
    class _Cfg(dict):
        def get(self, k, d=None):  # type: ignore
            return super().get(k, d)
    config = _Cfg()

# Backend integrations (soft imports)
try:
    from .backends.mlflow_backend import MLflowBackend  # type: ignore
except Exception:
    MLflowBackend = None  # type: ignore

try:
    from .backends.mongodb_backend import MongoBackend  # type: ignore
except Exception:
    MongoBackend = None  # type: ignore

try:
    from .backends.s3_backend import S3Backend  # type: ignore
except Exception:
    S3Backend = None  # type: ignore

try:
    from .backends.postgres_backend import PostgresBackend  # type: ignore
except Exception:
    PostgresBackend = None  # type: ignore

try:
    from .backends.dvc_backend import DVCBackend  # type: ignore
except Exception:
    DVCBackend = None  # type: ignore

# System monitoring components
try:
    from .monitoring.gpu_monitor import GPUMonitor  # type: ignore
except Exception:
    GPUMonitor = None  # type: ignore

try:
    from .monitoring.env_tags import EnvironmentTagger  # type: ignore
except Exception:
    EnvironmentTagger = None  # type: ignore

# Data usage tracking (optional component)
try:
    from .monitoring.data_usage import DataUsageLogger  # type: ignore
except Exception:
    DataUsageLogger = None  # type: ignore

# Core utilities
try:
    from .utils.logging import get_logger  # type: ignore
except Exception:
    import logging
    def get_logger(name="lezea.tracker"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        return logging.getLogger(name)

try:
    from .utils.validation import validate_experiment_name, validate_metrics  # type: ignore
except Exception:
    def validate_experiment_name(name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("Experiment name must be non-empty.")
        return name
    def validate_metrics(metrics: Mapping[str, Union[int, float]]) -> None:
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise ValueError("Metric key must be str")
            if not isinstance(v, (int, float)) or math.isnan(float(v)):
                raise ValueError(f"Metric {k} must be finite number")

# Business cost modeling (optional)
try:
    from .business.cost_model import CostModel  # type: ignore
except Exception:
    CostModel = None  # type: ignore

# LeZeA-specific modification tracking
try:
    from .modification.trees import ModificationTree  # type: ignore
except Exception:
    try:
        from .modification.trees import ModTree as ModificationTree  # type: ignore
    except Exception:
        ModificationTree = None  # type: ignore


# =================== helpers & dataclasses ===================

class NetworkType(Enum):
    TASKER = "tasker"
    BUILDER = "builder"
    HYBRID = "hybrid"

@dataclass
class NetworkLineage:
    network_id: str
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    modification_count: int = 0
    fitness_score: Optional[float] = None

@dataclass
class PopulationSnapshot:
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
    layer_id: str
    layer_type: str
    seed: int
    initialization_method: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardFlow:
    source_id: str
    target_id: str
    source_type: NetworkType
    target_type: NetworkType
    reward_value: float
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

def _compute_file_hash(filepath: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def _flatten(params: Dict[str, Any], max_len: int = 400) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v if len(v) <= max_len else v[: max_len - 3] + "..."
        else:
            try:
                s = json.dumps(v, separators=(",", ":"))
            except Exception:
                s = str(v)
            out[k] = s if len(s) <= max_len else s[: max_len - 3] + "..."
    return out

def _only_numeric(d: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


# =================== tracker ===================

class ExperimentTracker:
    """
    Central, backend-agnostic tracker with:
      - lifecycle + metadata
      - per-step metrics
      - modification trees + stats
      - data usage & relevance
      - per-part resource logging
      - rewards/results
      - checkpoints + datasets
      - business cost + comments + conclusion
    Always writes a local JSONL audit trail.
    """

    # ---------- lifecycle ----------

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
        validate_experiment_name(experiment_name)

        # meta
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.purpose = purpose
        self.tags = tags or {}
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.run_id = self.experiment_id
        self.is_active = False

        # logging + audit trail
        self.logger = get_logger(f"lezea.tracker.{experiment_name}")
        self._audit_dir = Path(config.get("OUTPUT_DIR", "./runs")) / self.experiment_id
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._audit_dir / "events.jsonl"
        self._events = open(self._events_path, "a", encoding="utf-8")

        # backends
        self.backends: Dict[str, Any] = {}
        self.backend_errors: Dict[str, str] = {}
        self._init_backends()

        # monitors
        try:
            self.gpu_monitor = GPUMonitor() if GPUMonitor else None
        except Exception:
            self.gpu_monitor = None
        try:
            self.env_tagger = EnvironmentTagger() if EnvironmentTagger else None
        except Exception:
            self.env_tagger = None

        # state
        self.strict_validation = bool(strict_validation)
        self.async_mode = bool(async_mode)
        self._async_q: "Queue[Tuple[str,str,tuple,dict]]" = Queue()
        self._stop_evt = Event()
        self._worker: Optional[Thread] = None

        # training stats
        self.training_steps = 0
        self._last_loss: Optional[float] = None
        self.checkpoints_saved = 0
        self.total_cost = 0.0

        # lezea structures
        self.network_lineages: Dict[str, NetworkLineage] = {}
        self.population_history: List[PopulationSnapshot] = []
        self.layer_seeds: Dict[str, LayerSeedConfig] = {}
        self.reward_flows: List[RewardFlow] = []
        self.modification_trees: Dict[int, Any] = {}

        # scope + resources
        self._scope_stack: List[Dict[str, str]] = []
        self.component_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._resource_accumulator: Dict[str, Dict[str, float]] = {}

        # rewards/results aggregations
        self._tasker_reward_sums: Dict[str, Counter] = defaultdict(Counter)
        self._tasker_reward_counts: Dict[str, int] = defaultdict(int)
        self._builder_reward_sums: Dict[str, Counter] = defaultdict(Counter)
        self._builder_reward_counts: Dict[str, int] = defaultdict(int)

        # data usage
        self.data_usage_logger = DataUsageLogger() if DataUsageLogger else None
        self.challenge_usage_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.sample_importance_weights: Dict[str, float] = {}

        # optional business model
        self.cost_model = CostModel.from_env() if CostModel else None

        # config/meta caches
        self.lezea_config: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}

        # greet
        print("LeZeA MLOps Experiment Tracker - Ready")
        print(f"Experiment: {self.experiment_name} | ID: {self.experiment_id[:8]} | Mode: {'async' if self.async_mode else 'sync'}")

        if auto_start:
            self.start()

    # ---------- internals ----------

    def _init_backends(self) -> None:
        registry = [
            ("mlflow", MLflowBackend, "MLflow tracking/UI"),
            ("mongodb", MongoBackend, "MongoDB document store"),
            ("s3", S3Backend, "S3 artifact store"),
            ("postgres", PostgresBackend, "PostgreSQL structured store"),
            ("dvc", DVCBackend, "DVC dataset tracking"),
        ]
        for name, cls, desc in registry:
            try:
                if cls:
                    self.backends[name] = cls(config)
                    self.logger.info(f"Backend OK: {name} ({desc})")
                else:
                    self.backends[name] = None
                    self.backend_errors[name] = "module not available"
            except Exception as e:
                self.backends[name] = None
                self.backend_errors[name] = str(e)
                self.logger.warning(f"Backend FAILED: {name} -> {e}")

    def _event(self, kind: str, payload: Mapping[str, Any]) -> None:
        rec = {"ts": datetime.utcnow().isoformat(), "kind": kind, "exp_id": self.experiment_id, **payload}
        try:
            self._events.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._events.flush()
        except Exception:
            pass

    def _exec(self, backend: str, method: str, *args, **kw):
        inst = self.backends.get(backend)
        if not inst:
            raise RuntimeError(f"backend {backend} not available")
        return getattr(inst, method)(*args, **kw)

    def _submit(self, backend: str, method: str, *args, async_allowed: bool = True, **kw):
        if self.async_mode and async_allowed:
            self._async_q.put((backend, method, args, kw))
            return None
        return self._exec(backend, method, *args, **kw)

    def _worker_loop(self):
        while not self._stop_evt.is_set() or not self._async_q.empty():
            try:
                b, m, a, k = self._async_q.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._exec(b, m, *a, **k)
            except Exception as e:
                # if dict payload present, dump to local fallback
                payload = k.get("payload") or (a[0] if a else None)
                if isinstance(payload, dict) and m in {"log_dict"}:
                    self._fallback_artifact("failed_async.json", payload)
                self.logger.warning(f"async {b}.{m} failed: {e}")

    def _fallback_artifact(self, rel: str, data: Dict[str, Any]) -> None:
        try:
            base = Path(config.get("FALLBACK_DIR", "artifacts/local")) / self.experiment_id
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ---------- scope helpers ----------

    def _scope_key(self) -> str:
        if not self._scope_stack:
            return "global"
        s = self._scope_stack[-1]
        return f"{s['level']}:{s['entity_id']}"

    def _add_scope(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        nums = _only_numeric(metrics)
        sk = self._scope_key()
        if sk == "global":
            return nums
        return {f"{sk}/{k}": v for k, v in nums.items()}

    @contextmanager
    def scope(self, level: str, entity_id: str):
        self._scope_stack.append({"level": level, "entity_id": entity_id})
        try:
            yield
        finally:
            self._scope_stack.pop()

    def builder_scope(self, builder_id: str): return self.scope("builder", builder_id)
    def tasker_scope(self, tasker_id: str): return self.scope("tasker", tasker_id)
    def algorithm_scope(self, algo: str): return self.scope("algorithm", algo)
    def network_scope(self, nid: str): return self.scope("network", nid)
    def layer_scope(self, lid: str): return self.scope("layer", lid)

    # ---------- public lifecycle ----------

    def start(self, *, prometheus_port: Optional[int] = 8000, strict: Optional[bool] = None) -> "ExperimentTracker":
        if self.is_active:
            print("Warning: tracker already active")
            return self
        strict_mode = self.strict_validation if strict is None else bool(strict)

        # MLflow experiment/run
        if self.backends.get("mlflow"):
            try:
                self._submit("mlflow", "create_experiment", self.experiment_name, self.experiment_id, async_allowed=False)
                run_id = self._exec("mlflow", "start_run", f"{self.experiment_name}_{self.experiment_id[:8]}", tags=self.tags)
                if isinstance(run_id, str):
                    self.run_id = run_id
            except Exception as e:
                self.logger.warning(f"mlflow start failed: {e}")

        # store experiment meta
        meta = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "purpose": self.purpose,
            "started_at": self.start_time.isoformat(),
            "tags": self.tags,
        }
        self._event("start", meta)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_params", _flatten(meta), async_allowed=False)
            self._submit("mlflow", "log_dict", meta, artifact_file="metadata/experiment.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_experiment_metadata", self.experiment_id, meta)

        # env + git + dataset fingerprints
        self.log_environment_info()
        self.log_git_commit()  # explicit
        # (dataset version is user-triggered via log_dataset_version)

        # GPU monitor
        try:
            if self.gpu_monitor:
                self.gpu_monitor.start_monitoring(self.experiment_id, prometheus_port=prometheus_port)
        except Exception:
            pass

        # health
        ok = self._health_matrix()
        if strict_mode and not ok:
            raise RuntimeError("Critical backend health check failed")

        # async worker
        self._stop_evt.clear()
        if self.async_mode:
            self._worker = Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

        self.is_active = True
        print(f"Experiment tracking started: {self.experiment_name}")
        return self

    def end(self) -> None:
        if not self.is_active:
            print("Warning: tracker not active")
            return
        self.end_time = datetime.utcnow()

        # stop monitors
        try:
            if self.gpu_monitor:
                self.gpu_monitor.stop_monitoring()
        except Exception:
            pass

        # summaries
        self._finalize_result_summaries(log_to_backends=True)
        self._persist_data_usage_analytics()
        self._persist_cost_analysis()

        # final resources summary
        res = self._compute_final_resource_summary()
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", res, artifact_file="resources/summary.json")
            self._submit("mlflow", "log_metrics", _only_numeric(res))
            try:
                self._exec("mlflow", "end_run")
            except Exception:
                pass
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_experiment_summary", self.experiment_id, self.get_experiment_summary())

        # wait for async
        if self._worker:
            self._stop_evt.set()
            self._worker.join(timeout=5)
            self._worker = None

        self.is_active = False
        dur = (self.end_time - self.start_time).total_seconds()
        print("Experiment completed successfully!")
        print(f"Duration: {dur:.1f}s | Steps: {self.training_steps} | Checkpoints: {self.checkpoints_saved} | Cost: €{self.total_cost:.2f}")
        try:
            if self.backends.get("mlflow"):
                print(f"MLflow run: {self.backends['mlflow'].get_run_url()}")
        except Exception:
            pass

    # --------------- Health ---------------

    def _health_matrix(self) -> bool:
        print("\nBackend Health Check:")
        ok_all = True
        for name, inst in self.backends.items():
            try:
                if not inst:
                    raise RuntimeError(self.backend_errors.get(name, "not available"))
                ok = getattr(inst, "ping", lambda: True)()
                ok_all &= bool(ok)
                print(f"  {'✓' if ok else '✗'} {name:10s} - {'ok' if ok else 'unreachable'}")
            except Exception as e:
                ok_all = False
                print(f"  ✗ {name:10s} - {e}")
        return ok_all

    # --------------- Spec: Metadata / Env / Git / Constraints ---------------

    def log_environment_info(self) -> None:
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
        }
        if self.env_tagger:
            try:
                info.update(self.env_tagger.tags())
            except Exception:
                pass
        self._event("env", info)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_params", {
                "env_platform": info["platform"],
                "env_python": info["python_version"],
            }, async_allowed=False)
            self._submit("mlflow", "log_dict", info, artifact_file="environment/full.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_environment", self.experiment_id, info)

    def log_git_commit(self) -> None:
        sha = _get_git_commit()
        if not sha:
            return
        self._event("git", {"commit": sha})
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_params", {"git_commit": sha}, async_allowed=False)

    def log_constraints(self, runtime_limit_seconds: Optional[int] = None, step_episode_limits: Optional[Dict[str, Any]] = None) -> None:
        self.constraints = {
            "runtime_limit_seconds": runtime_limit_seconds,
            "step_episode_limits": step_episode_limits or {},
            "ts": datetime.utcnow().isoformat(),
        }
        self._event("constraints", self.constraints)
        if self.backends.get("mlflow"):
            d = {}
            if runtime_limit_seconds is not None:
                d["runtime_limit_sec"] = int(runtime_limit_seconds)
            self._submit("mlflow", "log_params", d, async_allowed=False)
            self._submit("mlflow", "log_dict", self.constraints, artifact_file="constraints/constraints.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_constraints", self.experiment_id, self.constraints)

    # --------------- Spec: LeZeA Config / Population ---------------

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
        self.lezea_config = {
            "tasker_population_size": tasker_population_size,
            "builder_population_size": builder_population_size,
            "algorithm_type": algorithm_type,
            "start_network_id": start_network_id,
            "hyperparameters": hyperparameters or {},
            "seeds": seeds or {},
            "initialization_scheme": initialization_scheme or "",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._event("lezea_config", self.lezea_config)
        if self.backends.get("mlflow"):
            params = {
                "tasker_pop_size": tasker_population_size,
                "builder_pop_size": builder_population_size,
                "algorithm_type": algorithm_type,
                "start_network_id": start_network_id or "none",
                "init_scheme": initialization_scheme or "",
            }
            if hyperparameters:
                params.update({f"hp_{k}": v for k, v in hyperparameters.items()})
            if seeds:
                params.update({f"seed_{k}": v for k, v in seeds.items()})
            self._submit("mlflow", "log_params", params, async_allowed=False)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_lezea_config", self.experiment_id, self.lezea_config)

    def log_population_snapshot(self, tasker_count: int, builder_count: int, generation: int, fitness_scores: List[float], diversity_metric: float, step: Optional[int] = None) -> None:
        if fitness_scores:
            avg = sum(fitness_scores) / len(fitness_scores)
            best = max(fitness_scores); worst = min(fitness_scores)
        else:
            avg = best = worst = 0.0
        snap = PopulationSnapshot(datetime.utcnow(), tasker_count, builder_count, generation, avg, best, worst, diversity_metric)
        self.population_history.append(snap)
        self._event("population", {
            "generation": generation, "taskers": tasker_count, "builders": builder_count,
            "avg_fitness": avg, "best": best, "worst": worst, "diversity": diversity_metric
        })
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({
                "population/tasker_count": tasker_count,
                "population/builder_count": builder_count,
                "population/generation": generation,
                "population/avg_fitness": avg,
                "population/best_fitness": best,
                "population/worst_fitness": worst,
                "population/diversity": diversity_metric
            }), step=step)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {
                "kind": "population_snapshot",
                "timestamp": snap.timestamp.isoformat(),
                "tasker_count": tasker_count,
                "builder_count": builder_count,
                "generation": generation,
                "avg_fitness": avg,
                "best_fitness": best,
                "worst_fitness": worst,
                "diversity_metric": diversity_metric,
                "fitness_distribution": {"scores": fitness_scores, "count": len(fitness_scores)}
            })

    # --------------- Spec: Training / Metrics / Modifications ---------------

    def log_training_step(self, step: int, metrics: Mapping[str, Any], sample_ids: Optional[List[str]] = None, data_split: Optional[str] = None, modification_path: Optional[List[str]] = None, modification_statistics: Optional[Dict[str, Any]] = None) -> None:
        validate_metrics(metrics)
        self.training_steps = max(self.training_steps, int(step))
        m = dict(metrics)
        if "loss" in m:
            try:
                cur = float(m["loss"])
                if self._last_loss is not None:
                    m["delta_loss"] = self._last_loss - cur
                self._last_loss = cur
            except Exception:
                pass
        self._event("step", {"step": step, "metrics": m, "split": data_split, "samples": sample_ids or []})
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope(m), step=step)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {
                "kind": "training_step",
                "step": int(step),
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": m,
                "data_split": data_split,
                "sample_ids": sample_ids or [],
                "scope": self._scope_stack[-1] if self._scope_stack else {"level": "global", "entity_id": "root"},
                "modification_path": modification_path or [],
                "modification_statistics": modification_statistics or {},
            })

    def log_modification_tree(self, step: int, modifications: List[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
        if ModificationTree:
            try:
                self.modification_trees[step] = ModificationTree(step, modifications, statistics)  # type: ignore
            except Exception:
                pass
        accepted = sum(1 for m in modifications if m.get("accepted"))
        total = len(modifications)
        enhanced = dict(statistics)
        enhanced.update({
            "total_modifications": total,
            "accepted_modifications": accepted,
            "rejected_modifications": total - accepted,
            "acceptance_rate": (accepted / total) if total else 0.0,
            "modification_types": dict(Counter(m.get("type", "unknown") for m in modifications))
        })
        payload = {"step": step, "timestamp": datetime.utcnow().isoformat(), "modifications": modifications, "statistics": enhanced}
        self._event("mod_tree", {"step": step, **enhanced})
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({k: v for k, v in enhanced.items() if isinstance(v, (int, float))}), step=step)
            self._submit("mlflow", "log_dict", payload, artifact_file=f"modifications/step_{step}.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_modification_tree", self.experiment_id, payload)

    # --- Compatibility shims from your FINAL LIST ---
    def log_modification_tree_stats(self, tree: Dict[str, Any]) -> None:
        """Accept an already-aggregated tree stats dict and store it."""
        stats = {k: v for k, v in tree.items() if isinstance(v, (int, float))}
        self._event("mod_tree_stats", tree)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({f"modstats/{k}": v for k, v in stats.items()}))
            self._submit("mlflow", "log_dict", tree, artifact_file="modifications/summary.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "modification_stats", **tree})

    def log_modification_stats(self, stats_dict: Dict[str, Any]) -> None:
        """Alias to store arbitrary modification stats."""
        self.log_modification_tree_stats(stats_dict)

    # --------------- Spec: Data usage / relevance ---------------

    def log_data_splits(self, train_ids: Optional[List[str]] = None, validation_ids: Optional[List[str]] = None, test_ids: Optional[List[str]] = None) -> None:
        info = {
            "train_size": len(train_ids or []),
            "validation_size": len(validation_ids or []),
            "test_size": len(test_ids or []),
            "ts": datetime.utcnow().isoformat()
        }
        self._event("data_splits", info)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", info, artifact_file="data/splits.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "data_splits", **info})
        if self.data_usage_logger:
            try:
                for sid in train_ids or []: self.data_usage_logger.mark_seen(sid, split="train")
                for sid in validation_ids or []: self.data_usage_logger.mark_seen(sid, split="validation")
                for sid in test_ids or []: self.data_usage_logger.mark_seen(sid, split="test")
            except Exception:
                pass

    def log_data_exposure(self, sample_id: str, split: str, challenge_id: Optional[str] = None) -> None:
        if self.data_usage_logger:
            try:
                self.data_usage_logger.record(sample_id, split=split, challenge=challenge_id)
            except Exception:
                pass

    # --- Compatibility shims from your FINAL LIST ---
    def log_data_usage(self, sample_ids: List[str], epoch: int) -> None:
        """Record that these samples were used at this epoch."""
        for sid in sample_ids or []:
            self.log_data_exposure(sid, split="train")
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({"data/epoch_used": float(epoch), "data/samples_used": float(len(sample_ids or []))}), step=epoch)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "data_usage", "epoch": epoch, "sample_ids": sample_ids, "ts": datetime.utcnow().isoformat()})

    def log_data_score(self, sample_id: str, before_loss: float, after_loss: float) -> None:
        """Log improvement score for a sample (before/after)."""
        try:
            delta = float(before_loss) - float(after_loss)
        except Exception:
            delta = 0.0
        self._event("data_score", {"sample_id": sample_id, "before": before_loss, "after": after_loss, "delta": delta})
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({f"data/score/{sample_id}": float(delta)}))
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "data_score", "sample_id": sample_id, "before": before_loss, "after": after_loss, "delta": delta, "ts": datetime.utcnow().isoformat()})

    def log_delta_loss(self, epoch: int, loss: float, prev_loss: float) -> None:
        """Directly log delta loss when both values are known."""
        try:
            delta = float(prev_loss) - float(loss)
        except Exception:
            delta = 0.0
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({"loss": float(loss), "delta_loss": float(delta)}), step=epoch)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "delta_loss", "epoch": epoch, "loss": float(loss), "prev_loss": float(prev_loss), "delta_loss": float(delta), "ts": datetime.utcnow().isoformat()})

    # --------------- Spec: Resources ---------------

    def log_component_resources(self, component_id: str, component_type: str, cpu_percent: float, memory_mb: float, gpu_util_percent: Optional[float] = None, io_operations: Optional[int] = None, step: Optional[int] = None) -> None:
        data = {
            "cpu_percent": float(cpu_percent),
            "memory_mb": float(memory_mb),
            "gpu_util_percent": float(gpu_util_percent or 0.0),
            "io_operations": int(io_operations or 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        key = f"{component_type}_{component_id}"
        self.component_resources[key] = data
        self._accumulate_resource_metrics(self._scope_key(), {
            "cpu_percent": data["cpu_percent"],
            "memory_mb": data["memory_mb"],
            "gpu_util_percent": data["gpu_util_percent"],
        })
        if self.backends.get("mlflow"):
            mets = {
                f"resources/{component_type}/{component_id}/cpu_percent": data["cpu_percent"],
                f"resources/{component_type}/{component_id}/memory_mb": data["memory_mb"],
            }
            if gpu_util_percent is not None:
                mets[f"resources/{component_type}/{component_id}/gpu_util"] = float(gpu_util_percent)
            if io_operations is not None:
                mets[f"resources/{component_type}/{component_id}/io_ops"] = float(io_operations or 0)
            self._submit("mlflow", "log_metrics", self._add_scope(mets), step=step)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "insert_resource", self.experiment_id, component_id, data["cpu_percent"], data["gpu_util_percent"], data["memory_mb"], int(io_operations or 0))

    def _accumulate_resource_metrics(self, scope_key: str, vals: Mapping[str, Any]) -> None:
        bucket = self._resource_accumulator.setdefault(scope_key, {})
        for k, v in _only_numeric(vals).items():
            bucket[k] = bucket.get(k, 0.0) + float(v)

    def _compute_final_resource_summary(self) -> Dict[str, Any]:
        total_cpu = sum(v.get("cpu_percent", 0.0) for v in self.component_resources.values())
        total_mem = sum(v.get("memory_mb", 0.0) for v in self.component_resources.values())
        return {"total_cpu_percent_accumulated": total_cpu, "total_memory_mb_accumulated": total_mem, "scopes": self._resource_accumulator}

    # --- Compatibility shim from your FINAL LIST ---
    def log_resource(self, part_name: str, cpu: float, gpu: float, mem: float) -> None:
        """Map simple resource call to component logging."""
        self.log_component_resources(component_id=part_name, component_type="part", cpu_percent=cpu, memory_mb=mem, gpu_util_percent=gpu)

    def log_resource_cost(self, cpu_hours: float, mem_gb: float, io_bytes: int) -> None:
        """Record resource cost figures (also update total if cost model exists)."""
        payload = {"cpu_hours": float(cpu_hours), "mem_gb": float(mem_gb), "io_bytes": int(io_bytes), "ts": datetime.utcnow().isoformat()}
        self._event("resource_cost", payload)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_metrics", self._add_scope({f"cost/cpu_hours": float(cpu_hours), f"cost/mem_gb": float(mem_gb), f"cost/io_bytes": float(io_bytes)}))
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "resource_cost", **payload})
        # optional link to business cost model
        if self.cost_model:
            try:
                self.total_cost = float(self.cost_model.estimate_total_eur(cpu_hours=cpu_hours, mem_gb=mem_gb, io_bytes=io_bytes))
            except Exception:
                pass

    # --------------- Spec: Rewards / Results ---------------

    def log_results(self, result_dict: Optional[Dict[str, Any]] = None, *, tasker_rewards: Optional[Dict[str, float]] = None, builder_rewards: Optional[Dict[str, float]] = None, action_outputs: Optional[Dict[str, Any]] = None, step: Optional[int] = None) -> None:
        """Unified results logger (accepts either a single dict or granular kwargs)."""
        current_scope = self._scope_key()
        if result_dict:
            # legacy: push numeric metrics and store blob
            if self.backends.get("mlflow"):
                self._submit("mlflow", "log_metrics", self._add_scope({f"results/{k}": v for k, v in _only_numeric(result_dict).items()}), step=step)
                self._submit("mlflow", "log_dict", result_dict, artifact_file=f"results/result_step_{step or 0}.json")
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {"kind": "results_blob", "step": step, **result_dict})
            return

        # granular rewards
        if tasker_rewards:
            self._tasker_reward_sums[current_scope].update(tasker_rewards)
            self._tasker_reward_counts[current_scope] += 1
        if builder_rewards:
            self._builder_reward_sums[current_scope].update(builder_rewards)
            self._builder_reward_counts[current_scope] += 1

        if self.backends.get("mlflow"):
            mets: Dict[str, float] = {}
            for tid, r in (tasker_rewards or {}).items():
                mets[f"results/tasker_rewards/{tid}"] = float(r)
            for bid, r in (builder_rewards or {}).items():
                mets[f"results/builder_rewards/{bid}"] = float(r)
            if mets:
                self._submit("mlflow", "log_metrics", self._add_scope(mets), step=step)
            if action_outputs:
                self._submit("mlflow", "log_dict", action_outputs, artifact_file=f"results/action_outputs_step_{step or 0}.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {
                "kind": "results",
                "timestamp": datetime.utcnow().isoformat(),
                "scope": self._scope_stack[-1] if self._scope_stack else {"level": "global", "entity_id": "root"},
                "tasker_rewards": tasker_rewards or {},
                "builder_rewards": builder_rewards or {},
                "action_outputs": action_outputs or {},
                "step": step,
            })

    # --- Compatibility shims from your FINAL LIST ---
    def log_reward(self, tasker_id: str, reward: float) -> None:
        """Short-hand to log one tasker reward."""
        self.log_results(tasker_rewards={tasker_id: float(reward)})

    # --------------- Spec: Checkpoints / Dataset ---------------

    def log_checkpoint(self, file_path: str, step: Optional[int] = None, checkpoint_role: str = "model", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.checkpoints_saved += 1
        info = {
            "file": file_path,
            "file_hash": _compute_file_hash(file_path),
            "file_size_bytes": (os.path.getsize(file_path) if os.path.exists(file_path) else None),
            "checkpoint_role": checkpoint_role,
            "training_step": step,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._event("checkpoint", info)
        if self.backends.get("s3") and os.path.exists(file_path):
            try:
                s3key = self._exec("s3", "upload_file", file_path, prefix=f"{self.experiment_id}/checkpoints/")
                info["s3_key"] = s3key
            except Exception:
                pass
        if self.backends.get("mlflow") and os.path.exists(file_path):
            self._submit("mlflow", "log_artifact", file_path, artifact_path="checkpoints")
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", info, artifact_file=f"checkpoints/meta_{self.checkpoints_saved:04d}.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "checkpoint", **info})

    def log_dataset_version(self, dataset_name: str, version_tag: str, dataset_path: Optional[str] = None, version_metadata: Optional[Dict[str, Any]] = None) -> None:
        data = {
            "dataset_name": dataset_name,
            "version_tag": version_tag,
            "dataset_path": dataset_path,
            "version_metadata": version_metadata or {},
            "ts": datetime.utcnow().isoformat(),
        }
        if self.backends.get("dvc") and dataset_path:
            try:
                data["dvc"] = self._exec("dvc", "track", dataset_path, version=version_tag)
            except Exception:
                pass
        self._event("dataset_version", data)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", data, artifact_file=f"datasets/{dataset_name}_{version_tag}.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_dataset_version", self.experiment_id, data)

    # --------------- Spec: Business / Comments / Conclusion ---------------

    def log_business_metrics(self, resource_cost_eur: Optional[float] = None, analysis_comments: Optional[str] = None, experiment_conclusion: Optional[str] = None, visualization_files: Optional[Dict[str, str]] = None) -> None:
        if resource_cost_eur is not None:
            try:
                self.total_cost += float(resource_cost_eur)
            except Exception:
                pass
        data = {"resource_cost_eur": resource_cost_eur, "comments": analysis_comments, "conclusion": experiment_conclusion, "ts": datetime.utcnow().isoformat()}
        self._event("business", data)
        if self.backends.get("mlflow"):
            if resource_cost_eur is not None:
                self._submit("mlflow", "log_metrics", self._add_scope({"business/cost_eur": float(resource_cost_eur)}))
            if visualization_files:
                for name, p in visualization_files.items():
                    if os.path.exists(p):
                        self._submit("mlflow", "log_artifact", p, artifact_path=f"business/visuals/{name}")
            self._submit("mlflow", "log_dict", data, artifact_file="business/summary.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_business_metrics", self.experiment_id, data)

    # --- Compatibility shims from your FINAL LIST ---
    def log_comment(self, comment_text: str) -> None:
        """Free-form comment."""
        data = {"comment": str(comment_text), "ts": datetime.utcnow().isoformat()}
        self._event("comment", data)
        if self.backends.get("mongodb"):
            self._submit("mongodb", "insert_comment", self.experiment_id, comment_text)
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", data, artifact_file="notes/comments.json")

    def log_conclusion(self, text: str) -> None:
        """Final conclusion text."""
        self._event("conclusion", {"text": text})
        if self.backends.get("mongodb"):
            self._submit("mongodb", "insert_conclusion", self.experiment_id, text)
        if self.backends.get("mlflow"):
            bio = io.BytesIO(json.dumps({"conclusion": text, "ts": datetime.utcnow().isoformat()}, indent=2).encode("utf-8"))
            try:
                self.backends["mlflow"].log_artifact_stream(bio, "notes/conclusion.json")  # type: ignore
            except Exception:
                self._submit("mlflow", "log_dict", {"conclusion": text}, artifact_file="notes/conclusion.json")

    # --------------- Summaries / Analytics ---------------

    def _finalize_result_summaries(self, *, log_to_backends: bool) -> Dict[str, Any]:
        agg: Dict[str, Any] = {"scope_summaries": {}}
        for sk, sums in self._tasker_reward_sums.items():
            n = self._tasker_reward_counts.get(sk, 0) or 1
            agg["scope_summaries"].setdefault(sk, {})["avg_tasker_rewards"] = {k: v / float(n) for k, v in sums.items()}
        for sk, sums in self._builder_reward_sums.items():
            n = self._builder_reward_counts.get(sk, 0) or 1
            agg["scope_summaries"].setdefault(sk, {})["avg_builder_rewards"] = {k: v / float(n) for k, v in sums.items()}
        if log_to_backends:
            if self.backends.get("mlflow"):
                self._submit("mlflow", "log_dict", agg, artifact_file="results/final_aggregated_summary.json")
            if self.backends.get("mongodb"):
                self._submit("mongodb", "store_results", self.experiment_id, {"kind": "final_aggregated_summary", **agg})
        return agg

    def _persist_data_usage_analytics(self) -> None:
        if not self.data_usage_logger:
            return
        try:
            summary = self.data_usage_logger.summary()
        except Exception:
            summary = {}
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", summary, artifact_file="data_usage/usage_summary.json")
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "data_usage_state.json")
                try:
                    self.data_usage_logger.export_json(p)
                except Exception:
                    with open(p, "w") as f:
                        json.dump({"summary": summary}, f)
                self._submit("mlflow", "log_artifact", p, artifact_path="data_usage")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_results", self.experiment_id, {"kind": "data_usage_summary", **summary})

    def _persist_cost_analysis(self) -> None:
        if not self.cost_model:
            return
        try:
            s = self.cost_model.summary()
        except Exception:
            s = {}
        if self.backends.get("mlflow"):
            self._submit("mlflow", "log_dict", s, artifact_file="business/cost_analysis_summary.json")
        if self.backends.get("mongodb"):
            self._submit("mongodb", "store_business_metrics", self.experiment_id, {"cost_analysis": s})
        try:
            self.total_cost = float(s.get("total_eur", self.total_cost))
        except Exception:
            pass

    # --------------- Public getters ---------------

    def get_experiment_summary(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "purpose": self.purpose,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_training_steps": self.training_steps,
            "checkpoints_saved": self.checkpoints_saved,
            "total_cost_eur": self.total_cost,
            "lezea_configuration": self.lezea_config,
            "experiment_constraints": self.constraints,
        }

    def get_experiment_data(self) -> Dict[str, Any]:
        return {
            "experiment_metadata": self.get_experiment_summary(),
            "resource_utilization": self._compute_final_resource_summary(),
            "population_evolution": [s.__dict__ for s in self.population_history],
            "network_genealogy": {nid: vars(lin) for nid, lin in self.network_lineages.items()},
        }

    def get_recommendations(self) -> List[str]:
        rec: List[str] = []
        if not self.population_history:
            rec.append("Log population snapshots to visualize evolutionary dynamics.")
        if not self.modification_trees:
            rec.append("Record modification trees to audit network evolution.")
        if self.total_cost > 100.0:
            rec.append("Consider cost optimization strategies; cost exceeded €100.")
        return rec

    # --- Compatibility shim from your FINAL LIST ---
    def finalize_experiment(self) -> None:
        """Alias for end()."""
        self.end()
