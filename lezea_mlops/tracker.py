"""
LeZeA MLOps Experiment Tracker — FINAL
=====================================

Spec coverage (concise):
- Experiment metadata (IDs, names, purpose, timestamps)
- Environment capture (hardware/firmware when available, software, git)
- Constraints (runtime, steps, episodes)
- LeZeA config (pop sizes, algo, starting nets, seeds, hparams, init scheme)
- Per-epoch/step metrics with scoping (builder/tasker/algorithm/network/layer)
- Delta metrics (e.g., delta_loss)
- Modification trees + stats (numeric stats → metrics; full tree via Mongo/artifacts)
- Data splits + dataset version (via DVC if available; fallback marker)
- Resource usage summaries (per-scope + overall) + optional cost model hooks
- Checkpoints + final models
- Results (tasker/builder rewards, RL episodes, classification metrics, generation outputs) + summary
- Business metrics (manual cost/comments/conclusion)
- Data-usage rate & learning relevance attribution

Backends degrade gracefully if unavailable.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from collections import defaultdict, Counter

# Optional Prometheus HTTP exporter
try:
    from prometheus_client import start_http_server as _prom_start
except Exception:  # pragma: no cover
    _prom_start = None  # type: ignore

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
    ) -> None:
        validate_experiment_name(experiment_name)

        # Core metadata
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.purpose = purpose
        self.tags = tags or {}
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.is_active = False

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

        self._step_times: List[float] = []
        self._resource_warnings: List[str] = []
        self._last_loss: Optional[float] = None
        self._scope_stack: List[Dict[str, str]] = []  # each: {level, entity_id}
        self._resource_accum: Dict[str, Dict[str, float]] = {}  # key: scope_key -> metrics

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

        # Optional cost model instance (if available)
        self.cost = CostModel.from_env() if CostModel else None  # type: ignore

        # Data usage / learning relevance (optional)
        self.data_usage = DataUsageLogger() if DataUsageLogger else None  # type: ignore

        print("🚀 LeZeA MLOps Tracker Ready")
        print(f"   Experiment: {experiment_name}")
        print(f"   ID: {self.experiment_id[:8]}...")
        print(f"   Purpose: {purpose}")

        if auto_start:
            self.start()

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
    def start(self, *, prom_port: Optional[int] = 8000, strict: bool = False) -> "ExperimentTracker":
        """Start a run. If strict=True, fail when critical backends are down."""
        if self.is_active:
            self.logger.warning("Experiment already active")
            print("⚠️ Experiment already active")
            return self
        try:
            # MLflow experiment + run
            if self.backends.get("mlflow"):
                self.backends["mlflow"].create_experiment(self.experiment_name, self.experiment_id)
                self.backends["mlflow"].start_run(
                    f"{self.experiment_name}_{self.experiment_id[:8]}", tags=self.tags
                )
                self._log_experiment_metadata()
                self._log_environment()

            # Mongo: experiment record
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_experiment_metadata(
                    self.experiment_id,
                    {
                        "name": self.experiment_name,
                        "purpose": self.purpose,
                        "tags": self.tags,
                        "start_time": self.start_time.isoformat(),
                        "backends_available": [k for k, v in self.backends.items() if v is not None],
                    },
                )

            # Start Prometheus exporter (if requested and library present)
            if prom_port and _prom_start:
                try:
                    _prom_start(prom_port)
                    self.logger.info(f"Prometheus metrics on :{prom_port}")
                    print(f"📡 Prometheus metrics on :{prom_port}")
                except Exception as e:
                    self.logger.warning(f"Prometheus exporter not started: {e}")
                    if strict:
                        raise

            # Start GPU monitor if present
            if self.gpu_monitor:
                self.gpu_monitor.start_monitoring(self.experiment_id)

            # Health check (soft or strict)
            ok = self._health_check()
            if strict and not ok:
                raise RuntimeError("Backend health check failed")

            self.is_active = True
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

            # Persist data-usage summary
            if self.data_usage:
                try:
                    summary = self.data_usage.summary()
                    if self.backends.get("mlflow"):
                        self.backends["mlflow"].log_dict(summary, "data_usage/summary.json")
                    if self.backends.get("mongodb"):
                        self.backends["mongodb"].store_results(
                            self.experiment_id, {"kind": "data_usage_summary", **summary}
                        )
                except Exception:
                    pass

            # Persist cost summary (if model available)
            if self.cost:
                cost_summary = self.cost.summary()
                if self.backends.get("mlflow"):
                    self.backends["mlflow"].log_dict(cost_summary, "business/cost_summary.json")
                if self.backends.get("mongodb"):
                    try:
                        self.backends["mongodb"].store_business_metrics(self.experiment_id, {"cost_model": cost_summary})
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
                numeric_flat.update({
                    k: v for k, v in res_summary.items() if isinstance(v, (int, float))
                })
                final_metrics.update(numeric_flat)
                self.backends["mlflow"].log_metrics(final_metrics)
                self.backends["mlflow"].log_dict(res_summary, "resources/summary.json")
                self.backends["mlflow"].end_run()

            # Store summary in Mongo
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_experiment_summary(self.experiment_id, self.get_experiment_summary())

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
    # Spec: 1.4 LeZeA configuration
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
                self.backends["mlflow"].log_params(params)
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_lezea_config(self.experiment_id, self.lezea_config)
            print(f"📝 Logged LeZeA config: {tasker_pop_size} taskers, {builder_pop_size} builders")
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Failed to log LeZeA config: {e}")
            print(f"❌ Failed to log LeZeA config: {e}")

    # ---------------------------
    # Spec: 1.3 Constraints
    # ---------------------------
    def log_constraints(
        self,
        max_runtime: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        if not self.is_active:
            self.logger.warning("Experiment not active")
            return
        self.constraints = {
            "max_runtime_seconds": max_runtime,
            "max_steps": max_steps,
            "max_episodes": max_episodes,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            params: Dict[str, Any] = {}
            if self.backends.get("mlflow"):
                params = {k: v for k, v in self.constraints.items() if v is not None and k != "timestamp"}
                if params:
                    self.backends["mlflow"].log_params(params)
            print(f"⏱️ Logged constraints: {params if params else '{}'}")
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Failed to log constraints: {e}")
            print(f"❌ Failed to log constraints: {e}")

    # ---------------------------
    # Scoping helpers
    # ---------------------------
    def _current_scope(self) -> Optional[Dict[str, str]]:
        return self._scope_stack[-1] if self._scope_stack else None

    def _scope_key(self) -> str:
        sc = self._current_scope()
        if not sc:
            return "global"
        return f"{sc['level']}:{sc['entity_id']}"

    def _prefix_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        sc = self._current_scope()
        if not sc:
            return metrics
        prefix = f"{sc['level']}/{sc['entity_id']}"
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    @contextmanager
    def scope(self, level: str, entity_id: str, extra_tags: Optional[Dict[str, str]] = None):
        """Attribute subsequent logs to a specific LeZeA level."""
        scope_info = {"level": level, "entity_id": entity_id}
        self._scope_stack.append(scope_info)
        try:
            if self.backends.get("mlflow"):
                try:
                    self.backends["mlflow"].set_tags({"scope_level": level, "scope_entity": entity_id, **(extra_tags or {})})
                except Exception:
                    pass
            yield
        finally:
            self._scope_stack.pop()

    # ---------------------------
    # Training — metrics & delta + resource/cost sampling + data-usage
    # ---------------------------
    def log_training_step(
        self,
        step: int,
        episode: Optional[int] = None,
        sample_ids: Optional[List[str]] = None,
        split: Optional[str] = None,
        **metrics: Any
    ) -> None:
        if not self.is_active:
            self.logger.warning("Experiment not active")
            return
        t0 = time.time()
        try:
            validated = validate_metrics(metrics)

            # Delta loss (1.5.7.1)
            if "loss" in validated and isinstance(validated["loss"], (int, float)):
                if self._last_loss is not None:
                    validated["delta_loss"] = float(validated["loss"]) - float(self._last_loss)
                self._last_loss = float(validated["loss"])  # update

            # Data-usage & learning relevance (optional)
            if self.data_usage and sample_ids and split:
                try:
                    self.data_usage.update(sample_ids, split, delta_loss=validated.get("delta_loss"))
                    usage_metrics = self.data_usage.get_split_metrics(split)
                    if self.backends.get("mlflow"):
                        prefixed_usage = self._prefix_metrics({f"data/usage/{split}/{k}": v for k, v in usage_metrics.items()})
                        self.backends["mlflow"].log_metrics(prefixed_usage, step=step)
                    # Occasionally persist top-k attribution
                    if self.backends.get("mlflow") and (step % 200 == 0 or step < 10):
                        topk = self.data_usage.top_k(split, k=25)
                        self.backends["mlflow"].log_dict(
                            {"split": split, "step": step, "top_relevant": topk},
                            f"data_usage/{split}_top_relevant_step_{step}.json"
                        )
                except Exception:
                    pass

            # Step timing
            if self._step_times:
                validated["step_duration_seconds"] = t0 - self._step_times[-1]
            self._step_times.append(t0)
            self.training_steps += 1

            # Scope-aware metrics (prefix for MLflow)
            prefixed = self._prefix_metrics(validated)

            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_metrics(prefixed, step=step)
                if episode is not None:
                    ep_key = self._prefix_metrics({"episode": episode})
                    k = list(ep_key.keys())[0]
                    self.backends["mlflow"].log_metric(k, episode, step=step)

            if self.backends.get("mongodb"):
                doc = {
                    "step": step,
                    "episode": episode,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": validated,
                    "scope": self._current_scope() or {"level": "global", "entity_id": "-"},
                    "experiment_id": self.experiment_id,
                    "data_usage": {
                        "split": split,
                        "batch_count": len(sample_ids) if sample_ids else 0,
                    } if split else None,
                }
                self.backends["mongodb"].store_training_step(self.experiment_id, doc)

            # Resource + optional cost sampling
            usage = self._sample_resource_usage()
            self._update_cost_model(validated.get("step_duration_seconds"), usage)

            self._check_performance_warnings(step, validated)

            if step % 100 == 0 or step < 10:
                msg = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in validated.items()
                )
                print(f"📊 Step {step}: {msg}")
        except Exception as e:  # pragma: no cover — defensive
            self.logger.error(f"Failed to log training step {step}: {e}")
            print(f"❌ Failed to log step {step}: {e}")

    # ---------------------------
    # Modifications (1.5.2–1.5.3)
    # ---------------------------
    def log_modification_tree(self, step: int, modifications: List[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
        if not self.is_active:
            return
        try:
            payload = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "modifications": modifications,
                "statistics": statistics,
                "scope": self._current_scope() or {"level": "global", "entity_id": "-"},
            }
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_modification_tree(self.experiment_id, payload)
            if self.backends.get("mlflow") and statistics:
                numeric = {f"mod_{k}": v for k, v in statistics.items() if isinstance(v, (int, float))}
                if numeric:
                    self.backends["mlflow"].log_metrics(self._prefix_metrics(numeric), step=step)
            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_dict(payload, f"modifications/step_{step}.json")
            print(f"🌳 Logged modification tree: {len(modifications)} changes at step {step}")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log modification tree: {e}")
            print(f"❌ Failed to log modification tree: {e}")

    # ---------------------------
    # Data splits & dataset version (1.5.5, §2)
    # ---------------------------
    def log_data_splits(self, train: int, val: int, test: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_active:
            return
        info = {"train": train, "val": val, "test": test, **(extra or {})}
        try:
            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_params({"split_train": train, "split_val": val, "split_test": test})
                self.backends["mlflow"].log_dict(info, "data_splits.json")
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_data_splits(self.experiment_id, info)
            # Feed totals to data-usage logger if enabled
            if self.data_usage:
                try:
                    self.data_usage.set_split_totals({"train": int(train), "val": int(val), "test": int(test)})
                except Exception:
                    pass
            print("🧩 Logged data splits")
        except Exception as e:
            self.logger.error(f"Failed to log data splits: {e}")

    def log_dataset_version(self, dataset_name: str, dataset_root: str = "data/") -> Optional[Dict[str, Any]]:
        """Record dataset version using DVC if available; otherwise create a basic marker."""
        try:
            version_info: Optional[Dict[str, Any]] = None
            dvc = self.backends.get("dvc")
            if dvc and getattr(dvc, "available", False):
                version_info = dvc.get_active_dataset_version(dataset_root)
            else:
                version_info = {"name": dataset_name, "root": dataset_root, "version_tag": "manual"}

            if self.backends.get("mlflow") and version_info:
                self.backends["mlflow"].log_params({f"dataset_{dataset_name}_version": version_info.get("version_tag", "unknown")})
                self.backends["mlflow"].log_dict(version_info, f"datasets/{dataset_name}_version.json")
            if self.backends.get("mongodb") and version_info:
                self.backends["mongodb"].store_dataset_version(self.experiment_id, dataset_name, version_info)
            print(f"📦 Logged dataset version for {dataset_name}")
            return version_info
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log dataset version: {e}")
            print(f"❌ Failed to log dataset version: {e}")
            return None

    # ---------------------------
    # Resource & optional cost (1.5.4 + 1.7.1)
    # ---------------------------
    def _sample_resource_usage(self) -> Dict[str, Any]:
        sc_key = self._scope_key()
        bucket = self._resource_accum.setdefault(sc_key, {"cpu_pct_sum": 0.0, "gpu_mem_mb_sum": 0.0, "samples": 0})
        usage: Dict[str, Any] = {}
        try:
            if self.gpu_monitor and hasattr(self.gpu_monitor, "get_current_usage"):
                usage = self.gpu_monitor.get_current_usage() or {}
            cpu_pct = float(usage.get("cpu_percent", 0.0))
            gpu_mem_mb = float(usage.get("memory_mb", usage.get("gpu_memory_mb", 0.0)))
            bucket["cpu_pct_sum"] += cpu_pct
            bucket["gpu_mem_mb_sum"] += gpu_mem_mb
            bucket["samples"] += 1
        except Exception:
            pass
        return usage

    def _final_resource_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"scopes": {}}
        total_gpu_mb = 0.0
        total_cpu_pct_avg = 0.0
        n = 0
        for scope_key, b in self._resource_accum.items():
            samples = max(1, int(b.get("samples", 0)))
            cpu_avg = float(b.get("cpu_pct_sum", 0.0)) / samples
            gpu_mb_avg = float(b.get("gpu_mem_mb_sum", 0.0)) / samples
            out["scopes"][scope_key] = {"avg_cpu_percent": cpu_avg, "avg_gpu_mem_mb": gpu_mb_avg, "samples": samples}
            total_gpu_mb += gpu_mb_avg
            total_cpu_pct_avg += cpu_avg
            n += 1
        if n:
            out["avg_gpu_mem_mb_overall"] = total_gpu_mb / n
            out["avg_cpu_percent_overall"] = total_cpu_pct_avg / n
        return out

    def _update_cost_model(self, dt: Optional[float], usage: Dict[str, Any]) -> None:
        if not self.cost or not dt or dt <= 0:
            return
        try:
            self.cost.update(
                scope_key=self._scope_key(),
                dt_seconds=float(dt),
                gpu_util=usage.get("util", usage.get("gpu_util_percent", None)),
                gpu_count=usage.get("gpu_count", 1),
                cpu_percent=usage.get("cpu_percent", None),
                cpu_cores=os.cpu_count(),
                ram_gb=usage.get("ram_gb", None),
                io_read_bytes=usage.get("io_read_bytes", 0.0),
                io_write_bytes=usage.get("io_write_bytes", 0.0),
                gpu_mem_mb=usage.get("memory_mb", None),
            )
        except Exception:
            pass

    # ---------------------------
    # Checkpoints & final models (1.5.5–1.5.6)
    # ---------------------------
    def log_checkpoint(self, checkpoint_path: str, step: Optional[int] = None, role: str = "model", metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_active:
            return
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return
        try:
            s3_key = None
            if self.backends.get("s3"):
                s3_key = self.backends["s3"].upload_checkpoint(checkpoint_path, self.experiment_id, step, {"role": role, **(metadata or {})})
            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_artifact(checkpoint_path)
                if step is not None:
                    k = list(self._prefix_metrics({f"{role}_checkpoint_step": step}).keys())[0]
                    self.backends["mlflow"].log_metric(k, step, step=step)
            self.checkpoints_saved += 1
            print(f"💾 Logged checkpoint: {Path(checkpoint_path).name} ({role}) -> S3: {s3_key}")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log checkpoint: {e}")
            print(f"❌ Failed to log checkpoint: {e}")

    def log_final_models(self, tasker_model_path: Optional[str] = None, builder_model_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_active:
            return
        try:
            logged = []
            if tasker_model_path and os.path.exists(tasker_model_path) and self.backends.get("s3"):
                s3_key = self.backends["s3"].upload_final_model(tasker_model_path, self.experiment_id, "tasker", f"final_tasker_{self.experiment_id[:8]}", metadata)
                logged.append(f"tasker -> {s3_key}")
            if builder_model_path and os.path.exists(builder_model_path) and self.backends.get("s3"):
                s3_key = self.backends["s3"].upload_final_model(builder_model_path, self.experiment_id, "builder", f"final_builder_{self.experiment_id[:8]}", metadata)
                logged.append(f"builder -> {s3_key}")
            if logged:
                print(f"🏆 Logged final models: {', '.join(logged)}")
        except Exception as e:
            self.logger.error(f"Failed to log final models: {e}")
            print(f"❌ Failed to log final models: {e}")

    # ---------------------------
    # Results (1.6) — built in
    # ---------------------------
    def log_tasker_rewards(self, rewards: Dict[str, float], *, step: Optional[int] = None) -> None:
        """Log rewards on tasks for Tasker."""
        scope = self._current_scope()
        sk = self._scope_key()
        self._tasker_rewards_n[sk] += 1
        for k, v in (rewards or {}).items():
            if isinstance(v, (int, float)):
                self._tasker_rewards_sum[sk][k] += float(v)
        if self.backends.get("mlflow") and rewards:
            self.backends["mlflow"].log_metrics(self._prefix_metrics({f"tasker_reward/{k}": v for k, v in rewards.items()}), step=step or 0)
            self.backends["mlflow"].log_dict({"rewards": rewards, "scope": scope, "step": step}, f"results/tasker_rewards_{step or 'na'}.json")
        if self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "tasker_rewards", "rewards": rewards, "scope": scope, "step": step, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass

    def log_builder_rewards(self, rewards: Dict[str, float], *, step: Optional[int] = None) -> None:
        """Log rewards of Taskers for Builder."""
        scope = self._current_scope()
        sk = self._scope_key()
        self._builder_rewards_n[sk] += 1
        for k, v in (rewards or {}).items():
            if isinstance(v, (int, float)):
                self._builder_rewards_sum[sk][k] += float(v)
        if self.backends.get("mlflow") and rewards:
            self.backends["mlflow"].log_metrics(self._prefix_metrics({f"builder_reward/{k}": v for k, v in rewards.items()}), step=step or 0)
            self.backends["mlflow"].log_dict({"rewards": rewards, "scope": scope, "step": step}, f"results/builder_rewards_{step or 'na'}.json")
        if self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "builder_rewards", "rewards": rewards, "scope": scope, "step": step, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass

    def log_rl_episode(self, *, episode: int, total_reward: float, steps: int, actions: Optional[List[Any]] = None, action_dist: Optional[Dict[str, int]] = None, step: Optional[int] = None) -> None:
        """Log a single RL episode result and optional action distribution."""
        scope = self._current_scope()
        sk = self._scope_key()
        if action_dist is None and actions is not None:
            c = Counter(str(a) for a in actions)
            action_dist = dict(c)
        self._rl_episodes_n[sk] += 1
        self._rl_total_reward[sk] += float(total_reward)
        self._rl_total_steps[sk] += int(steps)
        if action_dist:
            self._rl_action_dist[sk].update(action_dist)
        # metrics
        metrics = {"episode_reward": float(total_reward), "episode_steps": int(steps)}
        for a, cnt in (action_dist or {}).items():
            metrics[f"actions/{a}"] = float(cnt)
        if self.backends.get("mlflow"):
            self.backends["mlflow"].log_metrics(self._prefix_metrics({f"rl/{k}": v for k, v in metrics.items()}), step=step or episode)
            self.backends["mlflow"].log_dict({"episode": episode, "total_reward": total_reward, "steps": steps, "action_dist": action_dist or {}, "scope": scope}, f"results/rl_episode_{episode}.json")
        if self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "rl_episode", "episode": episode, "total_reward": total_reward, "steps": steps, "action_dist": action_dist or {}, "scope": scope, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass

    def log_classification_results(self, y_true: List[Any], y_pred: List[Any], *, labels: Optional[List[Any]] = None, split: str = "val", step: Optional[int] = None) -> Dict[str, Any]:
        """Compute accuracy/precision/recall/F1 + confusion matrix and log."""
        scope = self._current_scope()
        sk = self._scope_key()
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        y_true_s = list(map(str, y_true))
        y_pred_s = list(map(str, y_pred))
        if labels is None:
            label_set = sorted(set(y_true_s) | set(y_pred_s))
        else:
            label_set = list(map(str, labels))

        # confusion
        conf: Dict[str, Dict[str, int]] = {t: {p: 0 for p in label_set} for t in label_set}
        for t, p in zip(y_true_s, y_pred_s):
            if t not in conf:
                conf[t] = {p2: 0 for p2 in label_set}
            if p not in conf[t]:
                conf[t][p] = 0
            conf[t][p] += 1

        # metrics
        correct = sum(conf[l].get(l, 0) for l in label_set)
        total = len(y_true_s)
        acc = correct / total if total else 0.0
        macro_p = macro_r = macro_f1 = 0.0
        n_labels = len(label_set) if label_set else 1
        for l in label_set:
            tp = conf[l].get(l, 0)
            fp = sum(conf[t].get(l, 0) for t in label_set if t != l)
            fn = sum(conf[l].get(p, 0) for p in label_set if p != l)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
            macro_p += prec
            macro_r += rec
            macro_f1 += f1
        macro_p /= n_labels
        macro_r /= n_labels
        macro_f1 /= n_labels

        self._cls_n[sk] += 1
        self._cls_acc_sum[sk] += acc
        self._cls_macro_f1_sum[sk] += macro_f1

        # log
        if self.backends.get("mlflow"):
            base = f"cls/{split}/"
            self.backends["mlflow"].log_metrics(self._prefix_metrics({f"{base}accuracy": acc, f"{base}macro_precision": macro_p, f"{base}macro_recall": macro_r, f"{base}macro_f1": macro_f1}), step=step or 0)
            self.backends["mlflow"].log_dict({"split": split, "labels": label_set, "accuracy": acc, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1, "confusion": conf, "scope": scope, "step": step}, f"results/classification_{split}_{step or 'na'}.json")
        if self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "classification", "split": split, "accuracy": acc, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1, "confusion": conf, "scope": scope, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass
        return {"accuracy": acc, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1, "confusion": conf, "labels": label_set}

    def log_generation_outputs(self, items: List[Dict[str, Any]], *, name: str = "default", step: Optional[int] = None) -> Dict[str, Any]:
        """Log generation samples; compute average score if provided."""
        scope = self._current_scope()
        sk = self._scope_key()
        scores = [it.get("score") for it in items if isinstance(it.get("score"), (int, float))]
        avg_score = (sum(scores) / len(scores)) if scores else None
        self._gen_n[sk] += len(items)
        if avg_score is not None:
            self._gen_score_sum[sk] += avg_score * len(items)  # weight by count for running mean
        if self.backends.get("mlflow"):
            if avg_score is not None:
                self.backends["mlflow"].log_metrics(self._prefix_metrics({f"gen/{name}/avg_score": float(avg_score)}), step=step or 0)
            self.backends["mlflow"].log_dict({"name": name, "items": items, "avg_score": avg_score, "count": len(items), "scope": scope, "step": step}, f"results/generation_{name}_{step or 'na'}.json")
        if self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "generation", "name": name, "count": len(items), "avg_score": avg_score, "scope": scope, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass
        return {"avg_score": avg_score, "count": len(items)}

    # Backward compatibility: simple results blob
    def log_results(
        self,
        tasker_rewards: Optional[Dict[str, float]] = None,
        builder_rewards: Optional[Dict[str, float]] = None,
        actions_outputs: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        if tasker_rewards:
            self.log_tasker_rewards(tasker_rewards, step=step)
        if builder_rewards:
            self.log_builder_rewards(builder_rewards, step=step)
        if actions_outputs and self.backends.get("mlflow"):
            self.backends["mlflow"].log_dict(actions_outputs, f"results/actions_outputs_{step or 'na'}.json")
        if actions_outputs and self.backends.get("mongodb"):
            try:
                self.backends["mongodb"].store_results(self.experiment_id, {"kind": "actions_outputs", "payload": actions_outputs, "scope": self._current_scope() or {"level": "global", "entity_id": "-"}, "timestamp": datetime.now().isoformat(), "step": step})
            except Exception:
                pass
        print("✅ Logged results")

    # ---------------------------
    # Business (1.7)
    # ---------------------------
    def log_business_metrics(self, cost: float, comments: str = "", conclusion: str = "") -> None:
        if not self.is_active:
            return
        try:
            payload = {
                "cost": float(cost),
                "comments": comments,
                "conclusion": conclusion,
                "timestamp": datetime.now().isoformat(),
            }
            if self.backends.get("mongodb"):
                self.backends["mongodb"].store_business_metrics(self.experiment_id, payload)
            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_metric("total_cost_manual", float(cost))
                note = comments + ("\n\nConclusion: " + conclusion if conclusion else "")
                self.backends["mlflow"].log_text(note, "business/notes.txt")
            self.total_cost += float(cost)
            print(f"💰 Logged business metrics (manual): €{cost:.2f}")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log business metrics: {e}")
            print(f"❌ Failed to log business metrics: {e}")

    # ---------------------------
    # Spec helpers: metadata & environment
    # ---------------------------
    def _log_experiment_metadata(self) -> None:
        try:
            if self.backends.get("mlflow"):
                meta = {
                    "experiment_id": self.experiment_id,
                    "experiment_name": self.experiment_name,
                    "purpose": self.purpose,
                    "start_timestamp": self.start_time.isoformat(),
                }
                meta.update(self.tags)
                self.backends["mlflow"].set_tags(meta)
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log experiment metadata: {e}")

    def _log_environment(self) -> None:
        try:
            if not self.env_tagger:
                return
            env_info = self.env_tagger.get_environment_info()
            try:
                if hasattr(self.env_tagger, "get_git_info"):
                    env_info["git"] = self.env_tagger.get_git_info()
            except Exception:
                pass
            if self.backends.get("mlflow"):
                env_tags = self.env_tagger.get_mlflow_tags(env_info)
                self.backends["mlflow"].set_tags(env_tags)
                self.backends["mlflow"].log_dict(env_info, "environment_info.json")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to log environment: {e}")

    def _check_performance_warnings(self, step: int, metrics: Dict[str, Any]) -> None:
        try:
            warnings: List[str] = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    if v != v:  # NaN
                        warnings.append(f"NaN value detected in {k} at step {step}")
                    elif abs(v) == float("inf"):
                        warnings.append(f"Infinite value detected in {k} at step {step}")
            if warnings:
                self._resource_warnings.extend(warnings)
                self._resource_warnings = self._resource_warnings[-20:]
                for w in warnings:
                    self.logger.warning(w)
        except Exception:  # pragma: no cover
            pass

    # ---------------------------
    # Final summaries & recommendations
    # ---------------------------
    def _finalize_results_summary(self, *, log_to_backends: bool = True) -> Dict[str, Any]:
        """Aggregate compact results per scope and optionally persist."""
        out: Dict[str, Any] = {"created_at": datetime.now().isoformat(), "scopes": {}}
        scope_keys = set(self._tasker_rewards_n) | set(self._builder_rewards_n) | set(self._rl_episodes_n) | set(self._cls_n) | set(self._gen_n)
        for sk in sorted(scope_keys):
            s: Dict[str, Any] = {}
            # tasker
            if self._tasker_rewards_n.get(sk, 0):
                n = self._tasker_rewards_n[sk]
                sums = dict(self._tasker_rewards_sum[sk])
                s["tasker_rewards_avg"] = {k: (v / n) for k, v in sums.items()}
            # builder
            if self._builder_rewards_n.get(sk, 0):
                n = self._builder_rewards_n[sk]
                sums = dict(self._builder_rewards_sum[sk])
                s["builder_rewards_avg"] = {k: (v / n) for k, v in sums.items()}
            # rl
            if self._rl_episodes_n.get(sk, 0):
                n = self._rl_episodes_n[sk]
                s["rl"] = {
                    "episodes": n,
                    "avg_reward": self._rl_total_reward[sk] / n,
                    "avg_steps": self._rl_total_steps[sk] / n,
                    "action_dist_total": dict(self._rl_action_dist[sk]),
                }
            # classification
            if self._cls_n.get(sk, 0):
                n = self._cls_n[sk]
                s["classification"] = {
                    "records": n,
                    "avg_accuracy": self._cls_acc_sum[sk] / n,
                    "avg_macro_f1": self._cls_macro_f1_sum[sk] / n,
                }
            # generation
            if self._gen_n.get(sk, 0):
                s["generation"] = {
                    "samples": self._gen_n[sk],
                    "avg_score": (self._gen_score_sum[sk] / self._gen_n[sk]) if self._gen_n[sk] else None,
                }
            out["scopes"][sk] = s

        if log_to_backends:
            if self.backends.get("mlflow"):
                self.backends["mlflow"].log_dict(out, "results/summary.json")
            if self.backends.get("mongodb"):
                try:
                    self.backends["mongodb"].store_results(self.experiment_id, {"kind": "results_summary", **out})
                except Exception:
                    pass
        return out

    def get_experiment_summary(self) -> Dict[str, Any]:
        try:
            duration = (
                (datetime.now() - self.start_time).total_seconds() if self.is_active else
                (self.end_time - self.start_time).total_seconds() if self.end_time else 0
            )
            summary: Dict[str, Any] = {
                "experiment_id": self.experiment_id,
                "name": self.experiment_name,
                "purpose": self.purpose,
                "status": "Running" if self.is_active else "Completed",
                "duration_seconds": duration,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "training_steps": self.training_steps,
                "checkpoints_saved": self.checkpoints_saved,
                "total_cost": self.total_cost,
                "lezea_config": self.lezea_config,
                "constraints": self.constraints,
                "backends_available": [k for k, v in self.backends.items() if v is not None],
                "backend_errors": self.backend_errors,
                "resource_summary": self._final_resource_summary(),
            }
            if self._step_times:
                summary["avg_step_time"] = sum(
                    self._step_times[i] - self._step_times[i - 1] for i in range(1, len(self._step_times))
                ) / max(1, len(self._step_times) - 1)
            if self._resource_warnings:
                summary["resource_warnings"] = self._resource_warnings[-5:]
            return summary
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to get experiment summary: {e}")
            return {"error": str(e)}

    def get_recommendations(self) -> List[str]:
        recs: List[str] = []
        try:
            duration = (datetime.now() - self.start_time).total_seconds()
            if duration > 3600:
                recs.append("Long experiment duration — checkpoint more often.")
            if len(self._step_times) > 10:
                recent = self._step_times[-10:]
                deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
                if sum(deltas) / max(1, len(deltas)) > 10:
                    recs.append("Slow steps — consider batch size / dataloader tuning.")
            failed = [k for k, v in self.backends.items() if v is None]
            if failed:
                recs.append(f"Backends unavailable: {', '.join(failed)} — fix for full functionality.")
            if not recs:
                recs.append("Experiment running efficiently — no immediate actions.")
            return recs
        except Exception:  # pragma: no cover
            return ["Unable to generate recommendations."]

    # ---------------------------
    # Context manager sugar
    # ---------------------------
    def __enter__(self) -> "ExperimentTracker":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.logger.error(f"Experiment failed with {exc_type.__name__}: {exc_val}")
            print(f"❌ Experiment failed: {exc_val}")
        self.end()
        # Do not suppress exceptions
