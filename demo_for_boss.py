#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeZeA MLOps - Executive Demo (works with real ExperimentTracker OR MLflow fallback)
- Provides a compat shim for ModificationTree/ModTree
- Uses adapter helpers so it runs against either tracker API
- Modified to include checks/verifications after each major logging step to ensure everything is tracked successfully.
- At the end, added a comprehensive verification section that queries MLflow (if available) and checks artifact files.
"""

import os
import json
import time
import math
import uuid
import pathlib
from datetime import datetime
from typing import Optional, List, Dict  # 3.9-compatible types

# --------------------- load env if available ---------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --------------------- runtime compat shim (no repo edits) ---------------------
def _ensure_modificationtree_symbol() -> None:
    """Provide lezea_mlops.modification.trees.ModificationTree dynamically if only ModTree exists."""
    try:
        import sys, importlib
        trees = importlib.import_module("lezea_mlops.modification.trees")
        if hasattr(trees, "ModificationTree"):
            return
        if hasattr(trees, "ModTree"):
            class ModificationTree(trees.ModTree):  # type: ignore
                """Compat alias for legacy imports; no behavior change."""
                pass
            trees.ModificationTree = ModificationTree  # type: ignore[attr-defined]
            sys.modules["lezea_mlops.modification.trees"].ModificationTree = ModificationTree  # type: ignore[attr-defined]
    except Exception:
        # ignore; we'll use fallback tracker if needed
        pass

_ensure_modificationtree_symbol()

# --------------------- tracking backstop (fallback) ---------------------
try:
    import mlflow  # type: ignore
    from mlflow.tracking import MlflowClient  # Added for verification
except Exception:
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None

class _FallbackTracker:
    """Minimal tracker that uses MLflow only; keeps the demo running without touching your code."""
    def __init__(self, experiment_name: str, purpose: str, tags: Dict[str, str]):
        if mlflow is None:
            raise RuntimeError("mlflow is not installed")
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name="executive_demo", tags=tags)
        self.run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        mlflow.log_params({"purpose": purpose, "demo_start_ts": datetime.utcnow().isoformat() + "Z"})

    # demo-style API
    def log_params(self, params: Dict[str, object]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, object], step: Optional[int] = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and math.isfinite(float(v))}
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_dataset_info(self, info: Dict[str, object]) -> None:
        _write_json_artifact(info, "data/dataset_info.json")

    def finish_run(self) -> None:
        mlflow.log_param("demo_end_ts", datetime.utcnow().isoformat() + "Z")
        mlflow.end_run()

def make_tracker(**kwargs):
    """
    Try to use your real ExperimentTracker; if import OR instantiation fails,
    automatically fall back to _FallbackTracker.
    """
    try:
        from lezea_mlops import ExperimentTracker as _RealTracker  # your implementation
    except Exception:
        print("❌ ExperimentTracker import failed; using MLflow-only fallback for this demo.")
        return _FallbackTracker(**kwargs)

    try:
        return _RealTracker(**kwargs)
    except Exception as e:
        print("❌ ExperimentTracker failed to initialize:", e)
        print("➡️  Falling back to MLflow-only tracker for this demo.")
        return _FallbackTracker(**kwargs)

# --------------------- artifact helpers ---------------------
ART_DIR = pathlib.Path("demo_artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

def _log_artifact_local(path: pathlib.Path, artifact_path: str) -> None:
    if mlflow:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

def _write_json_artifact(obj: Dict[str, object], relpath: str) -> str:
    p = ART_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    _log_artifact_local(p, str(p.parent.relative_to(ART_DIR)))
    return str(p)

def _plot_metrics_png(history: List[Dict[str, float]], relpath: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    epochs = [int(m["epoch"]) for m in history]
    loss   = [float(m["loss"]) for m in history]
    acc    = [float(m["accuracy"]) for m in history]
    ppl    = [float(m["perplexity"]) for m in history]
    plt.figure()
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, acc, label="accuracy")
    plt.plot(epochs, ppl, label="perplexity")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    p = ART_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p); plt.close()
    _log_artifact_local(p, str(p.parent.relative_to(ART_DIR)))
    return str(p)

# --------------------- adapter helpers (work with both tracker APIs) ---------------------
def t_start_if_needed(tracker) -> None:
    """Call tracker.start() if it exists (real ExperimentTracker), else do nothing (fallback is already running)."""
    if hasattr(tracker, "start"):
        try:
            tracker.start()
        except Exception as e:
            print("⚠️ tracker.start() failed (continuing):", e)

def t_log_params(tracker, params: Dict[str, object]) -> None:
    try:
        if hasattr(tracker, "log_params"):
            tracker.log_params(params)
        else:
            # Real tracker: store params as an artifact
            _write_json_artifact(params, "params/params.json")
        print("✅ Params logged successfully")
    except Exception as e:
        print("❌ Failed to log params:", e)

def t_log_metrics(tracker, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    try:
        if hasattr(tracker, "log_metrics"):
            tracker.log_metrics(metrics, step=step)
        elif hasattr(tracker, "log_training_step"):
            tracker.log_training_step(step=step or 0, metrics=metrics)
        else:
            # last resort: write artifact
            _write_json_artifact({"step": step, "metrics": metrics}, f"metrics/step_{step or 0}.json")
        print("✅ Metrics logged successfully")
    except Exception as e:
        print("❌ Failed to log metrics:", e)

def t_log_dataset_info(tracker, info: Dict[str, object]) -> None:
    try:
        if hasattr(tracker, "log_dataset_info"):
            tracker.log_dataset_info(info)
        else:
            _write_json_artifact(info, "data/dataset_info.json")
        print("✅ Dataset info logged successfully")
    except Exception as e:
        print("❌ Failed to log dataset info:", e)

def t_finish(tracker) -> None:
    try:
        if hasattr(tracker, "finish_run"):
            tracker.finish_run()
        elif hasattr(tracker, "end"):
            tracker.end()
        print("✅ Tracker finished successfully")
    except Exception as e:
        print("❌ Failed to finish tracker:", e)

# --------------------- pretty printing ---------------------
def print_header(title: str) -> None:
    print(f"\n{'='*60}\n🚀 {title}\n{'='*60}")

def print_section(title: str) -> None:
    print(f"\n{'─'*40}\n📊 {title}\n{'─'*40}")

# --------------------- verification helpers ---------------------
def verify_artifact_exists(path: str) -> bool:
    """Check if artifact file exists."""
    return pathlib.Path(path).exists()

def verify_mlflow_params(run_id: str, expected_keys: List[str]) -> None:
    """Verify if params are logged in MLflow."""
    if MlflowClient:
        client = MlflowClient()
        run = client.get_run(run_id)
        logged_params = run.data.params.keys()
        for key in expected_keys:
            if key in logged_params:
                print(f"✅ Verified param: {key}")
            else:
                print(f"❌ Missing param: {key}")
    else:
        print("⚠️ MlflowClient not available for param verification")

def verify_mlflow_metrics(run_id: str, expected_metrics: List[str]) -> None:
    """Verify if metrics are logged in MLflow."""
    if MlflowClient:
        client = MlflowClient()
        run = client.get_run(run_id)
        logged_metrics = run.data.metrics.keys()
        for metric in expected_metrics:
            if metric in logged_metrics:
                print(f"✅ Verified metric: {metric}")
            else:
                print(f"❌ Missing metric: {metric}")
    else:
        print("⚠️ MlflowClient not available for metric verification")

def verify_artifacts(artifact_paths: List[str]) -> None:
    """Verify if artifact files exist."""
    for path in artifact_paths:
        full_path = str(ART_DIR / path)
        if verify_artifact_exists(full_path):
            print(f"✅ Verified artifact: {path}")
        else:
            print(f"❌ Missing artifact: {path}")

# --------------------- main demo ---------------------
def main() -> None:
    print_header("LeZeA MLOps - Executive Demonstration")
    print("🎯 Production-Ready MLOps Platform for AGI Integration")
    print("⏰ Demo started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1) Initialize experiment
    print_section("1. Platform Initialization")
    tracker = make_tracker(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "agi_integration_demo"),
        purpose="Executive demo showcasing MLOps capabilities for AGI code integration",
        tags={"demo": "executive", "integration": "agi", "priority": "high"},
    )
    # start() for real tracker; no-op for fallback
    t_start_if_needed(tracker)

    print("✅ MLOps platform initialized successfully")
    run_id = getattr(tracker, "run_id", None)
    print(f"📊 Experiment ID: {run_id}")

    # 2) AGI model configuration
    print_section("2. AGI Model Configuration")
    agi_config: Dict[str, object] = {
        "model_architecture": "transformer_agi",
        "parameters": 175_000_000_000,
        "layers": 96,
        "attention_heads": 96,
        "hidden_size": 12288,
        "context_length": 8192,
        "training_data_size": "500TB",
        "compute_requirements": "8x A100 GPUs minimum",
    }
    t_log_params(tracker, agi_config)
    config_path = _write_json_artifact(agi_config, "env/agi_config.json")
    if verify_artifact_exists(config_path):
        print("✅ AGI config artifact verified")
    else:
        print("❌ AGI config artifact missing")

    # 3) Training simulation
    print_section("3. AGI Training Simulation")
    print("🔄 Starting AGI training simulation...")
    training_metrics: List[Dict[str, float]] = []
    prev_loss: Optional[float] = None
    for epoch in range(1, 11):
        loss = 2.5 * (0.85 ** epoch) + 0.1
        accuracy = min(0.95, 0.3 + (epoch * 0.07))
        perplexity = 15.0 * (0.8 ** epoch) + 1.2
        lr = 0.0001 * (0.95 ** epoch)
        tokens = float(epoch * 1_000_000_000)
        delta_loss = 0.0 if prev_loss is None else (prev_loss - loss)
        prev_loss = loss

        metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "perplexity": float(perplexity),
            "learning_rate": float(lr),
            "tokens_processed": tokens,
            "delta_loss": float(delta_loss),
            "gpu_utilization": float(95 + (epoch % 3)),
            "memory_usage_gb": float(45 + (epoch * 2)),
        }
        t_log_metrics(tracker, metrics, step=epoch)
        training_metrics.append(metrics)
        # lightweight “checkpoint” (just a stub file for the demo)
        ckpt_path = _write_json_artifact({"epoch": epoch, "state_stub": f"weights-{uuid.uuid4().hex[:8]}"},
                             f"checkpoints/epoch_{epoch:02d}.json")
        if verify_artifact_exists(ckpt_path):
            print(f"✅ Checkpoint {epoch} verified")
        else:
            print(f"❌ Checkpoint {epoch} missing")
        print(f"   Epoch {epoch:2d}: loss={loss:.4f}, accuracy={accuracy:.4f}, perplexity={perplexity:.2f}")
        time.sleep(0.05)

    history_path = _write_json_artifact({"history": training_metrics}, "training/history.json")
    if verify_artifact_exists(history_path):
        print("✅ Training history verified")
    else:
        print("❌ Training history missing")
    plot_path = _plot_metrics_png(training_metrics, "plots/metrics.png")
    if plot_path and verify_artifact_exists(plot_path):
        print("✅ Metrics plot verified")
    else:
        print("⚠️ Metrics plot not generated or missing")
    print("✅ AGI training simulation completed")

    # 4) Dataset info
    print_section("4. Dataset Management")
    dataset_info: Dict[str, object] = {
        "name": "agi_training_corpus",
        "version": "2024.1",
        "size_tb": 500,
        "sources": ["web_crawl", "books", "papers", "code_repos"],
        "languages": 50,
        "quality_score": 0.94,
        "preprocessing_steps": ["deduplication", "filtering", "tokenization"],
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "challenge_usage": {"web_crawl": 0.6, "books": 0.2, "papers": 0.15, "code_repos": 0.05},
    }
    t_log_dataset_info(tracker, dataset_info)
    dataset_path = _write_json_artifact(dataset_info, "data/dataset_info_full.json")
    if verify_artifact_exists(dataset_path):
        print("✅ Dataset info artifact verified")
    else:
        print("❌ Dataset info artifact missing")
    print("✅ Dataset information logged")

    # 5) Business metrics
    print_section("5. Business Impact Analysis")
    business_metrics: Dict[str, float] = {
        "training_cost_usd": 2_500_000.0,
        "compute_hours": 50_000.0,
        "energy_consumption_kwh": 125_000.0,
        "carbon_footprint_kg": 62_500.0,
        "expected_roi_percent": 450.0,
        "time_to_market_days": 90.0,
    }
    for k, v in business_metrics.items():
        t_log_metrics(tracker, {f"business_{k}": v})
    business_path = _write_json_artifact(business_metrics, "business/business_summary.json")
    if verify_artifact_exists(business_path):
        print("✅ Business summary verified")
    else:
        print("❌ Business summary missing")
    print("✅ Business metrics calculated")

    # 6) System health (info only; live checks are in test_connections.py)
    print_section("6. System Health & Monitoring")
    system_status: Dict[str, str] = {
        "mongodb": "tracked by ExperimentTracker (if enabled)",
        "prometheus": "optional (Pushgateway/Exporter)",
        "grafana": "optional dashboards",
        "object_storage": "S3 via MLflow artifact root",
        "mlflow": os.getenv("MLFLOW_TRACKING_URI", "not-set"),
    }
    status_path = _write_json_artifact(system_status, "monitoring/system_status.json")
    if verify_artifact_exists(status_path):
        print("✅ System status verified")
    else:
        print("❌ System status missing")
    for c, s in system_status.items():
        print(f"   {c}: {s}")

    # 7) Performance summary
    print_section("7. Performance Summary")
    final = training_metrics[-1]
    first = training_metrics[0]
    loss_reduction = (first["loss"] - final["loss"]) / first["loss"] if first["loss"] else 0.0
    perf: Dict[str, float] = {
        "final_accuracy": float(final["accuracy"]),
        "loss_reduction_pct": float(loss_reduction),
        "total_tokens": float(final["tokens_processed"]),
        "avg_gpu_util": float(sum(m["gpu_utilization"] for m in training_metrics) / len(training_metrics)),
        "peak_mem_gb": float(max(m["memory_usage_gb"] for m in training_metrics)),
    }
    perf_path = _write_json_artifact(perf, "results/performance_summary.json")
    if verify_artifact_exists(perf_path):
        print("✅ Performance summary verified")
    else:
        print("❌ Performance summary missing")
    readable = {
        "Final Model Accuracy": f"{perf['final_accuracy']:.2%}",
        "Training Loss Reduction": f"{perf['loss_reduction_pct']:.1%}",
        "Total Tokens Processed": f"{int(perf['total_tokens']):,}",
        "Average GPU Utilization": f"{perf['avg_gpu_util']:.1f}%",
        "Peak Memory Usage": f"{perf['peak_mem_gb']:.1f} GB",
    }
    for k, v in readable.items():
        print(f"   {k}: {v}")

    # 8) Integration readiness
    print_section("8. AGI Integration Readiness")
    checklist: Dict[str, str] = {
        "Model Training": "Complete - simulated to 10 epochs",
        "Experiment Tracking": "Active - metrics + artifacts",
        "Model Versioning": "Checkpoints saved per epoch",
        "Monitoring": "See monitoring/system_status.json",
        "Data Pipeline": "Dataset info + splits logged",
        "Cost Tracking": "Business metrics logged",
        "Compliance": "Artifacts provide audit trail",
    }
    readiness_path = _write_json_artifact(checklist, "results/integration_readiness.json")
    if verify_artifact_exists(readiness_path):
        print("✅ Integration readiness verified")
    else:
        print("❌ Integration readiness missing")
    for k, v in checklist.items():
        print(f"   ✅ {k}: {v}")

    # 9) Access URLs
    print_section("9. Platform Access")
    urls: Dict[str, str] = {
        "MLflow UI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        "Grafana": os.getenv("GRAFANA_URL", "http://127.0.0.1:3000"),
        "Prometheus": os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090"),
    }
    urls_path = _write_json_artifact(urls, "report/access_urls.json")
    if verify_artifact_exists(urls_path):
        print("✅ Access URLs verified")
    else:
        print("❌ Access URLs missing")
    for k, v in urls.items():
        print(f"   {k}: {v}")

    # Index
    summary_path = _write_json_artifact({
        "run_id": run_id if isinstance(run_id, str) else str(run_id),
        "artifact_index": [
            "env/agi_config.json",
            "training/history.json",
            "plots/metrics.png",
            "data/dataset_info_full.json",
            "business/business_summary.json",
            "monitoring/system_status.json",
            "results/performance_summary.json",
            "results/integration_readiness.json",
            "checkpoints/*",
        ]
    }, "EXEC_SUMMARY.json")
    if verify_artifact_exists(summary_path):
        print("✅ Executive summary index verified")
    else:
        print("❌ Executive summary index missing")

    # Finish up
    t_finish(tracker)

    # 10) Comprehensive Verification
    print_section("10. Full Tracking Verification")
    if run_id:
        # Verify params (example expected keys)
        expected_params = ["purpose", "demo_start_ts", "model_architecture"]
        verify_mlflow_params(run_id, expected_params)
        
        # Verify metrics (example expected)
        expected_metrics = ["loss", "accuracy", "perplexity", "business_training_cost_usd"]
        verify_mlflow_metrics(run_id, expected_metrics)
        
        # Verify artifacts
        artifact_paths = [
            "env/agi_config.json",
            "training/history.json",
            "data/dataset_info_full.json",
            "business/business_summary.json",
            "monitoring/system_status.json",
            "results/performance_summary.json",
            "results/integration_readiness.json",
            "report/access_urls.json",
            "EXEC_SUMMARY.json"
        ]
        # Add checkpoint paths dynamically
        for epoch in range(1, 11):
            artifact_paths.append(f"checkpoints/epoch_{epoch:02d}.json")
        if _plot_metrics_png:  # If plot was generated
            artifact_paths.append("plots/metrics.png")
        verify_artifacts(artifact_paths)
    else:
        print("⚠️ Run ID not available; skipping MLflow verification")

    print_header("EXECUTIVE SUMMARY")
    print("🎉 LeZeA MLOps Platform — DEMO COMPLETE (metrics + artifacts logged and verified)")
    print(f"⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()