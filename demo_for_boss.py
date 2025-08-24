#!/usr/bin/env python3
"""
LeZeA MLOps - Executive Demo (no changes to tree code)
- Runtime shim so `ExperimentTracker` can import `ModificationTree` even if only `ModTree` exists.
- If ExperimentTracker import OR __init__ fails, fall back to an MLflow-only tracker.
"""

import os
import json
import time
import math
import uuid
import pathlib
from datetime import datetime

# --------------------- load env if available ---------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --------------------- runtime compat shim (no repo edits) ---------------------
def _ensure_modificationtree_symbol():
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
            trees.ModificationTree = ModificationTree
            sys.modules["lezea_mlops.modification.trees"].ModificationTree = ModificationTree
    except Exception:
        # ignore; we'll use fallback tracker if needed
        pass

_ensure_modificationtree_symbol()

# --------------------- tracking backstop (fallback) ---------------------
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

class _FallbackTracker:
    """Minimal tracker that uses MLflow only; keeps the demo running without touching your code."""
    def __init__(self, experiment_name: str, purpose: str, tags: dict):
        if mlflow is None:
            raise RuntimeError("mlflow is not installed")
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name="executive_demo", tags=tags)
        self.run_id = mlflow.active_run().info.run_id
        mlflow.log_params({"purpose": purpose, "demo_start_ts": datetime.utcnow().isoformat() + "Z"})

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None):
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and math.isfinite(float(v))}
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_dataset_info(self, info: dict):
        _write_json_artifact(info, "data/dataset_info.json")

    def finish_run(self):
        mlflow.log_param("demo_end_ts", datetime.utcnow().isoformat() + "Z")
        mlflow.end_run()

def make_tracker(**kwargs):
    """
    Try to use your real ExperimentTracker; if import OR instantiation fails,
    automatically fall back to _FallbackTracker.
    """
    try:
        from lezea_mlops import ExperimentTracker as _RealTracker  # your implementation
    except Exception as e:
        print("❌ ExperimentTracker import failed; using MLflow-only fallback for this demo.")
        return _FallbackTracker(**kwargs)

    try:
        return _RealTracker(**kwargs)
    except Exception as e:
        print("❌ ExperimentTracker failed to initialize:")
        print(e)
        print("➡️  Falling back to MLflow-only tracker for this demo.")
        return _FallbackTracker(**kwargs)

# --------------------- artifact helpers ---------------------
ART_DIR = pathlib.Path("demo_artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

def _log_artifact_local(path: pathlib.Path, artifact_path: str):
    if mlflow:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

def _write_json_artifact(obj: dict, relpath: str) -> str:
    p = ART_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    _log_artifact_local(p, str(p.parent.relative_to(ART_DIR)))
    return str(p)

def _plot_metrics_png(history: list[dict], relpath: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    epochs = [m["epoch"] for m in history]
    loss   = [m["loss"] for m in history]
    acc    = [m["accuracy"] for m in history]
    ppl    = [m["perplexity"] for m in history]
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

# --------------------- pretty printing ---------------------
def print_header(title):
    print(f"\n{'='*60}\n🚀 {title}\n{'='*60}")

def print_section(title):
    print(f"\n{'─'*40}\n📊 {title}\n{'─'*40}")

# --------------------- main demo ---------------------
def main():
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
    print("✅ MLOps platform initialized successfully")
    run_id = getattr(tracker, "run_id", None)
    print(f"📊 Experiment ID: {run_id}")

    # 2) AGI model configuration
    print_section("2. AGI Model Configuration")
    agi_config = {
        "model_architecture": "transformer_agi",
        "parameters": 175_000_000_000,
        "layers": 96,
        "attention_heads": 96,
        "hidden_size": 12288,
        "context_length": 8192,
        "training_data_size": "500TB",
        "compute_requirements": "8x A100 GPUs minimum",
    }
    tracker.log_params(agi_config)
    _write_json_artifact(agi_config, "env/agi_config.json")
    print("✅ AGI model configuration logged")

    # 3) Training simulation
    print_section("3. AGI Training Simulation")
    print("🔄 Starting AGI training simulation...")
    training_metrics: list[dict] = []
    prev_loss = None
    for epoch in range(1, 11):
        loss = 2.5 * (0.85 ** epoch) + 0.1
        accuracy = min(0.95, 0.3 + (epoch * 0.07))
        perplexity = 15.0 * (0.8 ** epoch) + 1.2
        lr = 0.0001 * (0.95 ** epoch)
        tokens = epoch * 1_000_000_000
        delta_loss = 0.0 if prev_loss is None else (prev_loss - loss)
        prev_loss = loss

        metrics = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "learning_rate": lr,
            "tokens_processed": tokens,
            "delta_loss": delta_loss,
            "gpu_utilization": 95 + (epoch % 3),
            "memory_usage_gb": 45 + (epoch * 2),
        }
        tracker.log_metrics(metrics, step=epoch)
        training_metrics.append(metrics)
        # lightweight “checkpoint”
        _write_json_artifact({"epoch": epoch, "state_stub": f"weights-{uuid.uuid4().hex[:8]}"},
                             f"checkpoints/epoch_{epoch:02d}.json")
        print(f"   Epoch {epoch:2d}: loss={loss:.4f}, accuracy={accuracy:.4f}, perplexity={perplexity:.2f}")
        time.sleep(0.05)

    _write_json_artifact({"history": training_metrics}, "training/history.json")
    _plot_metrics_png(training_metrics, "plots/metrics.png")
    print("✅ AGI training simulation completed")

    # 4) Dataset info
    print_section("4. Dataset Management")
    dataset_info = {
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
    tracker.log_dataset_info(dataset_info)
    _write_json_artifact(dataset_info, "data/dataset_info_full.json")
    print("✅ Dataset information logged")

    # 5) Business metrics
    print_section("5. Business Impact Analysis")
    business_metrics = {
        "training_cost_usd": 2_500_000,
        "compute_hours": 50_000,
        "energy_consumption_kwh": 125_000,
        "carbon_footprint_kg": 62_500,
        "expected_roi_percent": 450,
        "time_to_market_days": 90,
    }
    for k, v in business_metrics.items():
        tracker.log_metrics({f"business_{k}": v})
    _write_json_artifact(business_metrics, "business/business_summary.json")
    print("✅ Business metrics calculated")

    # 6) System health (info only; live checks are in test_services.py)
    print_section("6. System Health & Monitoring")
    system_status = {
        "mongodb": "tracked by ExperimentTracker (if enabled)",
        "prometheus": "optional (Pushgateway)",
        "grafana": "optional dashboards",
        "object_storage": "S3 via MLflow artifact root",
        "mlflow": os.getenv("MLFLOW_TRACKING_URI", "not-set"),
    }
    _write_json_artifact(system_status, "monitoring/system_status.json")
    for c, s in system_status.items():
        print(f"   {c}: {s}")

    # 7) Performance summary
    print_section("7. Performance Summary")
    final = training_metrics[-1]
    perf = {
        "final_accuracy": final["accuracy"],
        "loss_reduction_pct": (training_metrics[0]["loss"] - final["loss"]) / training_metrics[0]["loss"],
        "total_tokens": final["tokens_processed"],
        "avg_gpu_util": sum(m["gpu_utilization"] for m in training_metrics) / len(training_metrics),
        "peak_mem_gb": max(m["memory_usage_gb"] for m in training_metrics),
    }
    _write_json_artifact(perf, "results/performance_summary.json")
    for k, v in {
        "Final Model Accuracy": f"{perf['final_accuracy']:.2%}",
        "Training Loss Reduction": f"{perf['loss_reduction_pct']:.1%}",
        "Total Tokens Processed": f"{perf['total_tokens']:,}",
        "Average GPU Utilization": f"{perf['avg_gpu_util']:.1f}%",
        "Peak Memory Usage": f"{perf['peak_mem_gb']:.1f} GB",
    }.items():
        print(f"   {k}: {v}")

    # 8) Integration readiness
    print_section("8. AGI Integration Readiness")
    checklist = {
        "Model Training": "Complete - simulated to 10 epochs",
        "Experiment Tracking": "Active - metrics + artifacts",
        "Model Versioning": "Checkpoints saved per epoch",
        "Monitoring": "See monitoring/system_status.json",
        "Data Pipeline": "Dataset info + splits logged",
        "Cost Tracking": "Business metrics logged",
        "Compliance": "Artifacts provide audit trail",
    }
    _write_json_artifact(checklist, "results/integration_readiness.json")
    for k, v in checklist.items():
        print(f"   ✅ {k}: {v}")

    # 9) Access URLs
    print_section("9. Platform Access")
    urls = {
        "MLflow UI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        "Grafana": os.getenv("GRAFANA_URL", "http://127.0.0.1:3000"),
        "Prometheus": os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090"),
    }
    _write_json_artifact(urls, "report/access_urls.json")
    for k, v in urls.items():
        print(f"   {k}: {v}")

    # Index
    _write_json_artifact({
        "run_id": run_id,
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

    # Finish up
    tracker.finish_run()

    print_header("EXECUTIVE SUMMARY")
    print("🎉 LeZeA MLOps Platform — DEMO COMPLETE (metrics + artifacts logged)")
    print(f"⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
