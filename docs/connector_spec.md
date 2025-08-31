docs/connector_spec.md
# LeZeA Tracker Connector Spec

This document defines the **public surface** Marcus should call from AGI code.  
Everything is backend-agnostic and safe to use even if some services are down.

## Import

```python
from lezea_mlops.tracker import ExperimentTracker, NetworkType

1) Lifecycle
tracker = ExperimentTracker(experiment_name, purpose="", tags=None, auto_start=False, async_mode=False)

Creates a run with a unique experiment_id (UUID4).

Always writes a local audit file: runs/<experiment_id>/events.jsonl.

tracker.start(strict=None, prometheus_port=8000)

Starts backends (MLflow, Mongo, etc.) if available.

Logs environment + git commit.

tracker.finalize_experiment() / tracker.end()

Flushes summaries, stops monitors, closes MLflow run.

2) Scopes (optional)

Use scopes to prefix metrics: builder:42/loss, tasker:A/reward, etc.

with tracker.builder_scope("42"):
    tracker.log_training_step(1, {"loss": 0.42})


Available: builder_scope, tasker_scope, algorithm_scope, network_scope, layer_scope.

3) LeZeA Configuration (Spec 1.4)
tracker.log_lezea_config(
    tasker_population_size=128,
    builder_population_size=8,
    algorithm_type="SAC",
    start_network_id="seed_001",
    hyperparameters={"lr": 3e-4, "gamma": 0.99},
    seeds={"global": 123, "layer": 456},
    initialization_scheme="kaiming_uniform",
)

4) Training (Spec 1.5)
Per-step metrics
tracker.log_training_step(step, {"loss": 0.731, "acc": 0.84}, data_split="train")


If "loss" is present, delta_loss is auto-computed.

Modification trees + stats
tracker.log_modification_tree(step, modifications=[...], statistics={"mutations": 5})
tracker.log_modification_tree_stats({"accept_rate": 0.31, "depth": 4})
tracker.log_modification_stats({"crossover": 12, "mutation": 33})

Data usage & relevance
tracker.log_data_splits(train_ids, val_ids, test_ids)
tracker.log_data_usage(sample_ids=["a","b","c"], epoch=3)
tracker.log_data_score(sample_id="a", before_loss=1.2, after_loss=0.9)
tracker.log_delta_loss(epoch=5, loss=0.44, prev_loss=0.53)

5) Resources (Spec 1.5.4, 1.5.4.1, 1.5.4.2)
Simple one-liner
tracker.log_resource("tasker#0", cpu=12.3, gpu=4.1, mem=512.0)  # mem in MB

Structured per-component
tracker.log_component_resources(
  component_id="tasker#0", component_type="tasker",
  cpu_percent=12.3, memory_mb=512.0, gpu_util_percent=4.1, io_operations=1024
)


Totals are aggregated and written on finalize.

6) Checkpoints & Datasets (Spec 1.5.5–1.5.6, 2.x)
tracker.log_checkpoint("checkpoints/step_100.pt", step=100)
tracker.log_dataset_version("imagenet", version_tag="v1.0", dataset_path="data/imagenet")

7) Results (Spec 1.6)
# unified
tracker.log_results(result_dict={"tasker_reward_avg": 0.71})

# granular
tracker.log_results(tasker_rewards={"t0": 0.8}, builder_rewards={"b0": 0.6}, action_outputs={"action": [0,1,0]}, step=10)
tracker.log_reward("t0", 0.91)

8) Business (Spec 1.7)
tracker.log_resource_cost(cpu_hours=3.2, mem_gb=11.7, io_bytes=5_000_000)
tracker.log_comment("Builder population converged early; widen mutation space.")
tracker.log_conclusion("SAC + curriculum v2 beats baseline by 4.3% with lower variance.")
tracker.log_business_metrics(resource_cost_eur=8.73)

9) Introspection
tracker.get_experiment_summary()
tracker.get_experiment_data()
tracker.get_recommendations()

Error Handling

All calls are safe: if a backend is missing/down, logs go to the local JSONL and any available backends.

Async mode queues writes to avoid blocking the training loop.

Minimal End-to-End Example
from lezea_mlops.tracker import ExperimentTracker

tracker = ExperimentTracker("Demo", purpose="sanity", auto_start=True)
tracker.log_lezea_config(64, 4, "SAC", hyperparameters={"lr":3e-4})
for step in range(1, 6):
    tracker.log_training_step(step, {"loss": 1.0/step})
tracker.log_reward("t0", 0.77)
tracker.finalize_experiment()