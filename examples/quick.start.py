
"""
QuickStart: Simulated AGI connector usage for LeZeA Tracker.

Run:
    python examples/quick_start.py

This script:
- Starts the tracker
- Logs LeZeA config + constraints
- Simulates training for a few steps
- Logs per-part resources using ResourceLogger
- Saves a fake checkpoint
- Finalizes the run cleanly
"""

import os
import json
import time
import tempfile
from random import random, gauss

from lezea_mlops.tracker import ExperimentTracker, NetworkType
from lezea_mlops.monitoring.resource_logger import ResourceLogger


def main():
    tracker = ExperimentTracker(
        experiment_name="QuickStart-Demo",
        purpose="Connector smoke test",
        async_mode=True,
        auto_start=True,
    )

    # Spec 1.4 LeZeA config
    tracker.log_lezea_config(
        tasker_population_size=32,
        builder_population_size=4,
        algorithm_type="SAC",
        start_network_id="seed_A",
        hyperparameters={"lr": 3e-4, "gamma": 0.99, "batch_size": 256},
        seeds={"global": 1337, "layer": 2025},
        initialization_scheme="kaiming_uniform",
    )

    # Optional: constraints (Spec 1.3)
    tracker.log_constraints(runtime_limit_seconds=60 * 10, step_episode_limits={"max_steps": 200})

    # Resource logger (per-part)
    rlog = ResourceLogger()
    rlog.register_part("tasker#0")
    rlog.register_part("builder#0")
    rlog.start("tasker#0")
    rlog.start("builder#0")

    # Fake training loop
    loss = 1.5
    for step in range(1, 11):
        # -------- metrics (Spec 1.5.1) --------
        loss = max(0.01, loss - gauss(0.12, 0.02))
        acc = min(0.999, 1.0 - loss * 0.1)
        tracker.log_training_step(step, {"loss": loss, "accuracy": acc}, data_split="train")

        # -------- modification stats (Spec 1.5.2/1.5.3) --------
        if step % 3 == 0:
            mods = [
                {"type": "mutation", "accepted": random() > 0.4},
                {"type": "crossover", "accepted": random() > 0.6},
            ]
            tracker.log_modification_tree(step, modifications=mods, statistics={"touched": len(mods)})

        # -------- data usage (Spec 1.5.6/1.5.7/1.5.7.1) --------
        sample_ids = [f"s{step}_{i}" for i in range(4)]
        tracker.log_data_usage(sample_ids, epoch=step)
        tracker.log_data_score(sample_ids[0], before_loss=loss + 0.05, after_loss=loss)

        # -------- per-part resources (Spec 1.5.4/1.5.4.1/1.5.4.2) --------
        # Take a quick sample and forward to tracker
        for part in ("tasker#0", "builder#0"):
            snap = rlog.sample(part)  # cpu_h, mem_gb_h, gpu_h, io_bytes, elapsed_s
            # Convert to percents for an approximate single-interval log view:
            # cpu_percent ≈ cpu_h / elapsed_h * 100
            elapsed_h = max(1e-9, snap["elapsed_s"] / 3600.0)
            cpu_pct = float(snap["cpu_h"] / elapsed_h * 100.0) if snap["elapsed_s"] > 0 else 0.0
            mem_mb = float(snap["mem_gb_h"] / elapsed_h * 1024.0) if snap["elapsed_s"] > 0 else 0.0
            gpu_pct = float(snap["gpu_h"] / elapsed_h * 100.0) if snap["elapsed_s"] > 0 else 0.0
            tracker.log_component_resources(
                component_id=part, component_type=("tasker" if "tasker" in part else "builder"),
                cpu_percent=cpu_pct, memory_mb=mem_mb, gpu_util_percent=gpu_pct, io_operations=int(snap["io_bytes"]),
                step=step
            )

        # -------- rewards/results (Spec 1.6) --------
        if step % 2 == 0:
            tracker.log_reward("tasker#0", reward=1.0 - loss)
            tracker.log_results(builder_rewards={"builder#0": 0.5 + random() * 0.2}, step=step)

        time.sleep(0.2)  # simulate work

    # Save a fake checkpoint (Spec 1.5.5/1.5.6)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "step_010.pt")
        with open(ckpt, "wb") as f:
            f.write(os.urandom(1024))  # fake weights
        tracker.log_checkpoint(ckpt, step=10, metadata={"note": "demo checkpoint"})

    # Business/costs (Spec 1.7)
    totals = rlog.get_totals()
    # naive CPU-hours sum as a "cost driver" example (you can wire CostModel if present)
    cpu_hours_total = sum(t["cpu_h"] for t in totals.values())
    mem_gb_total = sum(t["mem_gb_h"] for t in totals.values())
    io_bytes_total = int(sum(t["io_bytes"] for t in totals.values()))
    tracker.log_resource_cost(cpu_hours_total, mem_gb_total, io_bytes_total)
    tracker.log_comment("QuickStart finished without errors.")
    tracker.log_conclusion("Demo run shows full connector surface working end-to-end.")

    # Finish
    rlog.stop("tasker#0")
    rlog.stop("builder#0")
    tracker.finalize_experiment()


if __name__ == "__main__":
    main()
