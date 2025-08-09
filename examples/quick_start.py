#!/usr/bin/env python3
"""
LeZeA MLOps - Quick Start Example
================================

Minimal, copy-paste example for LeZeA MLOps.

Shows:
- Experiment start/end
- LeZeA config + constraints
- Per-step metric logging (with data-usage wiring)
- Data splits & dataset version
- Classification results
- Checkpoint artifact logging

Usage:
    python examples/quick_start.py
"""

import os
import time
import math
import random
import tempfile
from lezea_mlops import ExperimentTracker


def main():
    print("🚀 LeZeA MLOps Quick Start\n" + "=" * 40)

    # Start an experiment (auto-start) and auto-end via context manager
    with ExperimentTracker("quick_start_demo", purpose="Minimal example", auto_start=True) as tracker:
        # LeZeA settings + constraints
        tracker.log_lezea_config(
            tasker_pop_size=4,
            builder_pop_size=1,
            algorithm_type="DQN",
            hyperparameters={"lr": 1e-3, "batch": 32},
            seeds={"global": 42},
            init_scheme="kaiming",
        )
        tracker.log_constraints(max_runtime=120, max_steps=20)

        # Dataset bookkeeping
        tracker.log_data_splits(train=800, val=100, test=100)
        tracker.log_dataset_version("sample_data", dataset_root="data/")

        # Simulated training loop (with data-usage & delta_loss)
        accuracy = 0.0
        for step in range(1, 21):
            loss = 1.0 / math.sqrt(step)
            accuracy = min(0.99, 1.0 - loss * 0.30 + random.uniform(-0.02, 0.02))
            batch_ids = [f"train_{step}_{i}" for i in range(32)]  # pretend these come from your dataloader

            tracker.log_training_step(
                step,
                loss=loss,
                accuracy=accuracy,
                sample_ids=batch_ids,  # enables data-usage tracking
                split="train",
            )
            time.sleep(0.05)

        # Example results: classification
        y_true = [0, 1, 0, 1, 1, 0, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
        tracker.log_classification_results(y_true, y_pred, split="val")

        # Example artifact: checkpoint
        with tempfile.NamedTemporaryFile("w", suffix=".ckpt", delete=False) as f:
            f.write("dummy checkpoint contents\n")
            ckpt_path = f.name
        try:
            tracker.log_checkpoint(ckpt_path, step=20, role="tasker")
            print("💾 Logged a dummy checkpoint artifact")
        finally:
            os.unlink(ckpt_path)

        print("✅ Quick start run complete — results logged to backends.")


if __name__ == "__main__":
    main()
