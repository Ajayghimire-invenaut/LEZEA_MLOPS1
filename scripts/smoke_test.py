#!/usr/bin/env python3
"""
LeZeA MLOps — E2E Smoke Test with Dummy Data

What it exercises:
- Start/end run + backend health printout
- LeZeA config + constraints
- Per-step metrics (with delta_loss)
- Data-usage wiring (sample_ids + split)
- Modification tree + stats
- RL episodes + action distribution
- Classification metrics + confusion matrix (+ plot)
- Generation outputs
- Checkpoint + final-model artifact logging
- Business metrics (manual)
- Figure logging via MLflow backend
- Experiment summary printout
"""

import os
import random
import math
import tempfile
from pathlib import Path

import numpy as np

from lezea_mlops import ExperimentTracker

# Optional plotting helpers if you added utils/plots.py
try:
    from lezea_mlops.utils.plots import (
        plot_metric_curve,
        plot_confusion_matrix,
        plot_reward_curve,
    )
    HAS_PLOTS_UTILS = True
except Exception:
    HAS_PLOTS_UTILS = False
    import matplotlib.pyplot as plt  # fallback only


def main():
    print("🚀 LeZeA MLOps — E2E smoke test")

    with ExperimentTracker("smoke_test_demo", purpose="E2E dummy run", auto_start=True) as tr:
        # LeZeA config + constraints
        tr.log_lezea_config(
            tasker_pop_size=8,
            builder_pop_size=2,
            algorithm_type="SAC",
            start_network_id="net_0001",
            hyperparameters={"lr": 3e-4, "batch": 64, "gamma": 0.99},
            seeds={"global": 123, "builder": 999},
            init_scheme="xavier",
        )
        tr.log_constraints(max_runtime=180, max_steps=60, max_episodes=10)

        # Data splits + dataset version
        tr.log_data_splits(train=2000, val=300, test=300)
        tr.log_dataset_version("dummy_dataset", dataset_root="data/")

        # --- Simulated training ---
        steps = 60
        losses, accs, dl = [], [], []
        for step in range(1, steps + 1):
            # Smoothly improving dummy metrics with noise
            loss = 1.5 / math.sqrt(step) + random.uniform(-0.02, 0.02)
            acc = min(0.995, 1.0 - 0.35 * loss + random.uniform(-0.01, 0.01))

            # Sample IDs to exercise data-usage
            batch_ids = [f"train_{step}_{i}" for i in range(64)]

            tr.log_training_step(
                step=step,
                loss=loss,
                accuracy=acc,
                split="train",
                sample_ids=batch_ids,
            )

            # keep arrays for plots
            if tr._last_loss is not None and len(losses) > 0:
                dl.append(loss - losses[-1])
            else:
                dl.append(0.0)
            losses.append(loss)
            accs.append(acc)

            # Occasionally log a tiny modification tree
            if step in (10, 30, 50):
                tr.log_modification_tree(
                    step=step,
                    modifications=[
                        {"op": "prune", "layer": "fc2", "ratio": 0.1},
                        {"op": "lr_decay", "old": 3e-4, "new": 2.4e-4},
                    ],
                    statistics={"params_removed": 12345, "sparsity": 0.12},
                )

        # --- RL episodes (dummy) ---
        total_rewards = []
        for ep in range(1, 6):
            actions = [random.choice(["L", "R", "U", "D"]) for _ in range(50 + ep * 5)]
            ep_reward = np.clip(np.random.normal(loc=50 + ep * 5, scale=5), 10, 120)
            tr.log_rl_episode(
                episode=ep,
                total_reward=float(ep_reward),
                steps=len(actions),
                actions=actions,
            )
            total_rewards.append(float(ep_reward))

        # --- Classification results (dummy) ---
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, size=200).tolist()
        # biased to be "pretty good"
        y_pred = [
            y if rng.random() < 0.82 else int((y + rng.integers(1, 3)) % 3) for y in y_true
        ]
        cls_metrics = tr.log_classification_results(y_true, y_pred, split="val")

        # --- Generation outputs (dummy) ---
        gen_items = [{"text": f"sample_{i}", "score": float(rng.uniform(0.0, 1.0))} for i in range(25)]
        tr.log_generation_outputs(gen_items, name="samples")

        # --- Checkpoint + final model artifacts (dummy files) ---
        with tempfile.NamedTemporaryFile("w", suffix=".ckpt", delete=False) as f:
            f.write("dummy checkpoint bytes\n")
            ckpt = f.name
        tr.log_checkpoint(ckpt, step=steps, role="tasker", metadata={"note": "dummy"})
        Path(ckpt).unlink(missing_ok=True)

        with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as f:
            f.write("dummy final tasker model\n")
            tasker_final = f.name
        tr.log_final_models(tasker_model_path=tasker_final)
        Path(tasker_final).unlink(missing_ok=True)

        # --- Business metrics (manual) ---
        tr.log_business_metrics(cost=3.75, comments="Smoke test run", conclusion="Looks healthy")

        # --- Figures: loss, accuracy, delta_loss, rewards, confusion matrix ---
        mlb = tr.backends.get("mlflow")
        if mlb:
            if HAS_PLOTS_UTILS:
                fig1 = plot_metric_curve(losses, title="Train Loss")
                mlb.log_figure(fig1, "figures/train_loss.png")

                fig2 = plot_metric_curve(accs, title="Train Accuracy")
                mlb.log_figure(fig2, "figures/train_accuracy.png")

                fig3 = plot_metric_curve(dl, title="Delta Loss")
                mlb.log_figure(fig3, "figures/delta_loss.png")

                fig4 = plot_reward_curve(total_rewards, title="RL Episode Rewards")
                mlb.log_figure(fig4, "figures/rl_rewards.png")

                fig5 = plot_confusion_matrix(
                    y_true, y_pred, labels=cls_metrics["labels"], title="Confusion (val)"
                )
                mlb.log_figure(fig5, "figures/confusion_val.png")
            else:
                import matplotlib.pyplot as plt
                for name, series in [
                    ("train_loss", losses),
                    ("train_accuracy", accs),
                    ("delta_loss", dl),
                ]:
                    plt.figure()
                    plt.plot(series)
                    plt.title(name)
                    plt.xlabel("step")
                    plt.ylabel(name)
                    mlb.log_figure(plt.gcf(), f"figures/{name}.png")
                    plt.close()

        # Print a concise summary to console
        summary = tr.get_experiment_summary()
        print("\n==== SUMMARY (compact) ====")
        for k, v in summary.items():
            if k in ("resource_summary", "backend_errors", "lezea_config", "constraints"):
                continue
            print(f"{k}: {v}")
        print("===========================\n")
        if mlb:
            try:
                print(f"🔎 MLflow run URL: {mlb.get_run_url()}")
            except Exception:
                pass

    print("✅ Smoke test finished.")


if __name__ == "__main__":
    main()
