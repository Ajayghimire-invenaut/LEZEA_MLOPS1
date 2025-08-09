# lezea_mlops/utils/plots.py
"""
Lightweight plotting helpers for LeZeA.
- Matplotlib only (no seaborn).
- Can auto-log figures via your MLflow backend.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def _maybe_log_figure(fig, mlflow_backend, artifact_path: str) -> Optional[str]:
    """
    Try to log a Matplotlib figure using the provided MLflow backend.
    - Prefers backend.log_figure(fig, artifact_path) if it exists.
    - Falls back to saving a temp PNG and calling backend.log_artifact(path).
    Returns the local file path that was logged (or None if logged directly).
    """
    if mlflow_backend is None:
        return None

    try:
        if hasattr(mlflow_backend, "log_figure"):
            mlflow_backend.log_figure(fig, artifact_path)
            return None

        # Fallback path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight")
            tmp.flush()
            tmp_path = tmp.name
        try:
            # Some backends support log_artifact(path, artifact_path=...)
            if hasattr(mlflow_backend, "log_artifact") and "artifact_path" in mlflow_backend.log_artifact.__code__.co_varnames:
                mlflow_backend.log_artifact(tmp_path, artifact_path=os.path.dirname(artifact_path) or None)
            elif hasattr(mlflow_backend, "log_artifact"):
                mlflow_backend.log_artifact(tmp_path)
        finally:
            # Do not remove tmp file until MLflow has picked it up (safe to keep)
            pass
        return tmp_path
    except Exception:
        return None


def _moving_avg(values: Sequence[float], window: int) -> List[float]:
    if window <= 1 or window > len(values):
        return list(values)
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in values:
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def plot_rewards_curve(
    rewards: Sequence[float],
    *,
    episodes: Optional[Sequence[int]] = None,
    smooth_window: int = 0,
    title: str = "Episode Reward",
    xlabel: str = "Episode",
    ylabel: str = "Total Reward",
    mlflow_backend=None,
    artifact_path: str = "plots/rl_rewards.png",
    show: bool = False,
):
    """
    Plot episode rewards with an optional moving-average smoothing curve.
    """
    if episodes is None:
        episodes = list(range(1, len(rewards) + 1))

    smoothed = _moving_avg(rewards, smooth_window) if smooth_window else list(rewards)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(list(episodes), list(rewards), label="reward", linewidth=1.0)
    if smooth_window:
        ax.plot(list(episodes), smoothed, label=f"moving_avg({smooth_window})", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    _maybe_log_figure(fig, mlflow_backend, artifact_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"points": len(rewards), "smoothed": bool(smooth_window)}


def plot_confusion_matrix(
    confusion: Dict[str, Dict[str, int]],
    *,
    labels: Optional[List[str]] = None,
    normalize: Optional[str] = None,  # "true" | "pred" | "all" | None
    title: str = "Confusion Matrix",
    mlflow_backend=None,
    artifact_path: str = "plots/confusion_matrix.png",
    annotate: bool = True,
    show: bool = False,
):
    """
    Render a confusion matrix dict produced by tracker.log_classification_results().
    - `normalize`:
        - None: raw counts
        - "true": divide each row (per-true-label)
        - "pred": divide each column (per-pred-label)
        - "all": divide by total count
    """
    # Build dense matrix
    if labels is None:
        labels = sorted(set(confusion.keys()) | {p for row in confusion.values() for p in row.keys()})

    n = len(labels)
    import numpy as np  # local import to keep global deps simple

    M = np.zeros((n, n), dtype=float)
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            M[i, j] = float(confusion.get(t, {}).get(p, 0))

    if normalize in {"true", "row"}:
        row_sums = M.sum(axis=1, keepdims=True) + 1e-12
        M = M / row_sums
    elif normalize in {"pred", "col", "column"}:
        col_sums = M.sum(axis=0, keepdims=True) + 1e-12
        M = M / col_sums
    elif normalize == "all":
        denom = M.sum() + 1e-12
        M = M / denom

    fig = plt.figure(figsize=(max(4, n * 0.6), max(3.5, n * 0.6)))
    ax = fig.gca()
    im = ax.imshow(M, aspect="auto")  # default colormap
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if annotate:
        # Show values; if matrix appears normalized use 2 decimals
        is_prob = (M.max() <= 1.0 + 1e-9)
        fmt = "{:.2f}" if is_prob else "{:.0f}"
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)

    _maybe_log_figure(fig, mlflow_backend, artifact_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"labels": labels, "shape": (n, n), "normalized": normalize is not None}
