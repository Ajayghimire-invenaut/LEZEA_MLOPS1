# lezea_mlops/monitoring/data_usage.py
from __future__ import annotations

from collections import defaultdict, Counter
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json
import math
from pathlib import Path


class DataUsageLogger:
    """
    Tracks per-split data usage and a simple "learning relevance" score.

    - Usage:
        • set_split_totals({"train": 50000, "val": 10000, "test": 10000})
        • update(sample_ids, split="train", delta_loss=... )
        • get_split_metrics("train")  -> {unique_seen, exposures, usage_rate, ...}
        • top_k("train", k=20)        -> [(sample_id, score), ...]
        • summary()                    -> full dump for artifacts

    - Relevance heuristic (default):
        We use per-step delta_loss (loss_t - loss_{t-1}).
        Only *improvements* (negative delta) are credited:
            credit_total = max(-delta_loss, 0.0)
        By default we split credit equally across the batch. You may
        optionally pass sample_weights to allocate credit proportionally.

    - Extras in this version:
        • Per-sample exposure counts (per split)
        • Optional weighted credit allocation via `sample_weights`
        • Optional memory cap: max_unique_per_split (avoid runaway sets)
        • Distribution stats per split (Gini, quantiles) for fairness/coverage
        • Thread-safe updates
        • Persistence helpers: to_dict()/from_dict(), export_json()
    """

    def __init__(self, *, max_unique_per_split: Optional[int] = None) -> None:
        self.split_totals: Dict[str, int] = {}  # declared dataset sizes
        self.seen_ids: Dict[str, set[str]] = defaultdict(set)  # unique IDs seen per split
        self.exposures: Counter[str] = Counter()  # total sample exposures per split
        self.id_exposures: Dict[str, Counter[str]] = defaultdict(Counter)  # split -> id -> exposures
        self.relevance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # split -> id -> score
        self.total_updates: Counter[str] = Counter()  # number of update() calls per split
        self.max_unique_per_split = max_unique_per_split
        self._lock = RLock()

    # --------------------------
    # Configuration
    # --------------------------
    def set_split_totals(self, totals: Dict[str, int]) -> None:
        with self._lock:
            for k, v in (totals or {}).items():
                try:
                    n = int(v)
                    if n > 0:
                        self.split_totals[k] = n
                except Exception:
                    pass

    # --------------------------
    # Updates
    # --------------------------
    def update(
        self,
        sample_ids: Iterable[Any],
        split: str,
        delta_loss: Optional[float] = None,
        sample_weights: Optional[Iterable[float]] = None,
    ) -> None:
        """
        Record a batch usage update.

        Args:
            sample_ids: iterable of sample identifiers (any -> str)
            split: split name ('train', 'val', etc.)
            delta_loss: loss_t - loss_{t-1}; negative means improvement
            sample_weights: optional weights (same length as sample_ids) for
                            proportional credit allocation. If omitted, credit
                            is split equally across the batch.
        """
        ids = [str(s) for s in (sample_ids or [])]
        if not ids or not split:
            return

        with self._lock:
            # Exposures & seen sets
            self.exposures[split] += len(ids)
            self.total_updates[split] += 1

            # Cap unique IDs per split if requested (avoid unbounded memory)
            cap = self.max_unique_per_split
            if cap is None or len(self.seen_ids[split]) < cap:
                # we may still exceed cap if many new ids in one batch; clamp softly
                remaining = None if cap is None else max(cap - len(self.seen_ids[split]), 0)
                if remaining is None:
                    addable = ids
                else:
                    # add at most 'remaining' brand new ids; existing ids always ok
                    existing = [i for i in ids if i in self.seen_ids[split]]
                    new_ids = [i for i in ids if i not in self.seen_ids[split]][:remaining]
                    addable = existing + new_ids
                self.seen_ids[split].update(addable)
            else:
                # at cap: don't add new unique IDs to the set
                addable = [i for i in ids if i in self.seen_ids[split]]

            # Per-id exposures (for distribution/fairness metrics)
            for sid in ids:
                self.id_exposures[split][sid] += 1

            # Relevance attribution (credit only on improvement)
            credit_total = 0.0
            if isinstance(delta_loss, (int, float)) and delta_loss < 0.0:
                credit_total = float(-delta_loss)

            if credit_total > 0.0:
                rmap = self.relevance[split]
                if sample_weights is not None:
                    w = [float(x) for x in sample_weights]
                    if len(w) != len(ids):
                        # silently ignore malformed weights to stay robust
                        share = credit_total / max(1, len(ids))
                        for sid in ids:
                            rmap[sid] += share
                    else:
                        s = sum(abs(x) for x in w)  # allow negative weights but use magnitude
                        if s <= 0:
                            share = credit_total / max(1, len(ids))
                            for sid in ids:
                                rmap[sid] += share
                        else:
                            for sid, wi in zip(ids, w):
                                rmap[sid] += credit_total * (abs(wi) / s)
                else:
                    share = credit_total / max(1, len(ids))
                    for sid in ids:
                        rmap[sid] += share

    # --------------------------
    # Metrics / Reports
    # --------------------------
    def get_split_metrics(self, split: str) -> Dict[str, float]:
        with self._lock:
            uniq = len(self.seen_ids.get(split, ()))
            exp = int(self.exposures.get(split, 0))
            out = {
                "unique_seen": float(uniq),
                "exposures": float(exp),
            }
            total = self.split_totals.get(split)
            if isinstance(total, int) and total > 0:
                out["usage_rate"] = float(uniq) / float(total)
                out["coverage_rate"] = out["usage_rate"]  # alias
            if uniq > 0:
                out["avg_exposures_per_seen"] = float(exp) / float(uniq)
            # helpful aggregates
            rmap = self.relevance.get(split, {})
            if rmap:
                out["relevance_sum"] = float(sum(rmap.values()))
                out["relevance_mean"] = float(out["relevance_sum"] / max(1.0, len(rmap)))
            return out

    def rates(self) -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = {}
            for split, total in self.split_totals.items():
                uniq = len(self.seen_ids.get(split, ()))
                if total > 0:
                    out[split] = float(uniq) / float(total)
            return out

    def top_k(self, split: str, k: int = 20) -> List[Tuple[str, float]]:
        with self._lock:
            rmap = self.relevance.get(split, {})
            # stable tie-break by exposures (desc) then id
            return sorted(
                rmap.items(),
                key=lambda kv: (kv[1], self.id_exposures[split].get(kv[0], 0), kv[0]),
                reverse=True,
            )[: max(1, k)]

    def get_sample_metrics(self, split: str, sample_id: Any) -> Dict[str, float]:
        """Return per-sample exposures and relevance for a given split."""
        sid = str(sample_id)
        with self._lock:
            return {
                "exposures": float(self.id_exposures.get(split, {}).get(sid, 0)),
                "relevance": float(self.relevance.get(split, {}).get(sid, 0.0)),
                "seen": float(1.0 if sid in self.seen_ids.get(split, set()) else 0.0),
            }

    def distribution_stats(self, split: str) -> Dict[str, float]:
        """
        Exposure distribution fairness metrics for a split.
        Returns: {gini, min, max, p50, p90, p95, n}
        """
        with self._lock:
            counts = list(self.id_exposures.get(split, {}).values())
        if not counts:
            return {"gini": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "n": 0.0}
        counts.sort()
        n = len(counts)
        s = sum(counts)
        # Gini coefficient
        if s <= 0:
            gini = 0.0
        else:
            # Efficient Gini using sorted data:
            # G = (2*sum_i(i*xi)) / (n*sum xi) - (n+1)/n
            cum = 0
            for i, x in enumerate(counts, start=1):
                cum += i * x
            gini = (2.0 * cum) / (n * s) - (n + 1.0) / n
        def _q(p: float) -> float:
            if n == 1:
                return float(counts[0])
            pos = p * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(counts[lo])
            return float(counts[lo] + (counts[hi] - counts[lo]) * (pos - lo))
        return {
            "gini": float(gini),
            "min": float(counts[0]),
            "max": float(counts[-1]),
            "p50": _q(0.50),
            "p90": _q(0.90),
            "p95": _q(0.95),
            "n": float(n),
        }

    def summary(self) -> Dict[str, object]:
        with self._lock:
            sums = {}
            for split, rmap in self.relevance.items():
                uniq = len(self.seen_ids.get(split, ()))
                exp = int(self.exposures.get(split, 0))
                total = self.split_totals.get(split)
                usage_rate = (uniq / total) if (isinstance(total, int) and total > 0) else None
                sums[split] = {
                    "unique_seen": uniq,
                    "exposures": exp,
                    "usage_rate": usage_rate,
                    "top_relevant": self.top_k(split, k=50),
                    "distribution": self.distribution_stats(split),
                }
            return {
                "split_totals": dict(self.split_totals),
                "splits": sums,
                "total_updates": dict(self.total_updates),
            }

    # --------------------------
    # Persistence / I/O
    # --------------------------
    def to_dict(self) -> Dict[str, object]:
        with self._lock:
            return {
                "split_totals": dict(self.split_totals),
                "seen_ids": {s: list(v) for s, v in self.seen_ids.items()},
                "exposures": dict(self.exposures),
                "id_exposures": {s: dict(c) for s, c in self.id_exposures.items()},
                "relevance": {s: dict(m) for s, m in self.relevance.items()},
                "total_updates": dict(self.total_updates),
                "max_unique_per_split": self.max_unique_per_split,
            }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DataUsageLogger":
        obj = cls(max_unique_per_split=payload.get("max_unique_per_split"))  # type: ignore[arg-type]
        with obj._lock:
            obj.split_totals = {str(k): int(v) for k, v in (payload.get("split_totals") or {}).items()}  # type: ignore[union-attr]
            obj.seen_ids = defaultdict(set, {s: set(v) for s, v in (payload.get("seen_ids") or {}).items()})  # type: ignore[arg-type]
            obj.exposures = Counter({s: int(v) for s, v in (payload.get("exposures") or {}).items()})  # type: ignore[arg-type]
            obj.id_exposures = defaultdict(Counter, {s: Counter(d) for s, d in (payload.get("id_exposures") or {}).items()})  # type: ignore[arg-type]
            obj.relevance = defaultdict(lambda: defaultdict(float))
            for s, m in (payload.get("relevance") or {}).items():  # type: ignore[union-attr]
                obj.relevance[s] = defaultdict(float, {str(k): float(v) for k, v in m.items()})
            obj.total_updates = Counter({s: int(v) for s, v in (payload.get("total_updates") or {}).items()})  # type: ignore[arg-type]
        return obj

    def export_json(self, filepath: str) -> None:
        """Write a compact JSON artifact with full internal state + summary."""
        path = Path(filepath)
        payload = {
            "dump_version": 1,
            "summary": self.summary(),
            "state": self.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # --------------------------
    # Maintenance
    # --------------------------
    def reset_split(self, split: str) -> None:
        with self._lock:
            self.seen_ids.pop(split, None)
            self.exposures.pop(split, None)
            self.id_exposures.pop(split, None)
            self.relevance.pop(split, None)
            self.total_updates.pop(split, None)

    def reset_all(self) -> None:
        with self._lock:
            self.split_totals.clear()
            self.seen_ids.clear()
            self.exposures.clear()
            self.id_exposures.clear()
            self.relevance.clear()
            self.total_updates.clear()
