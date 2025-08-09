# lezea_mlops/monitoring/data_usage.py
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Optional, Tuple


class DataUsageLogger:
    """
    Tracks per-split data usage and a simple "learning relevance" score.

    - Usage:
        • set_split_totals({"train": 50000, "val": 10000, "test": 10000})
        • update(sample_ids, split="train", delta_loss=... )
        • get_split_metrics("train")  -> {unique_seen, exposures, usage_rate?}
        • top_k("train", k=20)        -> [(sample_id, score), ...]
        • summary()                    -> full dump for artifacts

    - Relevance heuristic:
        We use per-step delta_loss (loss_t - loss_{t-1}).
        Only *improvements* (negative delta) are credited:
            credit = max(-delta_loss, 0.0) / len(sample_ids)
        That credit is added to each sample in the batch.
        This is a lightweight, attribution-by-equal-share heuristic.
    """

    def __init__(self) -> None:
        self.split_totals: Dict[str, int] = {}                       # declared dataset sizes
        self.seen_ids: Dict[str, set] = defaultdict(set)             # unique IDs seen per split
        self.exposures: Counter = Counter()                          # total sample exposures per split
        self.relevance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # split -> id -> score

    # --------------------------
    # Configuration
    # --------------------------
    def set_split_totals(self, totals: Dict[str, int]) -> None:
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
    def update(self, sample_ids: Iterable[str], split: str, delta_loss: Optional[float] = None) -> None:
        ids = [str(s) for s in (sample_ids or [])]
        if not ids or not split:
            return

        # usage
        self.seen_ids[split].update(ids)
        self.exposures[split] += len(ids)

        # relevance attribution (credit only on improvement)
        credit = 0.0
        if isinstance(delta_loss, (int, float)) and delta_loss < 0.0:
            credit = float(-delta_loss) / max(1, len(ids))
        if credit > 0.0:
            rmap = self.relevance[split]
            for sid in ids:
                rmap[sid] += credit

    # --------------------------
    # Metrics / Reports
    # --------------------------
    def get_split_metrics(self, split: str) -> Dict[str, float]:
        uniq = len(self.seen_ids.get(split, ()))
        exp = int(self.exposures.get(split, 0))
        out = {"unique_seen": float(uniq), "exposures": float(exp)}
        total = self.split_totals.get(split)
        if isinstance(total, int) and total > 0:
            out["usage_rate"] = float(uniq) / float(total)
        return out

    def rates(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for split, total in self.split_totals.items():
            uniq = len(self.seen_ids.get(split, ()))
            if total > 0:
                out[split] = float(uniq) / float(total)
        return out

    def top_k(self, split: str, k: int = 20) -> List[Tuple[str, float]]:
        rmap = self.relevance.get(split, {})
        return sorted(rmap.items(), key=lambda kv: kv[1], reverse=True)[:max(1, k)]

    def summary(self) -> Dict[str, object]:
        sums = {}
        for split, rmap in self.relevance.items():
            sums[split] = {
                "unique_seen": len(self.seen_ids.get(split, ())),
                "exposures": int(self.exposures.get(split, 0)),
                "usage_rate": (len(self.seen_ids[split]) / self.split_totals[split])
                if self.split_totals.get(split) else None,
                "top_relevant": self.top_k(split, k=50),
            }
        return {
            "split_totals": dict(self.split_totals),
            "splits": sums,
        }
