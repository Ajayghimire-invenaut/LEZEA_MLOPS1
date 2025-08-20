# lezea_mlops/monitoring/data_usage.py
from __future__ import annotations

from collections import defaultdict, Counter
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set
import json
import math
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class DifficultyLevel(Enum):
    """Challenge difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    CUSTOM = "custom"


@dataclass
class ChallengeMetadata:
    """Metadata for a challenge"""
    challenge_id: str
    difficulty_level: DifficultyLevel
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    sample_ids: Set[str] = field(default_factory=set)
    importance_multiplier: float = 1.0


@dataclass
class SampleRelevanceData:
    """Enhanced relevance data for a sample"""
    sample_id: str
    relevance_score: float = 0.0
    importance_weight: float = 1.0
    challenge_assignments: List[str] = field(default_factory=list)
    exposure_count: int = 0
    last_seen: Optional[datetime] = None
    delta_contributions: List[float] = field(default_factory=list)
    performance_impact: float = 0.0


class DataUsageLogger:
    """
    Enhanced data usage tracker with LeZeA-specific features.

    NEW FEATURES:
    - Challenge-specific usage rate tracking (1.5.6)
    - Per-challenge difficulty analysis
    - Sample importance weighting with automatic adjustment
    - Enhanced learning relevance with automatic scoring (1.5.7)
    - Challenge importance ranking
    - Relevance-based sample selection
    - Real-time modification stats integration

    Features:
    - Usage:
        • set_split_totals({"train": 50000, "val": 10000, "test": 10000})
        • register_challenge(challenge_id, difficulty, sample_ids)
        • update(sample_ids, split="train", delta_loss=..., challenge_id=...)
        • get_split_metrics("train")  -> {unique_seen, exposures, usage_rate, ...}
        • get_challenge_metrics(challenge_id) -> challenge-specific stats
        • top_k("train", k=20)        -> [(sample_id, score), ...]
        • summary()                    -> full dump for artifacts

    - Enhanced relevance heuristic:
        We use per-step delta_loss (loss_t - loss_{t-1}) with challenge-aware weighting.
        Only *improvements* (negative delta) are credited:
            credit_total = max(-delta_loss, 0.0) * challenge_multiplier
        Credit allocation considers both sample weights and importance weights.

    - NEW: Challenge tracking:
        • Per-challenge usage rates and difficulty analysis
        • Sample importance weights that adapt based on performance
        • Challenge ranking by importance and engagement
        • Automatic relevance scoring integrated with training loop
    """

    def __init__(self, *, max_unique_per_split: Optional[int] = None) -> None:
        # Original tracking
        self.split_totals: Dict[str, int] = {}  # declared dataset sizes
        self.seen_ids: Dict[str, set[str]] = defaultdict(set)  # unique IDs seen per split
        self.exposures: Counter[str] = Counter()  # total sample exposures per split
        self.id_exposures: Dict[str, Counter[str]] = defaultdict(Counter)  # split -> id -> exposures
        self.relevance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # split -> id -> score
        self.total_updates: Counter[str] = Counter()  # number of update() calls per split
        
        # NEW: LeZeA-specific tracking
        self.challenges: Dict[str, ChallengeMetadata] = {}  # challenge_id -> metadata
        self.challenge_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # challenge -> split -> usage
        self.challenge_sample_map: Dict[str, Set[str]] = defaultdict(set)  # challenge -> sample_ids
        self.sample_challenge_map: Dict[str, Set[str]] = defaultdict(set)  # sample -> challenge_ids
        self.sample_relevance_data: Dict[str, SampleRelevanceData] = {}  # sample_id -> enhanced data
        self.importance_weights: Dict[str, float] = {}  # sample_id -> importance weight
        self.challenge_performance: Dict[str, List[float]] = defaultdict(list)  # challenge -> performance history
        self.automatic_relevance_enabled: bool = True
        self.relevance_decay_factor: float = 0.95  # Decay old relevance scores
        self.importance_learning_rate: float = 0.01  # Rate of importance weight adaptation
        
        # Configuration
        self.max_unique_per_split = max_unique_per_split
        self._lock = RLock()

    # --------------------------
    # NEW: Challenge management
    # --------------------------
    def register_challenge(
        self,
        challenge_id: str,
        difficulty_level: DifficultyLevel,
        sample_ids: Iterable[str],
        description: str = "",
        importance_multiplier: float = 1.0
    ) -> None:
        """Register a new challenge with its samples and difficulty."""
        with self._lock:
            sample_set = set(str(sid) for sid in sample_ids)
            
            challenge = ChallengeMetadata(
                challenge_id=challenge_id,
                difficulty_level=difficulty_level,
                description=description,
                sample_ids=sample_set,
                importance_multiplier=importance_multiplier
            )
            
            self.challenges[challenge_id] = challenge
            self.challenge_sample_map[challenge_id] = sample_set
            
            # Update sample-challenge mappings
            for sample_id in sample_set:
                self.sample_challenge_map[sample_id].add(challenge_id)
                
                # Initialize sample relevance data if not exists
                if sample_id not in self.sample_relevance_data:
                    self.sample_relevance_data[sample_id] = SampleRelevanceData(
                        sample_id=sample_id,
                        importance_weight=importance_multiplier
                    )
                
                # Update challenge assignments
                if challenge_id not in self.sample_relevance_data[sample_id].challenge_assignments:
                    self.sample_relevance_data[sample_id].challenge_assignments.append(challenge_id)
                
                # Set initial importance weight
                self.importance_weights[sample_id] = importance_multiplier

    def get_challenge_metadata(self, challenge_id: str) -> Optional[ChallengeMetadata]:
        """Get metadata for a specific challenge."""
        with self._lock:
            return self.challenges.get(challenge_id)

    def list_challenges(self) -> List[str]:
        """List all registered challenge IDs."""
        with self._lock:
            return list(self.challenges.keys())

    def get_sample_challenges(self, sample_id: str) -> List[str]:
        """Get all challenges that contain this sample."""
        with self._lock:
            return list(self.sample_challenge_map.get(str(sample_id), set()))

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

    def configure_relevance_settings(
        self,
        automatic_enabled: bool = True,
        decay_factor: float = 0.95,
        learning_rate: float = 0.01
    ) -> None:
        """Configure automatic relevance scoring parameters."""
        with self._lock:
            self.automatic_relevance_enabled = automatic_enabled
            self.relevance_decay_factor = max(0.0, min(1.0, decay_factor))
            self.importance_learning_rate = max(0.0, min(1.0, learning_rate))

    # --------------------------
    # Enhanced Updates
    # --------------------------
    def update(
        self,
        sample_ids: Iterable[Any],
        split: str,
        delta_loss: Optional[float] = None,
        sample_weights: Optional[Iterable[float]] = None,
        challenge_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Enhanced batch usage update with challenge tracking.

        Args:
            sample_ids: iterable of sample identifiers (any -> str)
            split: split name ('train', 'val', etc.)
            delta_loss: loss_t - loss_{t-1}; negative means improvement
            sample_weights: optional weights (same length as sample_ids)
            challenge_id: optional challenge this batch belongs to
            performance_metrics: optional additional performance data
        """
        ids = [str(s) for s in (sample_ids or [])]
        if not ids or not split:
            return

        with self._lock:
            # Original exposures & seen sets
            self.exposures[split] += len(ids)
            self.total_updates[split] += 1

            # Challenge-specific tracking
            if challenge_id and challenge_id in self.challenges:
                self.challenge_usage[challenge_id][split] += len(ids)
                
                # Track challenge performance
                if performance_metrics:
                    avg_performance = sum(performance_metrics.values()) / len(performance_metrics)
                    self.challenge_performance[challenge_id].append(avg_performance)

            # Cap unique IDs per split if requested
            cap = self.max_unique_per_split
            if cap is None or len(self.seen_ids[split]) < cap:
                remaining = None if cap is None else max(cap - len(self.seen_ids[split]), 0)
                if remaining is None:
                    addable = ids
                else:
                    existing = [i for i in ids if i in self.seen_ids[split]]
                    new_ids = [i for i in ids if i not in self.seen_ids[split]][:remaining]
                    addable = existing + new_ids
                self.seen_ids[split].update(addable)
            else:
                addable = [i for i in ids if i in self.seen_ids[split]]

            # Enhanced per-id exposures and relevance data
            current_time = datetime.now()
            for i, sid in enumerate(ids):
                self.id_exposures[split][sid] += 1
                
                # Update enhanced sample data
                if sid not in self.sample_relevance_data:
                    self.sample_relevance_data[sid] = SampleRelevanceData(sample_id=sid)
                
                sample_data = self.sample_relevance_data[sid]
                sample_data.exposure_count += 1
                sample_data.last_seen = current_time
                
                # Apply relevance decay
                if self.automatic_relevance_enabled:
                    sample_data.relevance_score *= self.relevance_decay_factor

            # Enhanced relevance attribution with challenge awareness
            if delta_loss is not None:
                self._update_relevance_scores(ids, split, delta_loss, sample_weights, challenge_id, performance_metrics)

    def _update_relevance_scores(
        self,
        sample_ids: List[str],
        split: str,
        delta_loss: float,
        sample_weights: Optional[Iterable[float]],
        challenge_id: Optional[str],
        performance_metrics: Optional[Dict[str, float]]
    ) -> None:
        """Update relevance scores with challenge-aware weighting."""
        # Base credit (only on improvement)
        base_credit = max(-delta_loss, 0.0) if delta_loss < 0.0 else 0.0
        
        if base_credit <= 0.0:
            return

        # Challenge multiplier
        challenge_multiplier = 1.0
        if challenge_id and challenge_id in self.challenges:
            challenge_multiplier = self.challenges[challenge_id].importance_multiplier

        total_credit = base_credit * challenge_multiplier
        rmap = self.relevance[split]

        # Distribute credit with importance weighting
        if sample_weights is not None:
            weights = [float(w) for w in sample_weights]
            if len(weights) != len(sample_ids):
                # Fallback to equal distribution
                self._distribute_credit_equally(sample_ids, total_credit, rmap)
            else:
                self._distribute_credit_weighted(sample_ids, weights, total_credit, rmap)
        else:
            # Use importance weights if available
            importance_weights = [self.importance_weights.get(sid, 1.0) for sid in sample_ids]
            self._distribute_credit_weighted(sample_ids, importance_weights, total_credit, rmap)

        # Update individual sample relevance data
        avg_credit = total_credit / len(sample_ids)
        for i, sid in enumerate(sample_ids):
            sample_data = self.sample_relevance_data[sid]
            sample_data.delta_contributions.append(avg_credit)
            sample_data.performance_impact += avg_credit
            
            # Adapt importance weights based on performance
            if self.automatic_relevance_enabled:
                current_importance = self.importance_weights.get(sid, 1.0)
                performance_factor = 1.0 + (avg_credit * self.importance_learning_rate)
                new_importance = current_importance * performance_factor
                self.importance_weights[sid] = max(0.1, min(5.0, new_importance))  # Clamp to reasonable range

    def _distribute_credit_equally(self, sample_ids: List[str], total_credit: float, rmap: Dict[str, float]) -> None:
        """Distribute credit equally among samples."""
        share = total_credit / max(1, len(sample_ids))
        for sid in sample_ids:
            rmap[sid] += share

    def _distribute_credit_weighted(self, sample_ids: List[str], weights: List[float], total_credit: float, rmap: Dict[str, float]) -> None:
        """Distribute credit proportionally based on weights."""
        weight_sum = sum(abs(w) for w in weights)
        if weight_sum <= 0:
            self._distribute_credit_equally(sample_ids, total_credit, rmap)
            return
        
        for sid, weight in zip(sample_ids, weights):
            credit = total_credit * (abs(weight) / weight_sum)
            rmap[sid] += credit

    def update_relevance(self, sample_id: str, relevance_score: float) -> None:
        """Manually update relevance score for a sample (used by tracker)."""
        with self._lock:
            sid = str(sample_id)
            if sid not in self.sample_relevance_data:
                self.sample_relevance_data[sid] = SampleRelevanceData(sample_id=sid)
            
            self.sample_relevance_data[sid].relevance_score = float(relevance_score)

    # --------------------------
    # NEW: Challenge-specific metrics
    # --------------------------
    def get_challenge_metrics(self, challenge_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a specific challenge."""
        with self._lock:
            if challenge_id not in self.challenges:
                return {"error": f"Challenge {challenge_id} not found"}

            challenge = self.challenges[challenge_id]
            sample_ids = self.challenge_sample_map[challenge_id]
            
            # Usage across splits
            usage_by_split = {}
            total_usage = 0
            for split in self.split_totals.keys():
                usage = self.challenge_usage[challenge_id].get(split, 0)
                usage_by_split[split] = usage
                total_usage += usage

            # Sample-level metrics
            sample_metrics = []
            total_relevance = 0.0
            for sid in sample_ids:
                if sid in self.sample_relevance_data:
                    data = self.sample_relevance_data[sid]
                    sample_metrics.append({
                        "sample_id": sid,
                        "relevance_score": data.relevance_score,
                        "importance_weight": data.importance_weight,
                        "exposure_count": data.exposure_count,
                        "performance_impact": data.performance_impact
                    })
                    total_relevance += data.relevance_score

            # Performance history
            performance_history = self.challenge_performance.get(challenge_id, [])
            avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.0

            return {
                "challenge_id": challenge_id,
                "difficulty_level": challenge.difficulty_level.value,
                "description": challenge.description,
                "sample_count": len(sample_ids),
                "importance_multiplier": challenge.importance_multiplier,
                "usage_by_split": usage_by_split,
                "total_usage": total_usage,
                "usage_rate": total_usage / len(sample_ids) if sample_ids else 0.0,
                "avg_relevance": total_relevance / len(sample_ids) if sample_ids else 0.0,
                "avg_performance": avg_performance,
                "performance_history": performance_history,
                "sample_metrics": sample_metrics
            }

    def get_challenge_difficulty_analysis(self) -> Dict[str, Any]:
        """Analyze usage rates and performance by difficulty level."""
        with self._lock:
            difficulty_stats = defaultdict(lambda: {
                "challenge_count": 0,
                "total_samples": 0,
                "total_usage": 0,
                "avg_relevance": 0.0,
                "avg_performance": 0.0,
                "challenges": []
            })

            for challenge_id, challenge in self.challenges.items():
                difficulty = challenge.difficulty_level.value
                metrics = self.get_challenge_metrics(challenge_id)
                
                stats = difficulty_stats[difficulty]
                stats["challenge_count"] += 1
                stats["total_samples"] += metrics["sample_count"]
                stats["total_usage"] += metrics["total_usage"]
                stats["avg_relevance"] += metrics["avg_relevance"]
                stats["avg_performance"] += metrics["avg_performance"]
                stats["challenges"].append(challenge_id)

            # Calculate averages
            for difficulty, stats in difficulty_stats.items():
                count = stats["challenge_count"]
                if count > 0:
                    stats["avg_relevance"] /= count
                    stats["avg_performance"] /= count
                    stats["avg_usage_rate"] = stats["total_usage"] / max(stats["total_samples"], 1)

            return dict(difficulty_stats)

    def get_challenge_ranking(self) -> List[Dict[str, Any]]:
        """Get challenges ranked by importance and performance."""
        with self._lock:
            rankings = []
            
            for challenge_id in self.challenges.keys():
                metrics = self.get_challenge_metrics(challenge_id)
                
                # Calculate importance score
                importance_score = (
                    metrics["avg_relevance"] * 0.4 +
                    metrics["avg_performance"] * 0.3 +
                    metrics["usage_rate"] * 0.2 +
                    metrics["importance_multiplier"] * 0.1
                )
                
                rankings.append({
                    "challenge_id": challenge_id,
                    "importance_score": importance_score,
                    "difficulty_level": metrics["difficulty_level"],
                    "avg_relevance": metrics["avg_relevance"],
                    "avg_performance": metrics["avg_performance"],
                    "usage_rate": metrics["usage_rate"],
                    "sample_count": metrics["sample_count"]
                })
            
            # Sort by importance score descending
            rankings.sort(key=lambda x: x["importance_score"], reverse=True)
            return rankings

    # --------------------------
    # NEW: Relevance-based sample selection
    # --------------------------
    def select_high_relevance_samples(self, split: str, k: int = 100, min_relevance: float = 0.5) -> List[str]:
        """Select samples with high relevance scores for focused training."""
        with self._lock:
            candidates = []
            rmap = self.relevance.get(split, {})
            
            for sample_id, relevance_score in rmap.items():
                if relevance_score >= min_relevance:
                    importance_weight = self.importance_weights.get(sample_id, 1.0)
                    combined_score = relevance_score * importance_weight
                    candidates.append((sample_id, combined_score))
            
            # Sort by combined score and return top k
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [sample_id for sample_id, _ in candidates[:k]]

    def select_challenge_samples(self, challenge_id: str, k: int = 50, strategy: str = "balanced") -> List[str]:
        """Select samples from a specific challenge using different strategies."""
        with self._lock:
            if challenge_id not in self.challenges:
                return []
            
            sample_ids = list(self.challenge_sample_map[challenge_id])
            
            if strategy == "random":
                import random
                return random.sample(sample_ids, min(k, len(sample_ids)))
            
            elif strategy == "high_relevance":
                candidates = []
                for sid in sample_ids:
                    if sid in self.sample_relevance_data:
                        relevance = self.sample_relevance_data[sid].relevance_score
                        candidates.append((sid, relevance))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                return [sid for sid, _ in candidates[:k]]
            
            elif strategy == "low_exposure":
                candidates = []
                for sid in sample_ids:
                    if sid in self.sample_relevance_data:
                        exposure = self.sample_relevance_data[sid].exposure_count
                        candidates.append((sid, -exposure))  # Negative for ascending sort
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                return [sid for sid, _ in candidates[:k]]
            
            else:  # balanced
                # Mix of high relevance and low exposure
                high_rel = self.select_challenge_samples(challenge_id, k//2, "high_relevance")
                low_exp = self.select_challenge_samples(challenge_id, k//2, "low_exposure")
                return list(set(high_rel + low_exp))[:k]

    # --------------------------
    # Enhanced Metrics / Reports
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
            
            # Enhanced relevance metrics
            rmap = self.relevance.get(split, {})
            if rmap:
                out["relevance_sum"] = float(sum(rmap.values()))
                out["relevance_mean"] = float(out["relevance_sum"] / max(1.0, len(rmap)))
                out["high_relevance_count"] = float(len([r for r in rmap.values() if r > 0.7]))
                out["high_relevance_ratio"] = out["high_relevance_count"] / max(1.0, len(rmap))
            
            # Challenge-specific metrics for this split
            challenge_usage_in_split = 0
            for challenge_id in self.challenges.keys():
                challenge_usage_in_split += self.challenge_usage[challenge_id].get(split, 0)
            out["challenge_usage_total"] = float(challenge_usage_in_split)
            
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
            # Enhanced sorting with importance weights
            candidates = []
            for sample_id, relevance in rmap.items():
                importance = self.importance_weights.get(sample_id, 1.0)
                combined_score = relevance * importance
                exposures = self.id_exposures[split].get(sample_id, 0)
                candidates.append((sample_id, combined_score, exposures, relevance))
            
            # Sort by combined score, then exposures, then sample_id for stable tie-breaking
            candidates.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
            return [(sample_id, relevance) for sample_id, _, _, relevance in candidates[:max(1, k)]]

    def get_sample_metrics(self, split: str, sample_id: Any) -> Dict[str, float]:
        """Return enhanced per-sample metrics."""
        sid = str(sample_id)
        with self._lock:
            base_metrics = {
                "exposures": float(self.id_exposures.get(split, {}).get(sid, 0)),
                "relevance": float(self.relevance.get(split, {}).get(sid, 0.0)),
                "seen": float(1.0 if sid in self.seen_ids.get(split, set()) else 0.0),
            }
            
            # Enhanced metrics from sample relevance data
            if sid in self.sample_relevance_data:
                data = self.sample_relevance_data[sid]
                base_metrics.update({
                    "importance_weight": data.importance_weight,
                    "performance_impact": data.performance_impact,
                    "total_exposures": float(data.exposure_count),
                    "challenge_count": len(data.challenge_assignments),
                    "avg_delta_contribution": sum(data.delta_contributions) / max(len(data.delta_contributions), 1)
                })
            
            return base_metrics

    def distribution_stats(self, split: str) -> Dict[str, float]:
        """
        Enhanced exposure distribution fairness metrics for a split.
        Returns: {gini, min, max, p50, p90, p95, n, importance_weighted_gini}
        """
        with self._lock:
            counts = list(self.id_exposures.get(split, {}).values())
            sample_ids = list(self.id_exposures.get(split, {}).keys())
        
        if not counts:
            return {"gini": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "n": 0.0, "importance_weighted_gini": 0.0}
        
        counts.sort()
        n = len(counts)
        s = sum(counts)
        
        # Standard Gini coefficient
        if s <= 0:
            gini = 0.0
        else:
            cum = 0
            for i, x in enumerate(counts, start=1):
                cum += i * x
            gini = (2.0 * cum) / (n * s) - (n + 1.0) / n
        
        # Importance-weighted Gini
        weighted_counts = []
        for sid in sample_ids:
            exposure = self.id_exposures[split].get(sid, 0)
            importance = self.importance_weights.get(sid, 1.0)
            weighted_counts.append(exposure * importance)
        
        weighted_counts.sort()
        ws = sum(weighted_counts)
        if ws <= 0:
            weighted_gini = 0.0
        else:
            wcum = 0
            for i, x in enumerate(weighted_counts, start=1):
                wcum += i * x
            weighted_gini = (2.0 * wcum) / (n * ws) - (n + 1.0) / n
        
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
            "importance_weighted_gini": float(weighted_gini),
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
                    "challenge_usage": {cid: self.challenge_usage[cid].get(split, 0) for cid in self.challenges.keys()}
                }
            
            return {
                "split_totals": dict(self.split_totals),
                "splits": sums,
                "total_updates": dict(self.total_updates),
                "challenges": {cid: self.get_challenge_metrics(cid) for cid in self.challenges.keys()},
                "challenge_difficulty_analysis": self.get_challenge_difficulty_analysis(),
                "challenge_ranking": self.get_challenge_ranking(),
                "total_challenges": len(self.challenges),
                "total_samples_with_importance": len(self.importance_weights),
                "avg_importance_weight": sum(self.importance_weights.values()) / max(len(self.importance_weights), 1),
                "relevance_settings": {
                    "automatic_enabled": self.automatic_relevance_enabled,
                    "decay_factor": self.relevance_decay_factor,
                    "learning_rate": self.importance_learning_rate
                }
            }

    # --------------------------
    # Enhanced Persistence / I/O
    # --------------------------
    def to_dict(self) -> Dict[str, object]:
        with self._lock:
            # Convert challenge metadata to serializable format
            challenges_dict = {}
            for cid, challenge in self.challenges.items():
                challenges_dict[cid] = {
                    "challenge_id": challenge.challenge_id,
                    "difficulty_level": challenge.difficulty_level.value,
                    "description": challenge.description,
                    "created_at": challenge.created_at.isoformat(),
                    "sample_ids": list(challenge.sample_ids),
                    "importance_multiplier": challenge.importance_multiplier
                }
            
            # Convert sample relevance data to serializable format
            sample_data_dict = {}
            for sid, data in self.sample_relevance_data.items():
                sample_data_dict[sid] = {
                    "sample_id": data.sample_id,
                    "relevance_score": data.relevance_score,
                    "importance_weight": data.importance_weight,
                    "challenge_assignments": data.challenge_assignments,
                    "exposure_count": data.exposure_count,
                    "last_seen": data.last_seen.isoformat() if data.last_seen else None,
                    "delta_contributions": data.delta_contributions,
                    "performance_impact": data.performance_impact
                }

            return {
                # Original data
                "split_totals": dict(self.split_totals),
                "seen_ids": {s: list(v) for s, v in self.seen_ids.items()},
                "exposures": dict(self.exposures),
                "id_exposures": {s: dict(c) for s, c in self.id_exposures.items()},
                "relevance": {s: dict(m) for s, m in self.relevance.items()},
                "total_updates": dict(self.total_updates),
                "max_unique_per_split": self.max_unique_per_split,
                
                # NEW: LeZeA-specific data
                "challenges": challenges_dict,
                "challenge_usage": {cid: dict(usage) for cid, usage in self.challenge_usage.items()},
                "challenge_sample_map": {cid: list(samples) for cid, samples in self.challenge_sample_map.items()},
                "sample_challenge_map": {sid: list(challenges) for sid, challenges in self.sample_challenge_map.items()},
                "sample_relevance_data": sample_data_dict,
                "importance_weights": dict(self.importance_weights),
                "challenge_performance": {cid: perf for cid, perf in self.challenge_performance.items()},
                "automatic_relevance_enabled": self.automatic_relevance_enabled,
                "relevance_decay_factor": self.relevance_decay_factor,
                "importance_learning_rate": self.importance_learning_rate
            }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DataUsageLogger":
        obj = cls(max_unique_per_split=payload.get("max_unique_per_split"))  # type: ignore[arg-type]
        
        with obj._lock:
            # Original data
            obj.split_totals = {str(k): int(v) for k, v in (payload.get("split_totals") or {}).items()}  # type: ignore[union-attr]
            obj.seen_ids = defaultdict(set, {s: set(v) for s, v in (payload.get("seen_ids") or {}).items()})  # type: ignore[arg-type]
            obj.exposures = Counter({s: int(v) for s, v in (payload.get("exposures") or {}).items()})  # type: ignore[arg-type]
            obj.id_exposures = defaultdict(Counter, {s: Counter(d) for s, d in (payload.get("id_exposures") or {}).items()})  # type: ignore[arg-type]
            obj.relevance = defaultdict(lambda: defaultdict(float))
            for s, m in (payload.get("relevance") or {}).items():  # type: ignore[union-attr]
                obj.relevance[s] = defaultdict(float, {str(k): float(v) for k, v in m.items()})
            obj.total_updates = Counter({s: int(v) for s, v in (payload.get("total_updates") or {}).items()})  # type: ignore[arg-type]
            
            # NEW: LeZeA-specific data restoration
            # Restore challenges
            challenges_data = payload.get("challenges") or {}
            for cid, cdata in challenges_data.items():  # type: ignore[union-attr]
                try:
                    challenge = ChallengeMetadata(
                        challenge_id=str(cdata["challenge_id"]),
                        difficulty_level=DifficultyLevel(str(cdata["difficulty_level"])),
                        description=str(cdata.get("description", "")),
                        created_at=datetime.fromisoformat(str(cdata["created_at"])),
                        sample_ids=set(cdata.get("sample_ids", [])),
                        importance_multiplier=float(cdata.get("importance_multiplier", 1.0))
                    )
                    obj.challenges[str(cid)] = challenge
                except Exception:
                    pass  # Skip malformed challenge data
            
            # Restore challenge mappings
            obj.challenge_usage = defaultdict(lambda: defaultdict(int))
            for cid, usage_data in (payload.get("challenge_usage") or {}).items():  # type: ignore[union-attr]
                for split, count in usage_data.items():
                    obj.challenge_usage[str(cid)][str(split)] = int(count)
            
            obj.challenge_sample_map = defaultdict(set)
            for cid, samples in (payload.get("challenge_sample_map") or {}).items():  # type: ignore[union-attr]
                obj.challenge_sample_map[str(cid)] = set(samples)
            
            obj.sample_challenge_map = defaultdict(set)
            for sid, challenges in (payload.get("sample_challenge_map") or {}).items():  # type: ignore[union-attr]
                obj.sample_challenge_map[str(sid)] = set(challenges)
            
            # Restore sample relevance data
            sample_data = payload.get("sample_relevance_data") or {}
            for sid, sdata in sample_data.items():  # type: ignore[union-attr]
                try:
                    relevance_data = SampleRelevanceData(
                        sample_id=str(sdata["sample_id"]),
                        relevance_score=float(sdata.get("relevance_score", 0.0)),
                        importance_weight=float(sdata.get("importance_weight", 1.0)),
                        challenge_assignments=list(sdata.get("challenge_assignments", [])),
                        exposure_count=int(sdata.get("exposure_count", 0)),
                        last_seen=datetime.fromisoformat(str(sdata["last_seen"])) if sdata.get("last_seen") else None,
                        delta_contributions=list(sdata.get("delta_contributions", [])),
                        performance_impact=float(sdata.get("performance_impact", 0.0))
                    )
                    obj.sample_relevance_data[str(sid)] = relevance_data
                except Exception:
                    pass  # Skip malformed sample data
            
            # Restore other LeZeA data
            obj.importance_weights = {str(k): float(v) for k, v in (payload.get("importance_weights") or {}).items()}  # type: ignore[union-attr]
            
            obj.challenge_performance = defaultdict(list)
            for cid, perf_list in (payload.get("challenge_performance") or {}).items():  # type: ignore[union-attr]
                obj.challenge_performance[str(cid)] = list(perf_list)
            
            # Restore settings
            obj.automatic_relevance_enabled = bool(payload.get("automatic_relevance_enabled", True))
            obj.relevance_decay_factor = float(payload.get("relevance_decay_factor", 0.95))
            obj.importance_learning_rate = float(payload.get("importance_learning_rate", 0.01))
        
        return obj

    def export_json(self, filepath: str) -> None:
        """Write a comprehensive JSON artifact with full internal state + enhanced summary."""
        path = Path(filepath)
        payload = {
            "dump_version": 2,  # Incremented for LeZeA features
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.summary(),
            "state": self.to_dict(),
            "metadata": {
                "total_challenges": len(self.challenges),
                "total_samples_tracked": len(self.sample_relevance_data),
                "difficulty_levels": list(set(c.difficulty_level.value for c in self.challenges.values())),
                "splits_tracked": list(self.split_totals.keys()),
                "features": [
                    "challenge_tracking",
                    "importance_weighting", 
                    "automatic_relevance",
                    "difficulty_analysis",
                    "sample_selection"
                ]
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    # --------------------------
    # NEW: Advanced analytics methods
    # --------------------------
    def analyze_sample_progression(self, sample_id: str) -> Dict[str, Any]:
        """Analyze how a specific sample's metrics have progressed over time."""
        with self._lock:
            sid = str(sample_id)
            if sid not in self.sample_relevance_data:
                return {"error": f"Sample {sid} not found"}
            
            data = self.sample_relevance_data[sid]
            
            # Calculate progression metrics
            if len(data.delta_contributions) > 1:
                recent_contributions = data.delta_contributions[-10:]  # Last 10 contributions
                early_contributions = data.delta_contributions[:10]   # First 10 contributions
                
                recent_avg = sum(recent_contributions) / len(recent_contributions)
                early_avg = sum(early_contributions) / len(early_contributions)
                improvement_trend = recent_avg - early_avg
            else:
                improvement_trend = 0.0
            
            return {
                "sample_id": sid,
                "current_relevance": data.relevance_score,
                "current_importance": data.importance_weight,
                "total_exposures": data.exposure_count,
                "performance_impact": data.performance_impact,
                "challenge_assignments": data.challenge_assignments,
                "contribution_history": data.delta_contributions,
                "improvement_trend": improvement_trend,
                "last_seen": data.last_seen.isoformat() if data.last_seen else None
            }

    def get_underperforming_challenges(self, min_samples: int = 10) -> List[Dict[str, Any]]:
        """Identify challenges that may need attention based on low performance metrics."""
        with self._lock:
            underperforming = []
            
            for challenge_id in self.challenges.keys():
                metrics = self.get_challenge_metrics(challenge_id)
                
                if (metrics["sample_count"] >= min_samples and
                    (metrics["avg_relevance"] < 0.3 or 
                     metrics["usage_rate"] < 0.1 or
                     metrics["avg_performance"] < 0.2)):
                    
                    underperforming.append({
                        "challenge_id": challenge_id,
                        "issues": {
                            "low_relevance": metrics["avg_relevance"] < 0.3,
                            "low_usage": metrics["usage_rate"] < 0.1,
                            "low_performance": metrics["avg_performance"] < 0.2
                        },
                        "metrics": metrics
                    })
            
            return sorted(underperforming, key=lambda x: x["metrics"]["avg_performance"])

    def suggest_training_focus(self, split: str = "train", focus_size: int = 1000) -> Dict[str, Any]:
        """Suggest which samples to focus on for improved training efficiency."""
        with self._lock:
            suggestions = {
                "high_impact_samples": self.select_high_relevance_samples(split, focus_size // 2, 0.5),
                "underexposed_samples": [],
                "challenge_focus": {},
                "strategy_recommendations": []
            }
            
            # Find underexposed high-relevance samples
            rmap = self.relevance.get(split, {})
            exposure_map = self.id_exposures.get(split, {})
            
            underexposed = []
            for sid, relevance in rmap.items():
                if relevance > 0.3:  # Some relevance
                    exposures = exposure_map.get(sid, 0)
                    if exposures < 5:  # Low exposure
                        underexposed.append((sid, relevance / max(exposures, 1)))
            
            underexposed.sort(key=lambda x: x[1], reverse=True)
            suggestions["underexposed_samples"] = [sid for sid, _ in underexposed[:focus_size // 4]]
            
            # Challenge-specific focus recommendations
            challenge_ranking = self.get_challenge_ranking()
            for i, challenge_data in enumerate(challenge_ranking[:5]):  # Top 5 challenges
                cid = challenge_data["challenge_id"]
                if challenge_data["avg_relevance"] > 0.4:
                    focus_samples = self.select_challenge_samples(cid, focus_size // 10, "balanced")
                    suggestions["challenge_focus"][cid] = {
                        "samples": focus_samples,
                        "priority": i + 1,
                        "reason": "high_relevance_challenge"
                    }
            
            # Strategy recommendations
            avg_relevance = sum(rmap.values()) / max(len(rmap), 1)
            if avg_relevance < 0.3:
                suggestions["strategy_recommendations"].append("Consider increasing learning rate or adjusting loss function")
            
            if len(suggestions["underexposed_samples"]) > focus_size // 2:
                suggestions["strategy_recommendations"].append("Focus on underexposed samples to improve coverage")
            
            total_challenges = len(self.challenges)
            active_challenges = len([c for c in challenge_ranking if c["usage_rate"] > 0.05])
            if active_challenges < total_challenges * 0.5:
                suggestions["strategy_recommendations"].append("Consider activating more challenge types for diversity")
            
            return suggestions

    # --------------------------
    # Maintenance with LeZeA support
    # --------------------------
    def reset_split(self, split: str) -> None:
        with self._lock:
            # Original reset
            self.seen_ids.pop(split, None)
            self.exposures.pop(split, None)
            self.id_exposures.pop(split, None)
            self.relevance.pop(split, None)
            self.total_updates.pop(split, None)
            
            # Reset challenge usage for this split
            for challenge_id in self.challenges.keys():
                self.challenge_usage[challenge_id].pop(split, None)

    def reset_challenge(self, challenge_id: str) -> None:
        """Reset a specific challenge's data."""
        with self._lock:
            if challenge_id not in self.challenges:
                return
            
            # Remove challenge
            challenge = self.challenges.pop(challenge_id)
            
            # Clean up mappings
            sample_ids = self.challenge_sample_map.pop(challenge_id, set())
            self.challenge_usage.pop(challenge_id, None)
            self.challenge_performance.pop(challenge_id, None)
            
            # Update sample-challenge mappings
            for sid in sample_ids:
                if sid in self.sample_challenge_map:
                    self.sample_challenge_map[sid].discard(challenge_id)
                    if not self.sample_challenge_map[sid]:
                        self.sample_challenge_map.pop(sid, None)
                
                # Reset sample data if no other challenges
                if sid in self.sample_relevance_data:
                    data = self.sample_relevance_data[sid]
                    if challenge_id in data.challenge_assignments:
                        data.challenge_assignments.remove(challenge_id)
                    
                    # If sample has no more challenges, reset importance weight
                    if not data.challenge_assignments:
                        self.importance_weights[sid] = 1.0

    def reset_all(self) -> None:
        with self._lock:
            # Original reset
            self.split_totals.clear()
            self.seen_ids.clear()
            self.exposures.clear()
            self.id_exposures.clear()
            self.relevance.clear()
            self.total_updates.clear()
            
            # NEW: Reset LeZeA data
            self.challenges.clear()
            self.challenge_usage.clear()
            self.challenge_sample_map.clear()
            self.sample_challenge_map.clear()
            self.sample_relevance_data.clear()
            self.importance_weights.clear()
            self.challenge_performance.clear()
            
            # Reset to defaults
            self.automatic_relevance_enabled = True
            self.relevance_decay_factor = 0.95
            self.importance_learning_rate = 0.01

    def cleanup_stale_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old sample data and optimize memory usage."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            cleaned = {
                "stale_samples": 0,
                "empty_challenges": 0,
                "reset_importance_weights": 0
            }
            
            # Remove stale sample data
            stale_samples = []
            for sid, data in self.sample_relevance_data.items():
                if data.last_seen and data.last_seen < cutoff_time:
                    stale_samples.append(sid)
            
            for sid in stale_samples:
                self.sample_relevance_data.pop(sid, None)
                self.importance_weights.pop(sid, None)
                cleaned["stale_samples"] += 1
            
            # Remove empty challenges
            empty_challenges = []
            for cid, challenge in self.challenges.items():
                if not challenge.sample_ids:
                    empty_challenges.append(cid)
            
            for cid in empty_challenges:
                self.reset_challenge(cid)
                cleaned["empty_challenges"] += 1
            
            # Reset extreme importance weights
            for sid, weight in list(self.importance_weights.items()):
                if weight > 10.0 or weight < 0.01:
                    self.importance_weights[sid] = 1.0
                    cleaned["reset_importance_weights"] += 1
            
            return cleaned