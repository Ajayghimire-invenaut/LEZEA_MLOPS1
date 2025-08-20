"""
Enhanced Modification Trees for LeZeA
====================================

Comprehensive utilities to build, update, analyze, and export **modification trees**
with **real-time statistics**, **acceptance/rejection tracking**, and **LeZeA integration**.

Enhanced features for LeZeA spec items:
- 1.5.2  Modification trees and paths (ENHANCED)
- 1.5.3  Modification statistics (ENHANCED)
- Real-time modification stats integration
- Acceptance/rejection tracking with detailed analytics
- Network modification change tracking
- Component-level modification attribution

## Concepts
- **Node**: a single modification (e.g., mutate layer width, swap activation).
- **Edge**: parent→child relation showing lineage of changes.
- **Scope**: optional tag identifying builder/tasker/algorithm/network/layer.
- **ModificationSession**: Real-time tracking of modification batches.
- **AcceptanceTracker**: Detailed acceptance/rejection analytics.

## Enhanced JSON format (stable)
{
  "nodes": [
    {
      "id": "n1",                 # unique
      "parent_id": null,           # root if null
      "op": "set_lr",             # operation type
      "params": {"lr": 0.001},    # op payload (JSON-serializable)
      "score": 0.73,               # optional fitness/metric
      "accepted": true,            # whether kept after evaluation
      "step": 12,                  # global training step when created
      "timestamp": "ISO-8601",
      "tags": {"network_id": "A"},# arbitrary labels
      "scope": {"level": "tasker", "entity_id": "T1"},
      "component_id": "layer_3",   # NEW: component that was modified
      "modification_type": "architecture", # NEW: type classification
      "impact_score": 0.15,        # NEW: measured impact on performance
      "acceptance_reason": "performance_gain", # NEW: why accepted/rejected
      "evaluation_time_ms": 45.2,  # NEW: time to evaluate this modification
      "dependencies": ["n0"],      # NEW: other modifications this depends on
      "rollback_info": {...}       # NEW: information needed to undo this change
    },
    ...
  ],
  "sessions": [...],              # NEW: modification sessions/batches
  "acceptance_stats": {...},      # NEW: detailed acceptance analytics
  "real_time_stats": {...}       # NEW: real-time statistics
}

## Example Enhanced Usage
>>> tracker = ModificationTracker()
>>> 
>>> # Start a modification session
>>> with tracker.start_session("layer_optimization", step=100) as session:
...     # Add modifications with enhanced tracking
...     n1 = session.add_modification(
...         op="increase_width", 
...         params={"delta": 32, "layer": "conv2"}, 
...         component_id="layer_conv2",
...         modification_type="architecture"
...     )
...     
...     # Evaluate and set acceptance
...     performance_gain = evaluate_modification(n1)
...     session.set_acceptance(n1.id, 
...                          accepted=performance_gain > 0.05,
...                          score=performance_gain,
...                          reason="performance_gain" if performance_gain > 0.05 else "no_improvement",
...                          evaluation_time_ms=42.1)
... 
>>> # Get comprehensive statistics
>>> stats = tracker.get_comprehensive_stats()
>>> acceptance_analysis = tracker.get_acceptance_analysis()
>>> 
>>> # Export for MLOps tracking
>>> tracker.export_for_mlops(step=100)

"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, Union
from pathlib import Path
from datetime import datetime, timedelta
import json
import uuid
import time
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, Counter

# -----------------------------
# Enhanced Data classes
# -----------------------------

class ModificationType(Enum):
    """Types of modifications for better categorization"""
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    OPTIMIZATION = "optimization"
    REGULARIZATION = "regularization"
    DATA_AUGMENTATION = "data_augmentation"
    LOSS_FUNCTION = "loss_function"
    ACTIVATION = "activation"
    INITIALIZATION = "initialization"
    OTHER = "other"

class AcceptanceReason(Enum):
    """Reasons for accepting or rejecting modifications"""
    PERFORMANCE_GAIN = "performance_gain"
    STABILITY_IMPROVEMENT = "stability_improvement"
    EFFICIENCY_GAIN = "efficiency_gain"
    NO_IMPROVEMENT = "no_improvement"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    INSTABILITY = "instability"
    RESOURCE_CONSTRAINT = "resource_constraint"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class ModNode:
    id: str
    parent_id: Optional[str]
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    accepted: Optional[bool] = None
    step: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: Dict[str, Any] = field(default_factory=dict)
    scope: Optional[Dict[str, str]] = None  # {level, entity_id}
    
    # NEW: Enhanced fields for LeZeA integration
    component_id: Optional[str] = None
    modification_type: Optional[ModificationType] = None
    impact_score: Optional[float] = None  # Measured impact on performance
    acceptance_reason: Optional[AcceptanceReason] = None
    evaluation_time_ms: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # Other node IDs this depends on
    rollback_info: Dict[str, Any] = field(default_factory=dict)  # Info needed to undo
    creation_time: float = field(default_factory=time.time)
    evaluation_start_time: Optional[float] = None
    evaluation_end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert enums to their values
        if self.modification_type:
            d['modification_type'] = self.modification_type.value
        if self.acceptance_reason:
            d['acceptance_reason'] = self.acceptance_reason.value
        return d

    def set_evaluation_start(self) -> None:
        """Mark the start of evaluation for this modification."""
        self.evaluation_start_time = time.time()

    def set_evaluation_result(self, accepted: bool, score: Optional[float] = None, 
                            reason: Optional[AcceptanceReason] = None, 
                            impact_score: Optional[float] = None) -> None:
        """Set the evaluation result for this modification."""
        self.evaluation_end_time = time.time()
        self.accepted = accepted
        self.score = score
        self.acceptance_reason = reason or AcceptanceReason.UNKNOWN
        self.impact_score = impact_score
        
        if self.evaluation_start_time:
            self.evaluation_time_ms = (self.evaluation_end_time - self.evaluation_start_time) * 1000

@dataclass 
class ModificationSession:
    """Tracks a batch of related modifications"""
    session_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    step: Optional[int] = None
    modification_ids: List[str] = field(default_factory=list)
    session_tags: Dict[str, Any] = field(default_factory=dict)
    target_component: Optional[str] = None
    session_type: Optional[str] = None  # e.g., "layer_optimization", "hyperparameter_sweep"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat()
        if self.end_time:
            d['end_time'] = self.end_time.isoformat()
        return d

@dataclass
class AcceptanceStats:
    """Detailed acceptance/rejection statistics"""
    total_modifications: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    pending_count: int = 0
    acceptance_rate: float = 0.0
    avg_evaluation_time_ms: float = 0.0
    acceptance_by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    acceptance_by_component: Dict[str, Dict[str, int]] = field(default_factory=dict)
    rejection_reasons: Counter = field(default_factory=Counter)
    acceptance_reasons: Counter = field(default_factory=Counter)
    impact_score_distribution: Dict[str, List[float]] = field(default_factory=dict)

# -----------------------------
# Enhanced Tree structure
# -----------------------------

class ModTree:
    def __init__(self) -> None:
        self._nodes: Dict[str, ModNode] = {}
        self._children: Dict[str, List[str]] = {}
        self._roots: Set[str] = set()
        
        # NEW: Enhanced tracking
        self._sessions: Dict[str, ModificationSession] = {}
        self._current_session: Optional[str] = None
        self._real_time_stats: Dict[str, Any] = {}
        self._modification_timeline: List[Tuple[float, str, str]] = []  # (timestamp, node_id, event_type)

    # ---- Enhanced mutation API ----
    def add(
        self,
        op: str,
        params: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        *,
        score: Optional[float] = None,
        accepted: Optional[bool] = None,
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
        scope: Optional[Dict[str, str]] = None,
        node_id: Optional[str] = None,
        # NEW: Enhanced parameters
        component_id: Optional[str] = None,
        modification_type: Optional[Union[ModificationType, str]] = None,
        dependencies: Optional[List[str]] = None,
        rollback_info: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> ModNode:
        """Create and add a node with enhanced tracking capabilities."""
        nid = node_id or str(uuid.uuid4())
        if nid in self._nodes:
            raise ValueError(f"Node id already exists: {nid}")
        if parent_id is not None and parent_id not in self._nodes:
            raise KeyError(f"Parent id not found: {parent_id}")

        # Handle modification type
        if isinstance(modification_type, str):
            try:
                modification_type = ModificationType(modification_type)
            except ValueError:
                modification_type = ModificationType.OTHER

        node = ModNode(
            id=nid,
            parent_id=parent_id,
            op=str(op),
            params=dict(params or {}),
            score=score,
            accepted=accepted,
            step=step,
            tags=dict(tags or {}),
            scope=dict(scope) if scope else None,
            component_id=component_id,
            modification_type=modification_type,
            dependencies=list(dependencies or []),
            rollback_info=dict(rollback_info or {}),
        )
        
        self._nodes[nid] = node
        if parent_id is None:
            self._roots.add(nid)
        else:
            self._children.setdefault(parent_id, []).append(nid)
        
        # Track in current session if active
        session_id = session_id or self._current_session
        if session_id and session_id in self._sessions:
            self._sessions[session_id].modification_ids.append(nid)
        
        # Add to timeline
        self._modification_timeline.append((time.time(), nid, "created"))
        
        return node

    def bulk_add(self, nodes: Iterable[Dict[str, Any]]) -> None:
        """Add multiple nodes using the enhanced JSON schema."""
        for n in nodes:
            # Convert string enums back to enum objects
            mod_type = n.get("modification_type")
            if isinstance(mod_type, str):
                try:
                    mod_type = ModificationType(mod_type)
                except ValueError:
                    mod_type = ModificationType.OTHER
            
            self.add(
                op=n.get("op", "unknown"),
                params=n.get("params"),
                parent_id=n.get("parent_id"),
                score=n.get("score"),
                accepted=n.get("accepted"),
                step=n.get("step"),
                tags=n.get("tags"),
                scope=n.get("scope"),
                node_id=n.get("id"),
                component_id=n.get("component_id"),
                modification_type=mod_type,
                dependencies=n.get("dependencies"),
                rollback_info=n.get("rollback_info"),
            )

    def set_acceptance(self, node_id: str, accepted: bool, score: Optional[float] = None,
                      reason: Optional[Union[AcceptanceReason, str]] = None,
                      impact_score: Optional[float] = None,
                      evaluation_time_ms: Optional[float] = None) -> None:
        """Set acceptance result for a modification with detailed tracking."""
        if node_id not in self._nodes:
            raise KeyError(f"Node not found: {node_id}")
        
        node = self._nodes[node_id]
        
        # Handle reason conversion
        if isinstance(reason, str):
            try:
                reason = AcceptanceReason(reason)
            except ValueError:
                reason = AcceptanceReason.UNKNOWN
        
        node.accepted = accepted
        node.score = score
        node.acceptance_reason = reason
        node.impact_score = impact_score
        if evaluation_time_ms:
            node.evaluation_time_ms = evaluation_time_ms
        
        # Add to timeline
        event_type = "accepted" if accepted else "rejected"
        self._modification_timeline.append((time.time(), node_id, event_type))

    # ---- Enhanced queries ----
    def node(self, node_id: str) -> ModNode:
        return self._nodes[node_id]

    def children(self, node_id: str) -> List[str]:
        return list(self._children.get(node_id, []))

    def roots(self) -> List[str]:
        return list(self._roots)

    def path_to_root(self, node_id: str) -> List[str]:
        path: List[str] = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = self._nodes[cur].parent_id if cur in self._nodes else None
        return path  # node -> ... -> root

    def leaves(self) -> List[str]:
        all_ids = set(self._nodes)
        non_leaves = set(self._children)
        return list(all_ids - non_leaves)

    def get_modifications_by_component(self, component_id: str) -> List[ModNode]:
        """Get all modifications for a specific component."""
        return [node for node in self._nodes.values() if node.component_id == component_id]

    def get_modifications_by_type(self, modification_type: Union[ModificationType, str]) -> List[ModNode]:
        """Get all modifications of a specific type."""
        if isinstance(modification_type, str):
            try:
                modification_type = ModificationType(modification_type)
            except ValueError:
                return []
        return [node for node in self._nodes.values() if node.modification_type == modification_type]

    def get_pending_evaluations(self) -> List[ModNode]:
        """Get modifications that haven't been evaluated yet."""
        return [node for node in self._nodes.values() if node.accepted is None]

    def get_dependency_chain(self, node_id: str) -> List[str]:
        """Get all dependencies (direct and indirect) for a modification."""
        visited = set()
        chain = []
        
        def walk_dependencies(nid: str):
            if nid in visited or nid not in self._nodes:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for dep_id in node.dependencies:
                walk_dependencies(dep_id)
                if dep_id not in chain:
                    chain.append(dep_id)
        
        walk_dependencies(node_id)
        return chain

    # ---- Enhanced analysis & stats ----
    def depth(self, node_id: str) -> int:
        return len(self.path_to_root(node_id)) - 1

    def max_depth(self) -> int:
        return max((self.depth(n) for n in self._nodes), default=0)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._children.values())

    def _op_hist(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for n in self._nodes.values():
            hist[n.op] = hist.get(n.op, 0) + 1
        return hist

    def acceptance_rate(self) -> Optional[float]:
        labeled = [n for n in self._nodes.values() if n.accepted is not None]
        if not labeled:
            return None
        acc = sum(1 for n in labeled if bool(n.accepted)) / float(len(labeled))
        return acc

    def branching_factor(self) -> float:
        non_leaf = [nid for nid in self._nodes if nid in self._children]
        if not non_leaf:
            return 0.0
        return self.edge_count() / float(len(non_leaf))

    def step_hist(self) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for n in self._nodes.values():
            if isinstance(n.step, int):
                hist[n.step] = hist.get(n.step, 0) + 1
        return hist

    def get_acceptance_analysis(self) -> AcceptanceStats:
        """Get detailed acceptance/rejection analysis."""
        stats = AcceptanceStats()
        
        all_nodes = list(self._nodes.values())
        stats.total_modifications = len(all_nodes)
        
        evaluated_nodes = [n for n in all_nodes if n.accepted is not None]
        stats.accepted_count = sum(1 for n in evaluated_nodes if n.accepted)
        stats.rejected_count = len(evaluated_nodes) - stats.accepted_count
        stats.pending_count = stats.total_modifications - len(evaluated_nodes)
        
        if evaluated_nodes:
            stats.acceptance_rate = stats.accepted_count / len(evaluated_nodes)
        
        # Evaluation time statistics
        timed_evaluations = [n for n in evaluated_nodes if n.evaluation_time_ms is not None]
        if timed_evaluations:
            stats.avg_evaluation_time_ms = sum(n.evaluation_time_ms for n in timed_evaluations) / len(timed_evaluations)
        
        # Acceptance by type
        for node in evaluated_nodes:
            if node.modification_type:
                type_key = node.modification_type.value
                if type_key not in stats.acceptance_by_type:
                    stats.acceptance_by_type[type_key] = {"accepted": 0, "rejected": 0}
                
                if node.accepted:
                    stats.acceptance_by_type[type_key]["accepted"] += 1
                else:
                    stats.acceptance_by_type[type_key]["rejected"] += 1
        
        # Acceptance by component
        for node in evaluated_nodes:
            if node.component_id:
                comp_key = node.component_id
                if comp_key not in stats.acceptance_by_component:
                    stats.acceptance_by_component[comp_key] = {"accepted": 0, "rejected": 0}
                
                if node.accepted:
                    stats.acceptance_by_component[comp_key]["accepted"] += 1
                else:
                    stats.acceptance_by_component[comp_key]["rejected"] += 1
        
        # Reasons analysis
        for node in evaluated_nodes:
            if node.acceptance_reason:
                if node.accepted:
                    stats.acceptance_reasons[node.acceptance_reason.value] += 1
                else:
                    stats.rejection_reasons[node.acceptance_reason.value] += 1
        
        # Impact score distribution
        for node in evaluated_nodes:
            if node.impact_score is not None and node.modification_type:
                type_key = node.modification_type.value
                if type_key not in stats.impact_score_distribution:
                    stats.impact_score_distribution[type_key] = []
                stats.impact_score_distribution[type_key].append(node.impact_score)
        
        return stats

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for monitoring."""
        now = time.time()
        hour_ago = now - 3600
        
        # Recent modifications (last hour)
        recent_modifications = [
            (ts, nid, event) for ts, nid, event in self._modification_timeline
            if ts > hour_ago
        ]
        
        # Count events by type
        event_counts = Counter(event for _, _, event in recent_modifications)
        
        # Active sessions
        active_sessions = [
            session for session in self._sessions.values()
            if session.end_time is None
        ]
        
        # Pending evaluations
        pending = self.get_pending_evaluations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "recent_hour": {
                "total_events": len(recent_modifications),
                "created": event_counts.get("created", 0),
                "accepted": event_counts.get("accepted", 0),
                "rejected": event_counts.get("rejected", 0),
            },
            "current_state": {
                "total_nodes": len(self._nodes),
                "pending_evaluations": len(pending),
                "active_sessions": len(active_sessions),
                "acceptance_rate": self.acceptance_rate() or 0.0,
            },
            "performance": {
                "avg_evaluation_time_ms": self._get_avg_evaluation_time(),
                "evaluation_queue_size": len(pending),
            }
        }

    def _get_avg_evaluation_time(self) -> float:
        """Calculate average evaluation time for completed evaluations."""
        timed_nodes = [
            n for n in self._nodes.values() 
            if n.evaluation_time_ms is not None
        ]
        if not timed_nodes:
            return 0.0
        return sum(n.evaluation_time_ms for n in timed_nodes) / len(timed_nodes)

    def stats(self) -> Dict[str, Any]:
        """Enhanced statistics including LeZeA-specific metrics."""
        base_stats = {
            "node_count": len(self._nodes),
            "edge_count": self.edge_count(),
            "root_count": len(self._roots),
            "leaf_count": len(self.leaves()),
            "max_depth": self.max_depth(),
            "avg_branching_factor": self.branching_factor(),
            "op_hist": self._op_hist(),
            "acceptance_rate": self.acceptance_rate(),
            "per_step": self.step_hist(),
        }
        
        # Add enhanced stats
        acceptance_stats = self.get_acceptance_analysis()
        real_time_stats = self.get_real_time_stats()
        
        # Modification type distribution
        type_dist = Counter()
        for node in self._nodes.values():
            if node.modification_type:
                type_dist[node.modification_type.value] += 1
        
        # Component modification counts
        component_dist = Counter()
        for node in self._nodes.values():
            if node.component_id:
                component_dist[node.component_id] += 1
        
        enhanced_stats = {
            **base_stats,
            "enhanced_metrics": {
                "modification_type_distribution": dict(type_dist),
                "component_modification_distribution": dict(component_dist),
                "total_sessions": len(self._sessions),
                "avg_evaluation_time_ms": acceptance_stats.avg_evaluation_time_ms,
                "pending_evaluations": len(self.get_pending_evaluations()),
            },
            "acceptance_analysis": acceptance_stats.__dict__,
            "real_time_stats": real_time_stats,
        }
        
        return enhanced_stats

    # ---- Session management ----
    @contextmanager
    def start_session(self, name: str, step: Optional[int] = None, 
                     target_component: Optional[str] = None,
                     session_type: Optional[str] = None,
                     tags: Optional[Dict[str, Any]] = None):
        """Context manager for modification sessions."""
        session_id = str(uuid.uuid4())
        session = ModificationSession(
            session_id=session_id,
            name=name,
            start_time=datetime.now(),
            step=step,
            target_component=target_component,
            session_type=session_type,
            session_tags=dict(tags or {})
        )
        
        self._sessions[session_id] = session
        old_session = self._current_session
        self._current_session = session_id
        
        try:
            yield ModificationSessionManager(self, session_id)
        finally:
            session.end_time = datetime.now()
            self._current_session = old_session

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        
        session = self._sessions[session_id]
        modifications = [self._nodes[nid] for nid in session.modification_ids if nid in self._nodes]
        
        evaluated = [n for n in modifications if n.accepted is not None]
        accepted = [n for n in evaluated if n.accepted]
        
        return {
            "session_info": session.to_dict(),
            "total_modifications": len(modifications),
            "evaluated_modifications": len(evaluated),
            "accepted_modifications": len(accepted),
            "acceptance_rate": len(accepted) / max(len(evaluated), 1),
            "modification_types": Counter(n.modification_type.value for n in modifications if n.modification_type),
            "avg_impact_score": sum(n.impact_score for n in modifications if n.impact_score) / max(len([n for n in modifications if n.impact_score]), 1),
        }

    # ---- Enhanced export ----
    def to_json(self) -> Dict[str, Any]:
        """Enhanced JSON export with all tracking data."""
        base_export = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "stats": self.stats()
        }
        
        # Add enhanced data
        enhanced_export = {
            **base_export,
            "sessions": [session.to_dict() for session in self._sessions.values()],
            "timeline": [
                {"timestamp": ts, "node_id": nid, "event_type": event}
                for ts, nid, event in self._modification_timeline
            ],
            "export_metadata": {
                "export_time": datetime.now().isoformat(),
                "version": "2.0",  # Enhanced version
                "features": ["sessions", "acceptance_tracking", "real_time_stats", "component_tracking"]
            }
        }
        
        return enhanced_export

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ModTree":
        """Enhanced JSON import with backward compatibility."""
        t = cls()
        
        # Import nodes (same as before but with enhanced fields)
        nodes = data.get("nodes", [])
        roots = [n for n in nodes if not n.get("parent_id")]
        others = [n for n in nodes if n.get("parent_id")]
        
        for n in roots + others:
            t.add(
                op=n.get("op", "unknown"),
                params=n.get("params"),
                parent_id=n.get("parent_id"),
                score=n.get("score"),
                accepted=n.get("accepted"),
                step=n.get("step"),
                tags=n.get("tags"),
                scope=n.get("scope"),
                node_id=n.get("id"),
                component_id=n.get("component_id"),
                modification_type=n.get("modification_type"),
                dependencies=n.get("dependencies"),
                rollback_info=n.get("rollback_info"),
            )
        
        # Import sessions if available (enhanced format)
        sessions_data = data.get("sessions", [])
        for s in sessions_data:
            session = ModificationSession(
                session_id=s["session_id"],
                name=s["name"],
                start_time=datetime.fromisoformat(s["start_time"]),
                end_time=datetime.fromisoformat(s["end_time"]) if s.get("end_time") else None,
                step=s.get("step"),
                modification_ids=s.get("modification_ids", []),
                session_tags=s.get("session_tags", {}),
                target_component=s.get("target_component"),
                session_type=s.get("session_type"),
            )
            t._sessions[session.session_id] = session
        
        # Import timeline if available
        timeline_data = data.get("timeline", [])
        for event in timeline_data:
            t._modification_timeline.append((
                event["timestamp"],
                event["node_id"],
                event["event_type"]
            ))
        
        return t

    def export_for_mlops(self, step: Optional[int] = None, 
                        artifact_prefix: str = "modifications") -> Dict[str, Any]:
        """Export data optimized for MLOps tracking systems."""
        export_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_modifications": len(self._nodes),
                "acceptance_rate": self.acceptance_rate() or 0.0,
                "pending_evaluations": len(self.get_pending_evaluations()),
                "active_sessions": len([s for s in self._sessions.values() if s.end_time is None]),
            },
            "recent_activity": self.get_real_time_stats(),
            "acceptance_analysis": self.get_acceptance_analysis().__dict__,
            "artifact_files": {
                f"{artifact_prefix}/tree_full.json": self.to_json(),
                f"{artifact_prefix}/tree_summary.json": self.stats(),
                f"{artifact_prefix}/tree_visualization.dot": self.to_dot(color_by="accepted"),
            }
        }
        
        return export_data

    def to_dot(self, highlight_ids: Optional[Set[str]] = None, color_by: str = "accepted") -> str:
        """Enhanced Graphviz DOT representation with more visualization options."""
        highlight_ids = highlight_ids or set()
        lines = ["digraph mod_tree {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=white];"]
        
        # Node lines with enhanced information
        for n in self._nodes.values():
            # Enhanced label with more information
            label_parts = [f"{n.op}"]
            if n.score is not None:
                label_parts.append(f"score={n.score:.3f}")
            if n.impact_score is not None:
                label_parts.append(f"impact={n.impact_score:.3f}")
            if n.step is not None:
                label_parts.append(f"step={n.step}")
            if n.component_id:
                label_parts.append(f"comp={n.component_id}")
            if n.modification_type:
                label_parts.append(f"type={n.modification_type.value}")
            
            label = "\\n".join(label_parts)
            attrs = {"label": label}
            
            # Color coding
            if color_by == "accepted" and n.accepted is not None:
                attrs["fillcolor"] = "#d5f5e3" if n.accepted else "#f5b7b1"
            elif color_by == "type" and n.modification_type:
                # Color by modification type
                type_colors = {
                    ModificationType.ARCHITECTURE: "#e3f2fd",
                    ModificationType.HYPERPARAMETER: "#f3e5f5", 
                    ModificationType.OPTIMIZATION: "#e8f5e8",
                    ModificationType.REGULARIZATION: "#fff3e0",
                    ModificationType.DATA_AUGMENTATION: "#fce4ec",
                    ModificationType.LOSS_FUNCTION: "#e0f2f1",
                    ModificationType.ACTIVATION: "#f1f8e9",
                    ModificationType.INITIALIZATION: "#e8eaf6",
                }
                attrs["fillcolor"] = type_colors.get(n.modification_type, "#f5f5f5")
            elif color_by == "impact" and n.impact_score is not None:
                # Color by impact score (red = negative, green = positive)
                if n.impact_score > 0.1:
                    attrs["fillcolor"] = "#c8e6c9"  # Light green
                elif n.impact_score > 0.05:
                    attrs["fillcolor"] = "#e8f5e8"  # Very light green
                elif n.impact_score < -0.1:
                    attrs["fillcolor"] = "#ffcdd2"  # Light red
                elif n.impact_score < -0.05:
                    attrs["fillcolor"] = "#ffebee"  # Very light red
                else:
                    attrs["fillcolor"] = "#f5f5f5"  # Neutral gray
            
            # Highlight special nodes
            if n.id in highlight_ids:
                attrs["penwidth"] = "3"
                attrs["color"] = "#1f77b4"
            
            # Pending evaluation marker
            if n.accepted is None:
                attrs["style"] = "filled,dashed"
                attrs["color"] = "#ff9800"
            
            attr_str = ",".join(f"{k}=\"{v}\"" for k, v in attrs.items())
            lines.append(f"  \"{n.id}\" [{attr_str}];")
        
        # Edge lines with dependency information
        for nid, n in self._nodes.items():
            if n.parent_id:
                lines.append(f"  \"{n.parent_id}\" -> \"{nid}\";")
            
            # Add dependency edges (dashed)
            for dep_id in n.dependencies:
                if dep_id in self._nodes:
                    lines.append(f"  \"{dep_id}\" -> \"{nid}\" [style=dashed, color=gray];")
        
        # Add legend
        lines.extend([
            "",
            "  // Legend",
            "  subgraph cluster_legend {",
            "    label=\"Legend\";",
            "    style=filled;",
            "    fillcolor=lightgray;",
            "    legend_accepted [label=\"Accepted\", fillcolor=\"#d5f5e3\", shape=box];",
            "    legend_rejected [label=\"Rejected\", fillcolor=\"#f5b7b1\", shape=box];",
            "    legend_pending [label=\"Pending\", fillcolor=white, style=\"filled,dashed\", color=\"#ff9800\", shape=box];",
            "    legend_dependency [label=\"Dependency\", shape=point];",
            "    legend_dependency -> legend_accepted [style=dashed, color=gray, label=\"depends on\"];",
            "  }",
        ])
        
        lines.append("}")
        return "\n".join(lines)

    # ---- Enhanced pruning / subgraph ----
    def keep_topk_children(self, parent_id: str, k: int, by: str = "score") -> None:
        """Keep only top-k children of a parent by given attribute (default score)."""
        ch = list(self._children.get(parent_id, []))
        if not ch:
            return
        
        def key_fn(nid: str) -> Any:
            n = self._nodes[nid]
            if by == "impact_score":
                return n.impact_score if n.impact_score is not None else float('-inf')
            elif by == "score":
                return n.score if n.score is not None else float('-inf')
            elif by == "evaluation_time":
                return -(n.evaluation_time_ms or float('inf'))  # Faster is better
            else:
                return getattr(n, by, None) if hasattr(n, by) else float('-inf')
        
        ch_sorted = sorted(ch, key=key_fn, reverse=True)
        to_remove = set(ch_sorted[k:])
        
        # Remove recursively
        for nid in list(to_remove):
            self._remove_subtree(nid)
        
        # Update child list
        self._children[parent_id] = [nid for nid in ch if nid not in to_remove]

    def _remove_subtree(self, node_id: str) -> None:
        """Enhanced removal that cleans up all tracking data."""
        # Remove children recursively
        for c in list(self._children.get(node_id, [])):
            self._remove_subtree(c)
        
        # Clean up tracking data
        node = self._nodes.get(node_id)
        if node:
            # Remove from sessions
            for session in self._sessions.values():
                if node_id in session.modification_ids:
                    session.modification_ids.remove(node_id)
            
            # Remove from timeline (keep for historical accuracy)
            # Timeline entries are kept for audit trail
        
        # Unlink from parent
        pid = self._nodes[node_id].parent_id if node_id in self._nodes else None
        if pid is None:
            self._roots.discard(node_id)
        else:
            if pid in self._children:
                self._children[pid] = [x for x in self._children[pid] if x != node_id]
        
        self._children.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def subgraph(self, root_id: str, depth: Optional[int] = None) -> "ModTree":
        """Enhanced subgraph extraction with session preservation."""
        if root_id not in self._nodes:
            raise KeyError(root_id)
        
        out = ModTree()
        copied_nodes = set()
        
        def walk(nid: str, d: int) -> None:
            n = self._nodes[nid]
            out.add(
                op=n.op,
                params=n.params,
                parent_id=None if nid == root_id else n.parent_id,
                score=n.score,
                accepted=n.accepted,
                step=n.step,
                tags=n.tags,
                scope=n.scope,
                node_id=n.id,
                component_id=n.component_id,
                modification_type=n.modification_type,
                dependencies=n.dependencies,
                rollback_info=n.rollback_info,
            )
            copied_nodes.add(nid)
            
            if depth is not None and d >= depth:
                return
            for c in self._children.get(nid, []):
                walk(c, d + 1)
        
        walk(root_id, 0)
        
        # Copy relevant sessions
        for session in self._sessions.values():
            relevant_modifications = [mid for mid in session.modification_ids if mid in copied_nodes]
            if relevant_modifications:
                new_session = ModificationSession(
                    session_id=session.session_id,
                    name=session.name,
                    start_time=session.start_time,
                    end_time=session.end_time,
                    step=session.step,
                    modification_ids=relevant_modifications,
                    session_tags=session.session_tags,
                    target_component=session.target_component,
                    session_type=session.session_type,
                )
                out._sessions[session.session_id] = new_session
        
        return out

    # ---- Enhanced I/O ----
    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")

    def save_dot(self, path: str, **kwargs: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_dot(**kwargs), encoding="utf-8")

    def save_mlops_export(self, base_path: str, step: Optional[int] = None) -> None:
        """Save all files needed for MLOps tracking."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        export_data = self.export_for_mlops(step)
        
        # Save main export
        (base_path / "export_summary.json").write_text(
            json.dumps(export_data, indent=2, default=str), encoding="utf-8"
        )
        
        # Save individual artifacts
        for artifact_path, content in export_data["artifact_files"].items():
            file_path = base_path / artifact_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if artifact_path.endswith('.json'):
                if isinstance(content, dict):
                    file_path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
                else:
                    file_path.write_text(str(content), encoding="utf-8")
            else:
                file_path.write_text(str(content), encoding="utf-8")


class ModificationSessionManager:
    """Helper class for managing modifications within a session."""
    
    def __init__(self, tree: ModTree, session_id: str):
        self.tree = tree
        self.session_id = session_id
        self.session = tree._sessions[session_id]
    
    def add_modification(self, op: str, params: Optional[Dict[str, Any]] = None,
                        parent_id: Optional[str] = None, **kwargs) -> ModNode:
        """Add a modification to this session."""
        return self.tree.add(op=op, params=params, parent_id=parent_id, 
                           session_id=self.session_id, **kwargs)
    
    def set_acceptance(self, node_id: str, accepted: bool, **kwargs) -> None:
        """Set acceptance for a modification in this session."""
        self.tree.set_acceptance(node_id, accepted, **kwargs)
    
    def get_session_modifications(self) -> List[ModNode]:
        """Get all modifications in this session."""
        return [self.tree._nodes[nid] for nid in self.session.modification_ids 
                if nid in self.tree._nodes]
    
    def get_pending_evaluations(self) -> List[ModNode]:
        """Get pending evaluations in this session."""
        session_mods = self.get_session_modifications()
        return [node for node in session_mods if node.accepted is None]
    
    def mark_evaluation_start(self, node_id: str) -> None:
        """Mark the start of evaluation for a modification."""
        if node_id in self.tree._nodes:
            self.tree._nodes[node_id].set_evaluation_start()


class ModificationTracker:
    """High-level interface for tracking modifications with multiple trees."""
    
    def __init__(self):
        self.trees: Dict[str, ModTree] = {}
        self.current_tree_id: Optional[str] = None
    
    def create_tree(self, tree_id: str) -> ModTree:
        """Create a new modification tree."""
        if tree_id in self.trees:
            raise ValueError(f"Tree {tree_id} already exists")
        
        tree = ModTree()
        self.trees[tree_id] = tree
        
        if self.current_tree_id is None:
            self.current_tree_id = tree_id
        
        return tree
    
    def get_tree(self, tree_id: Optional[str] = None) -> ModTree:
        """Get a modification tree (current if none specified)."""
        if tree_id is None:
            tree_id = self.current_tree_id
        
        if tree_id is None or tree_id not in self.trees:
            raise ValueError(f"Tree {tree_id} not found")
        
        return self.trees[tree_id]
    
    def start_session(self, name: str, tree_id: Optional[str] = None, **kwargs):
        """Start a modification session on a specific tree."""
        tree = self.get_tree(tree_id)
        return tree.start_session(name, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all trees."""
        all_stats = {}
        for tree_id, tree in self.trees.items():
            all_stats[tree_id] = tree.stats()
        
        # Aggregate statistics
        total_modifications = sum(stats["node_count"] for stats in all_stats.values())
        avg_acceptance_rate = sum(
            stats["acceptance_rate"] for stats in all_stats.values() 
            if stats["acceptance_rate"] is not None
        ) / max(len([s for s in all_stats.values() if s["acceptance_rate"] is not None]), 1)
        
        return {
            "trees": all_stats,
            "aggregate": {
                "total_trees": len(self.trees),
                "total_modifications": total_modifications,
                "avg_acceptance_rate": avg_acceptance_rate,
            }
        }
    
    def export_for_mlops(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Export all trees for MLOps tracking."""
        exports = {}
        for tree_id, tree in self.trees.items():
            exports[tree_id] = tree.export_for_mlops(step)
        
        return {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "trees": exports,
            "summary": self.get_comprehensive_stats()
        }


# -----------------------------
# Enhanced utility functions
# -----------------------------

def build_tree_and_stats(modifications: List[Dict[str, Any]], 
                         enable_sessions: bool = True) -> Tuple[ModTree, Dict[str, Any]]:
    """Enhanced convenience function for tracker with session support."""
    tree = ModTree()
    
    if enable_sessions:
        # Group modifications by step/timestamp for automatic sessions
        step_groups = defaultdict(list)
        for mod in modifications:
            step = mod.get("step", 0)
            step_groups[step].append(mod)
        
        # Create sessions for each step
        for step, step_mods in step_groups.items():
            with tree.start_session(f"step_{step}", step=step) as session:
                for mod in step_mods:
                    session.add_modification(
                        op=mod.get("op", "unknown"),
                        params=mod.get("params"),
                        parent_id=mod.get("parent_id"),
                        score=mod.get("score"),
                        accepted=mod.get("accepted"),
                        tags=mod.get("tags"),
                        scope=mod.get("scope"),
                        node_id=mod.get("id"),
                        component_id=mod.get("component_id"),
                        modification_type=mod.get("modification_type"),
                        dependencies=mod.get("dependencies"),
                        rollback_info=mod.get("rollback_info"),
                    )
    else:
        tree.bulk_add(modifications)
    
    return tree, tree.stats()

def create_modification_report(tree: ModTree, include_visualization: bool = True) -> Dict[str, Any]:
    """Create a comprehensive modification report."""
    stats = tree.stats()
    acceptance_analysis = tree.get_acceptance_analysis()
    real_time_stats = tree.get_real_time_stats()
    
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "2.0",
        },
        "executive_summary": {
            "total_modifications": stats["node_count"],
            "acceptance_rate": stats["acceptance_rate"] or 0.0,
            "pending_evaluations": len(tree.get_pending_evaluations()),
            "most_common_operation": max(stats["op_hist"].items(), key=lambda x: x[1])[0] if stats["op_hist"] else "none",
            "most_productive_session": None,  # TODO: Calculate based on session stats
        },
        "detailed_statistics": stats,
        "acceptance_analysis": acceptance_analysis.__dict__,
        "real_time_metrics": real_time_stats,
        "recommendations": generate_modification_recommendations(tree),
    }
    
    if include_visualization:
        report["visualization"] = {
            "dot_graph": tree.to_dot(color_by="accepted"),
            "acceptance_dot_graph": tree.to_dot(color_by="impact"),
        }
    
    return report

def generate_modification_recommendations(tree: ModTree) -> List[str]:
    """Generate actionable recommendations based on modification patterns."""
    recommendations = []
    
    stats = tree.stats()
    acceptance_rate = stats.get("acceptance_rate", 0)
    
    if acceptance_rate is not None:
        if acceptance_rate < 0.3:
            recommendations.append("Low acceptance rate detected. Consider refining modification criteria or evaluation metrics.")
        elif acceptance_rate > 0.8:
            recommendations.append("High acceptance rate suggests modifications might be too conservative. Consider more aggressive changes.")
    
    # Analyze pending evaluations
    pending = tree.get_pending_evaluations()
    if len(pending) > 10:
        recommendations.append(f"{len(pending)} pending evaluations. Consider increasing evaluation capacity or implementing automatic evaluation.")
    
    # Analyze modification types
    acceptance_analysis = tree.get_acceptance_analysis()
    for mod_type, type_stats in acceptance_analysis.acceptance_by_type.items():
        type_acceptance = type_stats["accepted"] / max(type_stats["accepted"] + type_stats["rejected"], 1)
        if type_acceptance < 0.2:
            recommendations.append(f"Low acceptance rate for {mod_type} modifications ({type_acceptance:.1%}). Consider different approaches.")
    
    # Analyze evaluation time
    if acceptance_analysis.avg_evaluation_time_ms > 1000:
        recommendations.append(f"High evaluation time ({acceptance_analysis.avg_evaluation_time_ms:.1f}ms avg). Consider optimizing evaluation process.")
    
    if not recommendations:
        recommendations.append("Modification patterns look healthy. Continue current approach.")
    
    return recommendations