"""
Modification Trees for LeZeA
=============================

Lightweight utilities to build, update, analyze, and export **modification trees**
(evolution of architectures/hyperparams/patches over time) with **no third‑party
dependencies**.

It covers your spec items:
- 1.5.2  Modification trees and paths
- 1.5.3  Modification statistics

## Concepts
- **Node**: a single modification (e.g., mutate layer width, swap activation).
- **Edge**: parent→child relation showing lineage of changes.
- **Scope**: optional tag identifying builder/tasker/algorithm/network/layer.

## JSON format (stable)
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
      "scope": {"level": "tasker", "entity_id": "T1"}
    },
    ...
  ]
}

## Example
>>> tree = ModTree()
>>> n1 = tree.add(op="init", params={"net": "A"}, accepted=True, step=0)
>>> n2 = tree.add(op="set_lr", params={"lr": 1e-3}, parent_id=n1.id, score=0.7, accepted=True, step=1)
>>> n3 = tree.add(op="mutate_width", params={"delta": +32}, parent_id=n2.id, score=0.6, step=2)
>>> stats = tree.stats()
>>> dot = tree.to_dot(highlight_ids={n2.id})

You can log the JSON and DOT strings via the tracker:
tracker.backends["mlflow"].log_dict(tree.to_json(), f"modifications/step_{step}.json")
tracker.backends["mlflow"].log_text(tree.to_dot(),   f"modifications/step_{step}.dot")

"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import json
import uuid

# -----------------------------
# Data classes
# -----------------------------

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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

# -----------------------------
# Tree structure
# -----------------------------

class ModTree:
    def __init__(self) -> None:
        self._nodes: Dict[str, ModNode] = {}
        self._children: Dict[str, List[str]] = {}
        self._roots: Set[str] = set()

    # ---- mutation API ----
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
    ) -> ModNode:
        """Create and add a node. Returns the created node.

        - If parent_id is None, node becomes a root.
        - node_id, if provided, must be unique.
        """
        nid = node_id or str(uuid.uuid4())
        if nid in self._nodes:
            raise ValueError(f"Node id already exists: {nid}")
        if parent_id is not None and parent_id not in self._nodes:
            # Auto-create a stub parent? Prefer explicit error to avoid silent corruption.
            raise KeyError(f"Parent id not found: {parent_id}")

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
        )
        self._nodes[nid] = node
        if parent_id is None:
            self._roots.add(nid)
        else:
            self._children.setdefault(parent_id, []).append(nid)
        return node

    def bulk_add(self, nodes: Iterable[Dict[str, Any]]) -> None:
        """Add multiple nodes using the JSON schema described in the module docstring.
        Parents must appear before children in the iterable (topological order).
        """
        for n in nodes:
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
            )

    # ---- queries ----
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

    # ---- analysis & stats ----
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

    def stats(self) -> Dict[str, Any]:
        return {
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

    # ---- pruning / subgraph ----
    def keep_topk_children(self, parent_id: str, k: int, by: str = "score") -> None:
        """Keep only top-k children of a parent by given attribute (default score)."""
        ch = list(self._children.get(parent_id, []))
        if not ch:
            return
        def key_fn(nid: str) -> Any:
            n = self._nodes[nid]
            return getattr(n, by, None) if hasattr(n, by) else (n.score if by == "score" else None)
        ch_sorted = sorted(ch, key=key_fn, reverse=True)
        to_remove = set(ch_sorted[k:])
        # remove recursively
        for nid in list(to_remove):
            self._remove_subtree(nid)
        # update child list
        self._children[parent_id] = [nid for nid in ch if nid not in to_remove]

    def _remove_subtree(self, node_id: str) -> None:
        for c in list(self._children.get(node_id, [])):
            self._remove_subtree(c)
        # unlink from parent
        pid = self._nodes[node_id].parent_id
        if pid is None:
            self._roots.discard(node_id)
        else:
            if pid in self._children:
                self._children[pid] = [x for x in self._children[pid] if x != node_id]
        self._children.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def subgraph(self, root_id: str, depth: Optional[int] = None) -> "ModTree":
        """Copy a subtree starting at root_id (up to depth if provided)."""
        if root_id not in self._nodes:
            raise KeyError(root_id)
        out = ModTree()
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
            )
            if depth is not None and d >= depth:
                return
            for c in self._children.get(nid, []):
                walk(c, d + 1)
        walk(root_id, 0)
        return out

    # ---- export ----
    def to_json(self) -> Dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self._nodes.values()], "stats": self.stats()}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ModTree":
        t = cls()
        nodes = data.get("nodes", [])
        # Insert in parent-first order if possible
        # Simple heuristic: roots first, then by presence of parent
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
            )
        return t

    def to_dot(self, highlight_ids: Optional[Set[str]] = None, color_by: str = "accepted") -> str:
        """Graphviz DOT representation for quick visualization.

        - color_by = "accepted" | "op"
        """
        highlight_ids = highlight_ids or set()
        lines = ["digraph mod_tree {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=white];"]
        # Node lines
        for n in self._nodes.values():
            label = f"{n.op}"\
                    f"\nscore={n.score if n.score is not None else '-'}"\
                    f"\nstep={n.step if n.step is not None else '-'}"
            attrs = {"label": label}
            if color_by == "accepted" and n.accepted is not None:
                attrs["fillcolor"] = "#d5f5e3" if n.accepted else "#f5b7b1"
            elif color_by == "op":
                # simple deterministic color hash from op name
                hue = (hash(n.op) % 360)
                attrs["fillcolor"] = f"/pastel13/{1 + (hue % 3)}"  # works in Graphviz schemes
            if n.id in highlight_ids:
                attrs["penwidth"] = "3"
                attrs["color"] = "#1f77b4"
            attr_str = ",".join(f"{k}=\"{v}\"" for k, v in attrs.items())
            lines.append(f"  \"{n.id}\" [{attr_str}];")
        # Edge lines
        for nid, n in self._nodes.items():
            if n.parent_id:
                lines.append(f"  \"{n.parent_id}\" -> \"{nid}\";")
        lines.append("}")
        return "\n".join(lines)

    # ---- convenience I/O ----
    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def save_dot(self, path: str, **kwargs: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_dot(**kwargs), encoding="utf-8")


# -----------------------------
# Utility: build tree from a flat list (parent-first) and compute stats only
# -----------------------------

def build_tree_and_stats(modifications: List[Dict[str, Any]]) -> Tuple[ModTree, Dict[str, Any]]:
    """Convenience for tracker: create a tree, compute stats, and return both."""
    tree = ModTree()
    tree.bulk_add(modifications)
    return tree, tree.stats()
