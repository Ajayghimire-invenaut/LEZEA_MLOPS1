"""
Business Cost Model for LeZeA MLOps
===================================

Purpose
-------
Estimate and attribute **euro cost** to training runs from sampled resource usage.
Covers spec 1.7.1 (price of resources) and provides clean per-scope breakdowns.

Design
------
- No third‑party deps. Pure Python.
- Flexible: you can update the model with partial samples (GPU only, CPU only, …).
- Scope-aware: costs can be tracked per **builder/tasker/algorithm/network/layer**.
- Configurable rates via env vars or constructor.

Units & Assumptions
-------------------
- Effective GPU hours: `gpu_util(%) / 100 * gpu_count * dt_seconds / 3600`.
- Effective CPU core hours: `cpu_percent(%) / 100 * cpu_cores * dt_seconds / 3600`.
- RAM GB-hours: `ram_gb * dt_seconds / 3600`.
- IO GB: `(io_read_bytes + io_write_bytes) / 1e9` (decimal GB).
- Egress GB: explicit; not derived from io bytes (often priced differently).
- Storage GB-months: add explicitly (e.g., checkpoints in object storage).

Environment Rates (defaults in brackets)
----------------------------------------
- `GPU_RATE_EUR_PER_HOUR` [2.00]
- `CPU_RATE_EUR_PER_CORE_HOUR` [0.05]
- `RAM_RATE_EUR_PER_GB_HOUR` [0.004]
- `IO_RATE_EUR_PER_GB` [0.01]
- `EGRESS_RATE_EUR_PER_GB` [0.06]
- `STORAGE_RATE_EUR_PER_GB_MONTH` [0.02]
- `GPU_MEM_RATE_EUR_PER_GB_HOUR` [0.00]  # optional, usually 0 (included in GPU price)

Integration snippet (tracker)
----------------------------
>>> from lezea_mlops.business.cost_model import CostModel
>>> cost = CostModel.from_env()
>>> # at each step: sample usage and update with dt
>>> cost.update(scope_key="tasker:T1", dt_seconds=1.0, gpu_util=65, gpu_count=1,
...             cpu_percent=120, cpu_cores=8, ram_gb=6.5, io_read_bytes=2e6, io_write_bytes=5e6)
>>> # at the end
>>> summary = cost.summary()
>>> total_eur = summary["total_eur"]
>>> breakdown_json = cost.to_json()

You can then log `breakdown_json` to MLflow & Mongo.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os

# ------------------
# Rate configuration
# ------------------

@dataclass
class CostRates:
    gpu_hour_eur: float = 2.00
    cpu_core_hour_eur: float = 0.05
    ram_gb_hour_eur: float = 0.004
    io_gb_eur: float = 0.01
    egress_gb_eur: float = 0.06
    storage_gb_month_eur: float = 0.02
    gpu_mem_gb_hour_eur: float = 0.00  # usually included in GPU price

    @classmethod
    def from_env(cls) -> "CostRates":
        def f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except Exception:
                return default
        return cls(
            gpu_hour_eur=f("GPU_RATE_EUR_PER_HOUR", 2.00),
            cpu_core_hour_eur=f("CPU_RATE_EUR_PER_CORE_HOUR", 0.05),
            ram_gb_hour_eur=f("RAM_RATE_EUR_PER_GB_HOUR", 0.004),
            io_gb_eur=f("IO_RATE_EUR_PER_GB", 0.01),
            egress_gb_eur=f("EGRESS_RATE_EUR_PER_GB", 0.06),
            storage_gb_month_eur=f("STORAGE_RATE_EUR_PER_GB_MONTH", 0.02),
            gpu_mem_gb_hour_eur=f("GPU_MEM_RATE_EUR_PER_GB_HOUR", 0.00),
        )

# -----------------
# Running totals
# -----------------

@dataclass
class Totals:
    gpu_hours: float = 0.0
    cpu_core_hours: float = 0.0
    ram_gb_hours: float = 0.0
    io_gb: float = 0.0
    egress_gb: float = 0.0
    storage_gb_months: float = 0.0
    gpu_mem_gb_hours: float = 0.0
    fixed_eur: float = 0.0  # one-off costs

    def add(self, other: "Totals") -> None:
        self.gpu_hours += other.gpu_hours
        self.cpu_core_hours += other.cpu_core_hours
        self.ram_gb_hours += other.ram_gb_hours
        self.io_gb += other.io_gb
        self.egress_gb += other.egress_gb
        self.storage_gb_months += other.storage_gb_months
        self.gpu_mem_gb_hours += other.gpu_mem_gb_hours
        self.fixed_eur += other.fixed_eur

# -----------------
# Main model
# -----------------

class CostModel:
    def __init__(self, rates: Optional[CostRates] = None) -> None:
        self.rates = rates or CostRates()
        # Per-scope totals: key -> Totals
        self._totals: Dict[str, Totals] = {}
        # Timestamps
        self.created_at = datetime.now().isoformat()

    # -------------
    # Update API
    # -------------
    def update(
        self,
        *,
        scope_key: str = "global",
        dt_seconds: float,
        gpu_util: Optional[float] = None,
        gpu_count: int = 1,
        cpu_percent: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        ram_gb: Optional[float] = None,
        io_read_bytes: float = 0.0,
        io_write_bytes: float = 0.0,
        egress_gb: float = 0.0,
        gpu_mem_mb: Optional[float] = None,
    ) -> None:
        """Update totals using a single time slice of usage.

        Provide what you have; unspecified values are ignored.
        - gpu_util: 0..100 (percentage per-GPU average). If you have multiple GPUs, pass gpu_count.
        - cpu_percent: 0..(100*cores) if measured as total across cores, else 0..100 with cpu_cores set.
        - cpu_cores: physical+logical cores available (for better normalization). If None and cpu_percent>100, we infer cores = round(cpu_percent/100).
        - gpu_mem_mb: optional, charges against gpu_mem_gb_hour_eur if non-zero.
        """
        if dt_seconds <= 0:
            return
        t = self._totals.setdefault(scope_key, Totals())

        # GPU effective hours
        if gpu_util is not None and gpu_util > 0 and gpu_count > 0:
            eff_gpu_hours = max(0.0, float(gpu_util)) / 100.0 * float(gpu_count) * (dt_seconds / 3600.0)
            t.gpu_hours += eff_gpu_hours
        # GPU memory hours (optional)
        if gpu_mem_mb is not None and gpu_mem_mb > 0:
            t.gpu_mem_gb_hours += (float(gpu_mem_mb) / 1024.0) * (dt_seconds / 3600.0)

        # CPU effective core hours
        if cpu_percent is not None and cpu_percent > 0:
            # If cpu_percent is aggregate across cores (typical), allow values > 100
            cores = cpu_cores or (int(round(cpu_percent / 100.0)) if cpu_percent > 100 else 1)
            # Normalize percent to 0..100 for a single core utilization average
            # If aggregate: avg per core = cpu_percent / cores
            per_core_util = float(cpu_percent) / float(cores)
            eff_core_hours = max(0.0, per_core_util) / 100.0 * float(cores) * (dt_seconds / 3600.0)
            t.cpu_core_hours += eff_core_hours

        # RAM GB-hours
        if ram_gb is not None and ram_gb > 0:
            t.ram_gb_hours += float(ram_gb) * (dt_seconds / 3600.0)

        # IO GB (decimal)
        total_bytes = float(io_read_bytes) + float(io_write_bytes)
        if total_bytes > 0:
            t.io_gb += total_bytes / 1e9

        # Egress GB (explicit)
        if egress_gb and egress_gb > 0:
            t.egress_gb += float(egress_gb)

    # Explicit adds for storage and fixed costs
    def add_storage_gb_months(self, gb_months: float, scope_key: str = "global") -> None:
        t = self._totals.setdefault(scope_key, Totals())
        t.storage_gb_months += max(0.0, float(gb_months))

    def add_fixed_cost(self, eur: float, scope_key: str = "global") -> None:
        t = self._totals.setdefault(scope_key, Totals())
        t.fixed_eur += max(0.0, float(eur))

    # -------------
    # Math
    # -------------
    def _cost_components(self, totals: Totals) -> Dict[str, float]:
        r = self.rates
        return {
            "gpu_eur": totals.gpu_hours * r.gpu_hour_eur,
            "cpu_eur": totals.cpu_core_hours * r.cpu_core_hour_eur,
            "ram_eur": totals.ram_gb_hours * r.ram_gb_hour_eur,
            "io_eur": totals.io_gb * r.io_gb_eur,
            "egress_eur": totals.egress_gb * r.egress_gb_eur,
            "storage_eur": totals.storage_gb_months * r.storage_gb_month_eur,
            "gpu_mem_eur": totals.gpu_mem_gb_hours * r.gpu_mem_gb_hour_eur,
            "fixed_eur": totals.fixed_eur,
        }

    def _sum_components(self, comp: Dict[str, float]) -> float:
        return sum(float(v) for v in comp.values())

    # -------------
    # Reports
    # -------------
    def summary(self) -> Dict[str, Any]:
        per_scope: Dict[str, Any] = {}
        total = Totals()
        for sk, t in self._totals.items():
            comp = self._cost_components(t)
            per_scope[sk] = {
                "totals": asdict(t),
                "components_eur": comp,
                "total_eur": self._sum_components(comp),
            }
            total.add(t)
        total_comp = self._cost_components(total)
        out = {
            "created_at": self.created_at,
            "rates_eur": asdict(self.rates),
            "scopes": per_scope,
            "grand_totals": asdict(total),
            "grand_components_eur": total_comp,
            "total_eur": self._sum_components(total_comp),
        }
        return out

    def to_json(self) -> Dict[str, Any]:
        return self.summary()

    # -------------
    # Convenience
    # -------------
    @classmethod
    def from_env(cls) -> "CostModel":
        return cls(CostRates.from_env())

    def reset(self) -> None:
        self._totals.clear()
        self.created_at = datetime.now().isoformat()

# -----------------
# Minimal smoke test (optional usage example)
# -----------------
if __name__ == "__main__":
    cm = CostModel.from_env()
    # simulate 10 seconds at given usage
    for _ in range(10):
        cm.update(scope_key="global", dt_seconds=1.0, gpu_util=70, gpu_count=1, cpu_percent=200, cpu_cores=8, ram_gb=8.0, io_read_bytes=1e6)
    cm.add_storage_gb_months(5.0)
    cm.add_fixed_cost(1.99)
    print(json.dumps(cm.summary(), indent=2))
