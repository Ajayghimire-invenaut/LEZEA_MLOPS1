"""
Business Cost Model for LeZeA MLOps System

This module provides a business-oriented cost estimation model for MLOps workflows,
focusing on resource usage attribution and euro cost breakdowns. It enables
tracking and attribution of costs to specific scopes in machine learning training runs.

Features:
    - Scope-aware cost tracking for components like builder/tasker/algorithm
    - Flexible updates with partial resource usage samples
    - Configurable pricing rates via environment variables
    - Detailed cost component breakdowns
    - JSON export for logging and integration
    - No third-party dependencies (pure Python)
    - Support for GPU, CPU, RAM, IO, egress, storage, and fixed costs

Author: [Your Name/Team]
Date: 2025-08-29
Version: 1.0.0
License: [Your License]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os


class CostRates:
    """
    Configuration for resource pricing rates in euros.
    
    This dataclass holds the pricing rates for various compute resources.
    Rates can be customized via environment variables or direct instantiation.
    
    Attributes:
        gpu_hour_eur: Cost per GPU hour
        cpu_core_hour_eur: Cost per CPU core hour
        ram_gb_hour_eur: Cost per GB RAM hour
        io_gb_eur: Cost per GB IO (read/write)
        egress_gb_eur: Cost per GB data egress
        storage_gb_month_eur: Cost per GB-month storage
        gpu_mem_gb_hour_eur: Cost per GB GPU memory hour (usually 0)
    """

    gpu_hour_eur: float = 2.00
    cpu_core_hour_eur: float = 0.05
    ram_gb_hour_eur: float = 0.004
    io_gb_eur: float = 0.01
    egress_gb_eur: float = 0.06
    storage_gb_month_eur: float = 0.02
    gpu_mem_gb_hour_eur: float = 0.00  # usually included in GPU price

    @classmethod
    def from_env(cls) -> "CostRates":
        """
        Create CostRates instance from environment variables.
        
        Loads rates from env vars with fallbacks to default values.
        Environment variable names match attribute names in uppercase with suffixes.
        
        Returns:
            CostRates instance with loaded values
        """
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


@dataclass
class Totals:
    """
    Accumulator for resource usage totals.
    
    Tracks cumulative resource consumption metrics that can be used
    to compute costs based on rates.
    
    Attributes:
        gpu_hours: Cumulative GPU hours
        cpu_core_hours: Cumulative CPU core hours
        ram_gb_hours: Cumulative RAM GB-hours
        io_gb: Cumulative IO in GB
        egress_gb: Cumulative egress in GB
        storage_gb_months: Cumulative storage GB-months
        gpu_mem_gb_hours: Cumulative GPU memory GB-hours
        fixed_eur: Fixed one-off costs in euros
    """
    gpu_hours: float = 0.0
    cpu_core_hours: float = 0.0
    ram_gb_hours: float = 0.0
    io_gb: float = 0.0
    egress_gb: float = 0.0
    storage_gb_months: float = 0.0
    gpu_mem_gb_hours: float = 0.0
    fixed_eur: float = 0.0  # one-off costs

    def add(self, other: "Totals") -> None:
        """
        Add another Totals instance to this one in-place.
        
        Args:
            other: Another Totals instance to add
        """
        self.gpu_hours += other.gpu_hours
        self.cpu_core_hours += other.cpu_core_hours
        self.ram_gb_hours += other.ram_gb_hours
        self.io_gb += other.io_gb
        self.egress_gb += other.egress_gb
        self.storage_gb_months += other.storage_gb_months
        self.gpu_mem_gb_hours += other.gpu_mem_gb_hours
        self.fixed_eur += other.fixed_eur


class CostModel:
    """
    Business cost model for estimating MLOps resource expenses.
    
    This class tracks resource usage across different scopes and computes
    euro costs based on configurable rates. It supports incremental updates
    from usage samples and provides detailed breakdowns.
    
    Key Features:
        - Incremental updates from resource usage samples
        - Scope-based tracking (e.g., per component or experiment)
        - Automatic cost calculations for GPU, CPU, memory, etc.
        - JSON export for logging/integration
        - Environment variable configuration support
        
    Attributes:
        rates: CostRates instance for pricing
        _totals: Internal dictionary of scope to Totals
        created_at: ISO timestamp of model creation
    """

    def __init__(self, rates: Optional[CostRates] = None) -> None:
        """
        Initialize the cost model.
        
        Args:
            rates: Optional custom CostRates, defaults to default rates
        """
        self.rates = rates or CostRates()
        # Per-scope totals: key -> Totals
        self._totals: Dict[str, Totals] = {}
        # Timestamps
        self.created_at = datetime.now().isoformat()

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
        """
        Update totals using a single time slice of resource usage.
        
        Provide available metrics; unspecified values are ignored.
        Effective hours are calculated based on utilization and time delta.
        
        Args:
            scope_key: Scope identifier (e.g., "tasker:T1")
            dt_seconds: Time delta in seconds for this sample
            gpu_util: GPU utilization percentage (0-100)
            gpu_count: Number of GPUs
            cpu_percent: CPU utilization percentage (can exceed 100 if aggregate)
            cpu_cores: Number of CPU cores (inferred if None and cpu_percent >100)
            ram_gb: RAM usage in GB
            io_read_bytes: IO read bytes
            io_write_bytes: IO write bytes
            egress_gb: Data egress in GB
            gpu_mem_mb: GPU memory usage in MB
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

    def add_storage_gb_months(self, gb_months: float, scope_key: str = "global") -> None:
        """
        Add storage usage in GB-months to a scope.
        
        Args:
            gb_months: Storage usage to add
            scope_key: Scope identifier
        """
        t = self._totals.setdefault(scope_key, Totals())
        t.storage_gb_months += max(0.0, float(gb_months))

    def add_fixed_cost(self, eur: float, scope_key: str = "global") -> None:
        """
        Add fixed one-off cost in euros to a scope.
        
        Args:
            eur: Fixed cost to add
            scope_key: Scope identifier
        """
        t = self._totals.setdefault(scope_key, Totals())
        t.fixed_eur += max(0.0, float(eur))

    def _cost_components(self, totals: Totals) -> Dict[str, float]:
        """
        Compute cost components for given totals.
        
        Args:
            totals: Totals instance to compute costs for
            
        Returns:
            Dictionary of cost components in euros
        """
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
        """
        Sum cost components to get total.
        
        Args:
            comp: Dictionary of cost components
            
        Returns:
            Total cost in euros
        """
        return sum(float(v) for v in comp.values())

    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost summary.
        
        Returns:
            Dictionary with rates, per-scope breakdowns, and grand totals
        """
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
        """
        Export summary as JSON-compatible dictionary.
        
        Returns:
            Same as summary() method
        """
        return self.summary()

    @classmethod
    def from_env(cls) -> "CostModel":
        """
        Create CostModel using rates from environment variables.
        
        Returns:
            Initialized CostModel instance
        """
        return cls(CostRates.from_env())

    def reset(self) -> None:
        """
        Reset all accumulated totals and update creation timestamp.
        """
        self._totals.clear()
        self.created_at = datetime.now().isoformat()

# Minimal smoke test (optional usage example)
if __name__ == "__main__":
    cm = CostModel.from_env()
    # simulate 10 seconds at given usage
    for _ in range(10):
        cm.update(scope_key="global", dt_seconds=1.0, gpu_util=70, gpu_count=1, cpu_percent=200, cpu_cores=8, ram_gb=8.0, io_read_bytes=1e6)
    cm.add_storage_gb_months(5.0)
    cm.add_fixed_cost(1.99)
    print(json.dumps(cm.summary(), indent=2))