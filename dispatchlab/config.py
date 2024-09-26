"""Runtime configuration, read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 7
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    noise: float = 1.0  # multiplier on the lognormal noise of pick/drive times
    n_stores: int = 5
    n_shoppers: int = 28
    orders_per_day: int = 340
    history_days: int = 40  # days of historical logs used to train the time model
    eval_days: int = 21  # simulated days per policy in the policy comparison
    ope_log_days: int = 14  # days per logged randomized-policy dataset
    ope_replicates: int = 8  # independent logged datasets (for CI coverage)
    ope_truth_days: int = 28  # days simulated per target policy for ground truth
    sla_min: float = 60.0  # promised delivery window (minutes after order placement)
    hold_min: float = 6.0  # max time an order is held waiting for batch partners
    wage_per_min: float = 0.30  # shopper pay, $/minute on shift (idle time is paid)
    fuel_per_km: float = 0.25  # $/km
    bootstrap: int = 400

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls()
        vals = {}
        for f in fields(cls):
            raw = os.environ.get("DISPATCHLAB_" + f.name.upper())
            if raw is None:
                continue
            kind = type(getattr(cfg, f.name))
            vals[f.name] = Path(raw) if kind is Path else kind(raw)
        vals.update(overrides)
        return replace(cfg, **vals)
