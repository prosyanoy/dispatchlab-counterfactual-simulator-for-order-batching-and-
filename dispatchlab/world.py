"""Synthetic world: stores, shoppers, and daily demand."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import truth
from .config import Config

GRID_KM = 12.0
VEHICLE_KMH = np.array([20.0, 26.0, 32.0])


@dataclass
class World:
    stores: pd.DataFrame  # store_id,x,y,store_size_sqft,[store_hist_min_per_item] + hidden layout,overhead_min
    shoppers: pd.DataFrame  # shopper_id,shopper_rating,shopper_tenure_days,shopper_vehicle_kmh + hidden skill,speed_factor

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.stores.to_parquet(data_dir / "stores.parquet", index=False)
        self.shoppers.to_parquet(data_dir / "shoppers.parquet", index=False)

    @classmethod
    def load(cls, data_dir: Path) -> "World":
        stores = pd.read_parquet(data_dir / "stores.parquet")
        stats = data_dir / "store_stats.parquet"
        if stats.exists():
            stores = stores.drop(columns=["store_hist_min_per_item"], errors="ignore").merge(
                pd.read_parquet(stats), on="store_id", how="left"
            )
        return cls(stores, pd.read_parquet(data_dir / "shoppers.parquet"))


def make_world(cfg: Config) -> World:
    rng = np.random.default_rng([cfg.seed, 1])
    n = cfg.n_stores
    size = rng.uniform(15_000, 60_000, n)
    stores = pd.DataFrame(
        {
            "store_id": np.arange(n),
            "x": rng.uniform(2, GRID_KM - 2, n),
            "y": rng.uniform(2, GRID_KM - 2, n),
            "store_size_sqft": size,
            # hidden: layout slowness correlated with size, plus idiosyncratic part
            "layout": 0.8 + 0.35 * (size - 15_000) / 45_000 + rng.normal(0, 0.06, n),
            "overhead_min": rng.uniform(3.0, 5.5, n),
        }
    )
    m = cfg.n_shoppers
    tenure = rng.integers(10, 900, m).astype(float)
    skill = np.clip(0.75 + 0.35 * np.minimum(tenure / 400, 1) + rng.normal(0, 0.1, m), 0.6, 1.4)
    shoppers = pd.DataFrame(
        {
            "shopper_id": np.arange(m),
            "shopper_rating": np.clip(3.6 + 1.6 * (skill - 1) + rng.normal(0, 0.25, m), 1, 5),
            "shopper_tenure_days": tenure,
            "shopper_vehicle_kmh": rng.choice(VEHICLE_KMH, m),
            "skill": skill,  # hidden
            "speed_factor": np.clip(rng.normal(1.0, 0.08, m), 0.8, 1.2),  # hidden
        }
    )
    return World(stores, shoppers)


def sample_day_orders(world: World, cfg: Config, day: int, id_offset: int = 0) -> pd.DataFrame:
    """Orders for one day. Deterministic in (seed, day) so policies share demand (CRN)."""
    rng = np.random.default_rng([cfg.seed, 2, day])
    minutes = np.arange(truth.DAY_MIN)
    dow = day % 7
    rate = truth.demand_profile(minutes)
    rate = rate / rate.sum() * cfg.orders_per_day * truth.DOW_FACTOR[dow]
    counts = rng.poisson(rate)
    arrival = np.repeat(minutes, counts) + rng.random(counts.sum())
    n = len(arrival)
    st = world.stores
    store = rng.choice(len(st), n, p=np.full(len(st), 1 / len(st)))
    cx = np.clip(st.x.to_numpy()[store] + rng.normal(0, 2.5, n), 0, GRID_KM)
    cy = np.clip(st.y.to_numpy()[store] + rng.normal(0, 2.5, n), 0, GRID_KM)
    items = np.clip(np.round(rng.lognormal(2.7, 0.6, n)), 1, 80)
    dist = truth.ROAD_FACTOR * np.hypot(cx - st.x.to_numpy()[store], cy - st.y.to_numpy()[store])
    return pd.DataFrame(
        {
            "order_id": id_offset + np.arange(n),
            "day": day,
            "dow": dow,
            "arrival_min": arrival,
            "store_id": store,
            "cust_x": cx,
            "cust_y": cy,
            "items": items,
            "dist_km": dist,
            "pick_noise": truth.noise_multiplier(rng.standard_normal(n), truth.PICK_SIGMA * cfg.noise),
            "drive_noise": truth.noise_multiplier(rng.standard_normal(n), truth.DRIVE_SIGMA * cfg.noise),
            "u_action": rng.random(n),
        }
    )
