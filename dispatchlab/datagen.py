"""Synthetic history generator: writes orders/shoppers/stores Parquet files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import truth
from .config import Config
from .world import World, make_world, sample_day_orders


def generate_history(world: World, cfg: Config) -> pd.DataFrame:
    """Historical delivered orders under a random (exploratory) dispatch process."""
    rng = np.random.default_rng([cfg.seed, 3])
    frames, offset = [], 0
    for day in range(cfg.history_days):
        o = sample_day_orders(world, cfg, day=1000 + day, id_offset=offset)
        o["day"] = day
        o["dow"] = day % 7
        frames.append(o)
        offset += len(o)
    o = pd.concat(frames, ignore_index=True)
    n = len(o)
    k = rng.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1])
    stop = np.floor(rng.random(n) * k).astype(int)
    leg = o.dist_km.to_numpy() * np.where(stop == 0, 1.0, rng.uniform(0.2, 0.9, n))
    sid = rng.integers(0, len(world.shoppers), n)
    start = o.arrival_min.to_numpy() + rng.uniform(0, 12, n)
    hour = truth.hour_of(start)
    st, sh = world.stores, world.shoppers
    pick = truth.pick_minutes(
        o["items"].to_numpy(), k, st.layout.to_numpy()[o.store_id], st.overhead_min.to_numpy()[o.store_id],
        hour, sh.skill.to_numpy()[sid], o.pick_noise.to_numpy(),
    )
    drive = truth.drive_minutes(
        leg, sh.shopper_vehicle_kmh.to_numpy()[sid], sh.speed_factor.to_numpy()[sid],
        truth.hour_of(start + pick), o.dow.to_numpy(), o.drive_noise.to_numpy(),
    )
    out = o[["order_id", "day", "dow", "store_id", "items", "dist_km"]].copy()
    out["shopper_id"] = sid
    out["batch_size"] = k
    out["stop_index"] = stop
    out["leg_dist_km"] = leg
    out["start_min"] = start
    out["pick_min"] = pick
    out["drive_min"] = drive
    return out


def generate(cfg: Config) -> World:
    """Create the world + history and persist everything under cfg.data_dir."""
    world = make_world(cfg)
    world.save(cfg.data_dir)
    history = generate_history(world, cfg)
    history.to_parquet(cfg.data_dir / "orders.parquet", index=False)
    from .warehouse import Warehouse

    Warehouse(cfg.data_dir).write_store_stats(train_days=cfg.history_days)
    return World.load(cfg.data_dir)
