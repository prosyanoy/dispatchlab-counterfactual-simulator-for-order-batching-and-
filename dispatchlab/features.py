"""Feature engineering shared verbatim by training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import truth
from .world import World

RAW_COLUMNS = [
    "dow", "items", "batch_size", "stop_index", "leg_dist_km", "start_min", "store_size_sqft",
    "store_hist_min_per_item", "shopper_rating", "shopper_tenure_days", "shopper_vehicle_kmh",
]
FEATURES = [
    "hour", "dow", "is_weekend", "items", "log_items", "batch_size", "stop_index", "leg_dist_km",
    "store_size_sqft", "store_hist_min_per_item", "shopper_rating", "shopper_tenure_days",
    "shopper_vehicle_kmh", "pick_prior_min", "leg_prior_min",
]


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature transform (no cross-row dependence, so batch == single-row)."""
    f = pd.DataFrame(index=raw.index)
    f["hour"] = truth.hour_of(raw["start_min"].to_numpy(float))
    f["dow"] = raw["dow"].astype(float)
    f["is_weekend"] = (raw["dow"] >= 5).astype(float)
    f["items"] = raw["items"].astype(float)
    f["log_items"] = np.log1p(raw["items"].astype(float))
    for c in ("batch_size", "stop_index", "leg_dist_km", "store_size_sqft", "store_hist_min_per_item",
              "shopper_rating", "shopper_tenure_days", "shopper_vehicle_kmh"):
        f[c] = raw[c].astype(float)
    f["pick_prior_min"] = f["items"] * f["store_hist_min_per_item"] / f["batch_size"]
    f["leg_prior_min"] = f["leg_dist_km"] / f["shopper_vehicle_kmh"] * 60.0
    return f[FEATURES]


def serving_frame(world: World, store_id, shopper_id, items, batch_size, stop_index, leg_dist_km,
                  start_min, dow) -> pd.DataFrame:
    """Assemble raw feature rows by looking up reference data in memory (the serving path)."""
    store_id = np.asarray(store_id, dtype=int)
    shopper_id = np.asarray(shopper_id, dtype=int)
    st = world.stores.set_index("store_id")
    sh = world.shoppers.set_index("shopper_id")
    n = len(store_id)
    return pd.DataFrame(
        {
            "dow": np.broadcast_to(np.asarray(dow), n),
            "items": np.broadcast_to(np.asarray(items, dtype=float), n),
            "batch_size": np.broadcast_to(np.asarray(batch_size), n),
            "stop_index": np.broadcast_to(np.asarray(stop_index), n),
            "leg_dist_km": np.broadcast_to(np.asarray(leg_dist_km, dtype=float), n),
            "start_min": np.broadcast_to(np.asarray(start_min, dtype=float), n),
            "store_size_sqft": st.store_size_sqft.to_numpy()[store_id],
            "store_hist_min_per_item": st.store_hist_min_per_item.to_numpy()[store_id],
            "shopper_rating": sh.shopper_rating.to_numpy()[shopper_id],
            "shopper_tenure_days": sh.shopper_tenure_days.to_numpy()[shopper_id],
            "shopper_vehicle_kmh": sh.shopper_vehicle_kmh.to_numpy()[shopper_id],
        }
    )
