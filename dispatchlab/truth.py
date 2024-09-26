"""Ground-truth physics of the synthetic grocery world.

The ML model never sees these functions; it only sees the resulting (noisy) logs and a
subset of the latent variables. The simulator uses the same functions to realize times.
"""
from __future__ import annotations

import numpy as np

DAY_MIN = 720  # service window 08:00-20:00, in minutes
DAY_START_HOUR = 8.0
DOW_FACTOR = np.array([0.9, 0.9, 0.95, 1.0, 1.1, 1.3, 1.25])
HANDOFF_MIN = 2.5
ROAD_FACTOR = 1.3


def _bump(h, center, width):
    return np.exp(-(((h - center) / width) ** 2))


def hour_of(minute):
    return DAY_START_HOUR + np.asarray(minute, dtype=float) / 60.0


def demand_profile(minutes: np.ndarray) -> np.ndarray:
    """Relative arrival intensity by minute of day (lunch and evening peaks)."""
    h = hour_of(minutes)
    return 0.5 + 0.9 * _bump(h, 12.5, 1.5) + 1.3 * _bump(h, 18.0, 1.5)


def traffic_speed_factor(hour, dow):
    """Multiplier on driving speed (<1 in rush hour, slightly worse on weekends)."""
    slow = 0.25 * _bump(hour, 17.8, 1.2) + 0.15 * _bump(hour, 12.5, 1.0)
    return 1.0 / (1.0 + slow + 0.05 * (np.asarray(dow) >= 5))


def store_busy_factor(hour):
    return 1.0 + 0.25 * _bump(hour, 12.5, 1.5) + 0.30 * _bump(hour, 18.0, 1.5)


def pick_minutes(items, batch_size, layout, overhead, hour, skill, noise_mult):
    """Per-order picking minutes. Store overhead is shared across a batch."""
    per_item = 0.55 * layout * store_busy_factor(hour) / skill
    eff = 1.0 - 0.08 * (np.asarray(batch_size) - 1)
    return overhead / batch_size + np.power(items, 0.92) * per_item * eff * noise_mult


def drive_minutes(leg_km, speed_kmh, speed_factor, hour, dow, noise_mult, handoff=HANDOFF_MIN):
    """Minutes to drive one leg (and hand off, unless handoff=0)."""
    v = speed_kmh * speed_factor * traffic_speed_factor(hour, dow)
    return leg_km / v * 60.0 * noise_mult + handoff


def noise_multiplier(z, sigma):
    """Mean-one lognormal multiplier from a standard normal draw."""
    return np.exp(sigma * z - 0.5 * sigma**2)


PICK_SIGMA = 0.28
DRIVE_SIGMA = 0.22
