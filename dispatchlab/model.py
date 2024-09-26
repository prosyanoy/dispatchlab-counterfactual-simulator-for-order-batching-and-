"""Gradient-boosted pick-time and delivery(drive)-time model with offline evaluation."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, engineer

TARGETS = {"pick": "pick_min", "drive": "drive_min"}
PARAMS = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, n_jobs=1, verbose=-1)


class TimeModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def fit(self, raw: pd.DataFrame) -> "TimeModel":
        X = engineer(raw)
        for name, col in TARGETS.items():
            m = lgb.LGBMRegressor(random_state=self.seed, **PARAMS)
            m.fit(X, raw[col])
            self.models[name] = m
        return self

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Predict per-order pick and delivery (drive + handoff) minutes from raw feature rows."""
        X = engineer(raw)
        out = pd.DataFrame({n: m.booster_.predict(X.to_numpy()) for n, m in self.models.items()}, index=raw.index)
        out["total"] = out["pick"] + out["drive"]
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"seed": self.seed, "models": self.models}, path)

    @classmethod
    def load(cls, path: Path) -> "TimeModel":
        blob = joblib.load(path)
        m = cls(blob["seed"])
        m.models = blob["models"]
        return m

    def evaluate(self, raw: pd.DataFrame, bins: int = 10) -> dict:
        """MAE, p90 absolute error, bias and decile calibration for each target."""
        pred = self.predict(raw)
        report = {}
        for name, col in TARGETS.items():
            y, p = raw[col].to_numpy(), pred[name].to_numpy()
            err = np.abs(y - p)
            q = pd.qcut(p, bins, duplicates="drop")
            cal = pd.DataFrame({"pred": p, "actual": y}).groupby(q, observed=True).mean().reset_index(drop=True)
            baseline = (raw["items"] * raw["store_hist_min_per_item"] / raw["batch_size"]).to_numpy() \
                if name == "pick" else raw["leg_dist_km"].to_numpy() / raw["shopper_vehicle_kmh"].to_numpy() * 60 + 2.5
            report[name] = {
                "mae": float(err.mean()),
                "p90_abs_error": float(np.quantile(err, 0.9)),
                "bias": float((p - y).mean()),
                "calibration_mae": float((cal.pred - cal.actual).abs().mean()),
                "baseline_mae": float(np.abs(y - baseline).mean()),
                "calibration": cal.to_dict("list"),
            }
        return report
