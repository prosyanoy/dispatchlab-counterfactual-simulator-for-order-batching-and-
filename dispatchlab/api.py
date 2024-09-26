"""FastAPI service: time predictions and quick policy simulations.

On first use, if no trained model exists in DISPATCHLAB_DATA_DIR, a small demo dataset is generated
and a model trained (safe local default; no external services or secrets).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import policies as P
from .config import Config
from .experiments import prepare, simulate_days
from .features import serving_frame
from .model import TimeModel
from .world import World

app = FastAPI(title="DispatchLab", version="0.1.0")


@lru_cache(maxsize=1)
def _state():
    cfg = Config.from_env()
    if not (cfg.data_dir / "time_model.joblib").exists():
        cfg = replace(cfg, history_days=min(cfg.history_days, 14))
        world, model, _ = prepare(cfg)
        return cfg, world, model
    return cfg, World.load(cfg.data_dir), TimeModel.load(cfg.data_dir / "time_model.joblib")


class PredictRequest(BaseModel):
    store_id: int = Field(ge=0)
    shopper_id: int = Field(ge=0)
    items: int = Field(ge=1, le=200)
    leg_dist_km: float = Field(gt=0, le=50)
    start_min: float = Field(ge=0, le=900, description="Minutes since 08:00")
    day_of_week: int = Field(ge=0, le=6)
    batch_size: int = Field(default=1, ge=1, le=3)
    stop_index: int = Field(default=0, ge=0, le=2)


class SimulateRequest(BaseModel):
    policy: str = "greedy_nearest"
    days: int = Field(default=1, ge=1, le=7)
    first_day: int = Field(default=9000, ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    _, world, model = _state()
    if req.store_id >= len(world.stores) or req.shopper_id >= len(world.shoppers):
        raise HTTPException(404, "unknown store_id or shopper_id")
    raw = serving_frame(world, [req.store_id], [req.shopper_id], req.items, req.batch_size, req.stop_index,
                        req.leg_dist_km, req.start_min, req.day_of_week)
    p = model.predict(raw).iloc[0]
    return {"pick_min": float(p["pick"]), "delivery_min": float(p["drive"]), "total_min": float(p["total"])}


@app.get("/policies")
def list_policies():
    return [{"name": p.name, "description": p.description} for p in P.main_policies()]


@app.post("/simulate")
def simulate(req: SimulateRequest):
    cfg, world, model = _state()
    pols = {p.name: p for p in P.main_policies(cfg.sla_min)}
    if req.policy not in pols:
        raise HTTPException(404, f"unknown policy; choose from {sorted(pols)}")
    m, _ = simulate_days(world, cfg, pols[req.policy], model, range(req.first_day, req.first_day + req.days))
    return {"policy": req.policy, "days": req.days, "metrics": m.drop(columns="day").mean().to_dict()}
