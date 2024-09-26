"""Event-driven dispatch simulator (arrivals, hold expiries, shopper-free events)."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import truth
from .config import Config
from .features import serving_frame
from .model import TimeModel
from .policies import N_ACTIONS, Policy
from .world import World

SLOT_MIN = 30.0  # decision slot length for cluster-randomized batching actions
MAX_PAIR_KM = 5.0  # max customer-to-customer distance within a batch


@dataclass
class Order:
    oid: int
    store: int
    arrival: float
    items: float
    cx: float
    cy: float
    dist_km: float
    pick_noise: float
    drive_noise: float
    pred_solo: float
    cap: int = 1
    action: int = 0
    propensity: float = 1.0
    ctx: dict = field(default_factory=dict)
    dispatched: float = np.nan
    delivered: float = np.nan
    batch_size: int = 1
    cost: float = np.nan


@dataclass
class Shopper:
    sid: int
    x: float
    y: float
    speed: float
    speed_factor: float
    skill: float
    skill_obs: float  # what a rule-of-thumb planner can see (rating-derived)
    free_at: float = 0.0
    busy: list = field(default_factory=list)  # (start, end) intervals


@dataclass
class SimResult:
    orders: pd.DataFrame
    metrics: dict


class Simulator:
    def __init__(self, world: World, cfg: Config, policy: Policy, model: TimeModel):
        self.world, self.cfg, self.policy, self.model = world, cfg, policy, model
        self.sx = world.stores.x.to_numpy()
        self.sy = world.stores.y.to_numpy()

    # ---- helpers used by policies ----
    def predicted_completion(self, batches, idle, now) -> np.ndarray:
        """ML-predicted minutes from `now` until each batch is fully delivered, per idle shopper."""
        rows, index = [], []
        for bi, b in enumerate(batches):
            route = self._route(b)
            sx, sy = self.sx[b[0].store], self.sy[b[0].store]
            k = len(b)
            for si, s in enumerate(idle):
                travel = np.hypot(s.x - sx, s.y - sy) * truth.ROAD_FACTOR / s.speed * 60.0
                px, py = sx, sy
                for stop, o in enumerate(route):
                    leg = truth.ROAD_FACTOR * np.hypot(o.cx - px, o.cy - py)
                    rows.append((b[0].store, s.sid, o.items, k, stop, leg, now + travel))
                    index.append((bi, si))
                    px, py = o.cx, o.cy
        a = np.array(rows, dtype=float)
        dow = int(self.day % 7)
        raw = serving_frame(self.world, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], dow)
        pred = self.model.predict(raw)
        cost = np.zeros((len(batches), len(idle)))
        for (bi, si), t in zip(index, pred["total"].to_numpy()):
            cost[bi, si] += t
        for bi, b in enumerate(batches):
            sx, sy = self.sx[b[0].store], self.sy[b[0].store]
            for si, s in enumerate(idle):
                cost[bi, si] += np.hypot(s.x - sx, s.y - sy) * truth.ROAD_FACTOR / s.speed * 60.0
        return cost

    def _route(self, batch):
        left, route = list(batch), []
        px, py = self.sx[batch[0].store], self.sy[batch[0].store]
        while left:
            o = min(left, key=lambda o: np.hypot(o.cx - px, o.cy - py))
            left.remove(o)
            route.append(o)
            px, py = o.cx, o.cy
        return route

    # ---- main loop ----
    def run(self, orders: pd.DataFrame, day: int) -> SimResult:
        self.day = day
        dow = day % 7
        w, cfg = self.world, self.cfg
        sh = w.shoppers
        self.shoppers = [
            Shopper(int(r.shopper_id), float(self.sx[i % len(self.sx)]), float(self.sy[i % len(self.sx)]),
                    float(r.shopper_vehicle_kmh), float(r.speed_factor), float(r.skill),
                    float(np.clip(0.75 + 0.3 * (r.shopper_rating - 3.6) / 1.6 + 0.0, 0.6, 1.4)))
            for i, r in enumerate(sh.itertuples())
        ]
        pred_solo = self._predict_solo(orders)
        objs = [
            Order(int(r.order_id), int(r.store_id), float(r.arrival_min), float(r["items"]), float(r.cust_x),
                  float(r.cust_y), float(r.dist_km), float(r.pick_noise), float(r.drive_noise), float(p))
            for (_, r), p in zip(orders.iterrows(), pred_solo)
        ]
        u = orders.u_action.to_numpy()
        self.pending: list[Order] = []
        self.idle = list(self.shoppers)
        self._arrivals: list[tuple[float, int]] = []
        self._slot_actions: dict = {}
        heap = []
        self._seq = itertools.count()
        self._fuel = 0.0
        for o in objs:
            heapq.heappush(heap, (o.arrival, 1, next(self._seq), o))
        while heap:
            now, kind, _, payload = heapq.heappop(heap)
            if kind == 0:  # shopper becomes free
                self.idle.append(payload)
            elif kind == 1:  # order arrival
                i = payload.oid - objs[0].oid
                self._on_arrival(payload, u[i], now, dow, heap)
            self._dispatch(now, dow, heap)
        return self._summarize(objs, orders)

    def _predict_solo(self, orders: pd.DataFrame) -> np.ndarray:
        n = len(orders)
        if n == 0:
            return np.zeros(0)
        sh = self.world.shoppers
        nominal = int((sh.skill - sh.skill.median()).abs().idxmin())  # a median shopper
        raw = serving_frame(self.world, orders.store_id, np.full(n, nominal), orders["items"], 1, 0,
                            orders.dist_km, orders.arrival_min + 6.0, self.day % 7)
        return self.model.predict(raw)["total"].to_numpy()

    def _on_arrival(self, o: Order, u: float, now: float, dow: int, heap) -> None:
        """Choose the batching action. It is made once per (store, time slot), on the slot's first
        order, and shared by every order of that slot (cluster randomization): this keeps batch
        partners under the same treatment so logged actions are close to non-interfering."""
        key = (o.store, int(now // SLOT_MIN))
        if key not in self._slot_actions:
            n_idle = len(self.idle)
            recent = sum(1 for t, s in self._arrivals if s == o.store and now - t <= 20.0)
            ctx = {
                "items": o.items, "store_id": o.store, "hour": float(truth.hour_of(now)), "dow": dow,
                "dist_km": o.dist_km, "pred_solo_min": o.pred_solo, "pending": len(self.pending),
                "idle": n_idle, "load": (len(self.pending) + 1) / (n_idle + 1), "store_recent": recent,
            }
            probs = self.policy.cap_rule({k: np.array([v]) for k, v in ctx.items()})[0]
            a = int(min(np.searchsorted(np.cumsum(probs), u, side="right"), N_ACTIONS - 1))
            self._slot_actions[key] = (a, float(probs[a]), ctx)
        self._arrivals.append((now, o.store))
        a, prop, ctx = self._slot_actions[key]
        o.action, o.propensity, o.cap, o.ctx = a, prop, a + 1, ctx
        self.pending.append(o)
        if o.cap > 1:
            heapq.heappush(heap, (now + self.cfg.hold_min, 2, next(self._seq), o))

    def _ready_batches(self, now: float) -> list[list[Order]]:
        by_store: dict[int, list[Order]] = {}
        for o in sorted(self.pending, key=lambda o: o.arrival):
            by_store.setdefault(o.store, []).append(o)
        batches = []
        for lst in by_store.values():
            rest = lst
            while rest:
                seed = rest[0]
                members = [seed]
                if seed.cap > 1:
                    cands = sorted(rest[1:], key=lambda o: np.hypot(o.cx - seed.cx, o.cy - seed.cy))
                    for o in cands:
                        if len(members) >= seed.cap:
                            break
                        if o.cap >= 2 and np.hypot(o.cx - seed.cx, o.cy - seed.cy) * truth.ROAD_FACTOR <= MAX_PAIR_KM:
                            members.append(o)
                    while len(members) > min(m.cap for m in members):
                        members.pop()
                ready = seed.cap == 1 or len(members) >= seed.cap or now >= seed.arrival + self.cfg.hold_min - 1e-9
                ids = {m.oid for m in members}
                rest = [o for o in rest if o.oid not in ids]
                if ready:
                    batches.append(members)
        batches.sort(key=lambda b: min(o.arrival for o in b))
        return batches

    def _dispatch(self, now: float, dow: int, heap) -> None:
        if not self.pending or not self.idle:
            return
        batches = self._ready_batches(now)
        if not batches:
            return
        for batch, s in self.policy.assign(self, batches, list(self.idle), now):
            self.idle.remove(s)
            ids = {o.oid for o in batch}
            self.pending = [o for o in self.pending if o.oid not in ids]
            self._start(batch, s, now, dow, heap)

    def _start(self, batch, s: Shopper, now: float, dow: int, heap) -> None:
        st = self.world.stores.iloc[batch[0].store]
        k = len(batch)
        route = self._route(batch)
        travel = truth.drive_minutes(np.hypot(s.x - st.x, s.y - st.y) * truth.ROAD_FACTOR, s.speed, s.speed_factor,
                                     truth.hour_of(now), dow, 1.0, handoff=0.0)
        t = now + travel
        km = np.hypot(s.x - st.x, s.y - st.y) * truth.ROAD_FACTOR
        for o in batch:
            t += float(truth.pick_minutes(o.items, k, st.layout, st.overhead_min, truth.hour_of(t), s.skill, o.pick_noise))
        px, py = st.x, st.y
        for o in route:
            leg = truth.ROAD_FACTOR * np.hypot(o.cx - px, o.cy - py)
            t += float(truth.drive_minutes(leg, s.speed, s.speed_factor, truth.hour_of(t), dow, o.drive_noise))
            km += leg
            o.delivered, px, py = t, o.cx, o.cy
        self._fuel += self.cfg.fuel_per_km * km
        cost = (self.cfg.wage_per_min * (t - now) + self.cfg.fuel_per_km * km) / k
        for o in batch:
            o.dispatched, o.batch_size, o.cost = now, k, cost
        s.busy.append((now, t))
        s.free_at, s.x, s.y = t, px, py
        heapq.heappush(heap, (t, 0, next(self._seq), s))

    def _summarize(self, objs: list[Order], orders: pd.DataFrame) -> SimResult:
        df = pd.DataFrame(
            {
                "order_id": [o.oid for o in objs], "store_id": [o.store for o in objs],
                "arrival_min": [o.arrival for o in objs], "action": [o.action for o in objs],
                "propensity": [o.propensity for o in objs], "batch_size": [o.batch_size for o in objs],
                "dispatched_min": [o.dispatched for o in objs], "delivered_min": [o.delivered for o in objs],
                "order_cost": [o.cost for o in objs],
            }
        )
        for key in ("items", "store_id", "hour", "dow", "dist_km", "pred_solo_min", "pending", "idle", "load",
                    "store_recent"):
            df["ctx_" + key] = [o.ctx[key] for o in objs]
        df["day"] = self.day
        df["total_min"] = df.delivered_min - df.arrival_min
        df["late"] = (df.total_min > self.cfg.sla_min).astype(float)
        df["wait_min"] = df.dispatched_min - df.arrival_min
        n = max(len(df), 1)
        idle_min, labor = [], 0.0
        for s in self.shoppers:
            busy = sum(max(0.0, min(e, truth.DAY_MIN) - min(b, truth.DAY_MIN)) for b, e in s.busy)
            idle_min.append(truth.DAY_MIN - busy)
            labor += self.cfg.wage_per_min * max(truth.DAY_MIN, s.free_at)
        metrics = {
            "orders": len(df),
            "late_rate": float(df.late.mean()) if len(df) else 0.0,
            "mean_delivery_min": float(df.total_min.mean()) if len(df) else 0.0,
            "p90_delivery_min": float(df.total_min.quantile(0.9)) if len(df) else 0.0,
            "idle_min_per_shopper": float(np.mean(idle_min)),
            "idle_pct": float(np.mean(idle_min) / truth.DAY_MIN),
            "attributed_cost_per_order": float(df.order_cost.mean()) if len(df) else 0.0,
            "total_cost_per_order": float((labor + self._fuel) / n),
            "mean_batch_size": float(df.batch_size.mean()) if len(df) else 1.0,
        }
        return SimResult(df, metrics)
