"""Batching-cap rules and shopper-assignment strategies.

A policy = (cap rule, assignment strategy). The cap rule picks, per order at arrival, one of
three actions: the max batch size (1, 2 or 3) that order may be grouped into. The assignment
strategy then matches ready batches to idle shoppers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from ortools.graph.python import linear_sum_assignment

from . import truth

N_ACTIONS = 3  # action a -> batch cap a + 1
Ctx = Mapping[str, np.ndarray]


# ---- cap rules: ctx (dict of arrays) -> (n, 3) action probabilities --------------------------
def never_batch(ctx: Ctx) -> np.ndarray:
    p = np.zeros((len(ctx["items"]), N_ACTIONS))
    p[:, 0] = 1
    return p


def always_batch(ctx: Ctx) -> np.ndarray:
    p = np.zeros((len(ctx["items"]), N_ACTIONS))
    p[:, 2] = 1
    return p


RULE = dict(s2=8.0, s3=12.0, l3=1.5)


def ml_slack_rule(ctx: Ctx, sla: float = 60.0) -> np.ndarray:
    """Batch when the ML-predicted slack can absorb the hold time and partners are likely.

    slack = SLA - predicted solo delivery time - expected queueing delay. Partner likelihood is
    approximated by recent same-store arrivals. Congestion (load) lets us batch even more.
    """
    load = ctx["load"]
    slack = sla - ctx["pred_solo_min"] - 3.0 * load
    action = np.zeros(len(load), dtype=int)
    action[(ctx["store_recent"] >= 1) & (slack >= RULE["s2"])] = 1
    action[(ctx["store_recent"] >= 2) & (slack >= RULE["s3"]) & (load >= RULE["l3"])] = 2
    p = np.zeros((len(load), N_ACTIONS))
    p[np.arange(len(load)), action] = 1
    return p


def epsilon_mix(base: Callable[[Ctx], np.ndarray], eps: float) -> Callable[[Ctx], np.ndarray]:
    """Randomized logging policy: follow `base` w.p. 1-eps, else uniform. Propensities are known."""
    return lambda ctx: (1 - eps) * base(ctx) + eps / N_ACTIONS


# ---- assignment strategies -------------------------------------------------------------------
def _store_xy(sim, batch):
    s = sim.world.stores.iloc[batch[0].store]
    return s.x, s.y


def assign_nearest(sim, batches, idle, now):
    """Greedy: most urgent batch first, nearest idle shopper to the store."""
    idle = list(idle)
    pairs = []
    for b in batches:
        if not idle:
            break
        sx, sy = _store_xy(sim, b)
        j = min(range(len(idle)), key=lambda i: np.hypot(idle[i].x - sx, idle[i].y - sy))
        pairs.append((b, idle.pop(j)))
    return pairs


def assign_ml_eta(sim, batches, idle, now):
    """Deadline-aware assignment using the ML time model.

    For every candidate batch the model predicts completion time per idle shopper. Batches that can
    still make the SLA are served first, tightest slack first; batches already doomed go last.
    Each batch takes the shopper with the smallest predicted completion time.
    """
    if not batches or not idle:
        return []
    batches = batches[:12]
    cost = sim.predicted_completion(batches, idle, now)  # (batches, shoppers) minutes from now
    sla = sim.cfg.sla_min
    slack = np.array([sla - (now - min(o.arrival for o in b)) - cost[i].min() for i, b in enumerate(batches)])
    order = sorted(range(len(batches)), key=lambda i: (slack[i] < 0, slack[i] if slack[i] >= 0 else -slack[i]))
    free = list(range(len(idle)))
    pairs = []
    for bi in order:
        if not free:
            break
        j = min(free, key=lambda k: cost[bi, k])
        free.remove(j)
        pairs.append((batches[bi], idle[j]))
    return pairs


def assign_ortools(sim, batches, idle, now):
    """Globally optimal batch->shopper matching (OR-Tools min-cost assignment).

    Cost = rule-of-thumb completion time (no ML) minus the batch's waiting time, so older batches
    are served first and fast shoppers go where they save the most. The matrix is padded to square
    with zero-cost dummies and real costs are shifted below zero so real pairs are always preferred.
    """
    if not batches or not idle:
        return []
    nb, ns = len(batches), len(idle)
    n = max(nb, ns)
    cost = np.zeros((nb, ns))
    for i, b in enumerate(batches):
        sx, sy = _store_xy(sim, b)
        wait = now - min(o.arrival for o in b)
        pick_est = sum(0.6 * o.items for o in b) * (1 - 0.08 * (len(b) - 1))
        for j, s in enumerate(idle):
            travel = np.hypot(s.x - sx, s.y - sy) * truth.ROAD_FACTOR / s.speed * 60.0
            cost[i, j] = travel + pick_est / s.skill_obs - wait
    big = 100 + int(max(cost.max(), 0) * 100)
    solver = linear_sum_assignment.SimpleLinearSumAssignment()
    for i in range(n):
        for j in range(n):
            c = int(cost[i, j] * 100) - big if (i < nb and j < ns) else 0
            solver.add_arc_with_cost(i, j, c)
    if solver.solve() != solver.OPTIMAL:  # pragma: no cover
        return assign_nearest(sim, batches, idle, now)
    return [(batches[i], idle[solver.right_mate(i)]) for i in range(nb) if solver.right_mate(i) < ns]


@dataclass(frozen=True)
class Policy:
    name: str
    cap_rule: Callable[[Ctx], np.ndarray]
    assign: Callable
    description: str = ""


def greedy_nearest(**_):
    return Policy("greedy_nearest", never_batch, assign_nearest, "No batching, nearest idle shopper")


def ml_batching(sla: float = 60.0, **_):
    return Policy("ml_batching", lambda c: ml_slack_rule(c, sla), assign_ml_eta,
                  "ML-slack batching caps + ML predicted-completion assignment")


def ortools_assignment(**_):
    return Policy("ortools_assignment", never_batch, assign_ortools,
                  "No batching, globally optimal OR-Tools matching")


def main_policies(sla: float = 60.0) -> list[Policy]:
    return [greedy_nearest(), ml_batching(sla), ortools_assignment()]


def cap_policies(sla: float = 60.0) -> dict[str, Policy]:
    """Batching-cap policies under nearest-shopper assignment (the OPE target family)."""
    return {
        "never_batch": Policy("never_batch", never_batch, assign_nearest),
        "always_batch": Policy("always_batch", always_batch, assign_nearest),
        "ml_slack": Policy("ml_slack", lambda c: ml_slack_rule(c, sla), assign_nearest),
    }


def logging_policy(sla: float = 60.0, eps: float = 0.5) -> Policy:
    return Policy("logging_randomized", epsilon_mix(lambda c: ml_slack_rule(c, sla), eps), assign_nearest)
