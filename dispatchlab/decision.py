"""Launch decision rule applied to the simulated policy comparison."""
from __future__ import annotations

import pandas as pd

from .experiments import paired_lift

BASELINE = "greedy_nearest"
COST_GUARDRAIL = 0.02  # max tolerated relative increase in total cost per order


def decide(per_day: pd.DataFrame, n_boot: int = 400, baseline: str = BASELINE) -> dict:
    """Pick the policy to launch.

    A candidate qualifies if its late-rate reduction vs the baseline has a 95% paired-bootstrap CI
    entirely below zero AND its total cost per order does not rise more than the guardrail.
    Among qualifiers the lowest late rate wins; if none qualifies the baseline stays.
    """
    metrics = ["late_rate", "total_cost_per_order", "attributed_cost_per_order", "idle_pct", "p90_delivery_min"]
    lifts = pd.concat([paired_lift(per_day, m, baseline, n_boot) for m in metrics], ignore_index=True)
    late = lifts[lifts.metric == "late_rate"].set_index("policy")
    cost = lifts[lifts.metric == "total_cost_per_order"].set_index("policy")
    qualifiers = [p for p in late.index if late.loc[p, "hi"] < 0 and cost.loc[p, "rel_change"] <= COST_GUARDRAIL]
    means = per_day.groupby("policy").late_rate.mean()
    choice = min(qualifiers, key=lambda p: means[p]) if qualifiers else baseline
    return {"recommended": choice, "qualifiers": qualifiers, "lifts": lifts, "baseline": baseline,
            "launch": choice != baseline}
