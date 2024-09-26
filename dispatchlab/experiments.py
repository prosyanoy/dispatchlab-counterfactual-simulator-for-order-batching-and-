"""End-to-end experiments: model training, policy comparison, OPE validation, launch decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ope, policies as P
from .config import Config
from .datagen import generate
from .model import TimeModel
from .simulator import Simulator
from .warehouse import Warehouse
from .world import World, sample_day_orders

EVAL_DAY0 = 3003  # day offsets are multiples of 7 so every block covers all weekdays equally
TRUTH_DAY0 = 5005
LOG_DAY0 = 7000
REWARDS = ("late", "order_cost")


def prepare(cfg: Config) -> tuple[World, TimeModel, dict]:
    """Generate data, train the model on the first 80% of days, evaluate on the rest."""
    world = generate(cfg)
    raw = Warehouse(cfg.data_dir).raw_training_frame()
    cut = int(cfg.history_days * 0.8)
    model = TimeModel(cfg.seed).fit(raw[raw.day < cut])
    evaluation = model.evaluate(raw[raw.day >= cut])
    model = TimeModel(cfg.seed).fit(raw)  # refit on everything for use in the simulator/API
    model.save(cfg.data_dir / "time_model.joblib")
    return world, model, evaluation


def simulate_days(world, cfg, policy, model, days, keep_orders=False):
    metrics, orders = [], []
    for d in days:
        res = Simulator(world, cfg, policy, model).run(sample_day_orders(world, cfg, d), d)
        metrics.append({"day": d, **res.metrics})
        if keep_orders:
            orders.append(res.orders)
    return pd.DataFrame(metrics), (pd.concat(orders, ignore_index=True) if keep_orders else None)


def compare_policies(world, cfg, model, policies=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every policy on the same simulated days (common random numbers)."""
    policies = policies or P.main_policies(cfg.sla_min)
    days = range(EVAL_DAY0, EVAL_DAY0 + cfg.eval_days)
    frames, orders = [], []
    for pol in policies:
        m, o = simulate_days(world, cfg, pol, model, days, keep_orders=True)
        m["policy"], o["policy"] = pol.name, pol.name
        frames.append(m)
        orders.append(o)
    return pd.concat(frames, ignore_index=True), pd.concat(orders, ignore_index=True)


def paired_lift(per_day: pd.DataFrame, metric: str, baseline: str, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Relative change of `metric` vs baseline with a paired bootstrap CI over days."""
    wide = per_day.pivot(index="day", columns="policy", values=metric)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(wide), len(wide)) for _ in range(n_boot)]
    rows = []
    for p in wide.columns:
        if p == baseline:
            continue
        rel = np.array([wide[p].iloc[i].mean() / wide[baseline].iloc[i].mean() - 1 for i in idx])
        pt = wide[p].mean() / wide[baseline].mean() - 1
        rows.append({"policy": p, "metric": metric, "rel_change": pt, "lo": np.quantile(rel, 0.025),
                     "hi": np.quantile(rel, 0.975)})
    return pd.DataFrame(rows)


def logged_data(world, cfg, model, replicate: int) -> pd.DataFrame:
    pol = P.logging_policy(cfg.sla_min)
    days = range(LOG_DAY0 + replicate * cfg.ope_log_days, LOG_DAY0 + (replicate + 1) * cfg.ope_log_days)
    return simulate_days(world, cfg, pol, model, days, keep_orders=True)[1]


def ground_truth(world, cfg, model) -> pd.DataFrame:
    """Simulate each target policy online on fresh days: the counterfactual ground truth."""
    rows = []
    for name, pol in P.cap_policies(cfg.sla_min).items():
        days = range(TRUTH_DAY0, TRUTH_DAY0 + cfg.ope_truth_days)
        _, o = simulate_days(world, cfg, pol, model, days, keep_orders=True)
        for r in REWARDS:
            per_day = o.groupby("day")[r].mean()
            rows.append({"policy": name, "reward": r, "truth": float(o[r].mean()),
                         "truth_se": float(per_day.std(ddof=1) / np.sqrt(len(per_day)))})
    return pd.DataFrame(rows)


def ope_validation(world, cfg, model) -> dict:
    """Run OPE on R independent logged datasets and compare with simulator ground truth."""
    targets = {k: v.cap_rule for k, v in P.cap_policies(cfg.sla_min).items()}
    truth = ground_truth(world, cfg, model)
    tv = truth.set_index(["policy", "reward"])
    base_truth = {r: tv.loc[("never_batch", r), "truth"] for r in REWARDS}
    results, ess = [], []
    for rep in range(cfg.ope_replicates):
        log = logged_data(world, cfg, model, rep)
        for r in REWARDS:
            res = ope.evaluate_policies(log, r, targets, P.N_ACTIONS, "never_batch", n_boot=cfg.bootstrap, seed=rep)
            res["reward"], res["replicate"] = r, rep
            res["truth"] = [
                tv.loc[(p, r), "truth"] - (base_truth[r] if k == "lift" else 0.0)
                for p, k in zip(res.policy, res.kind)
            ]
            results.append(res)
        for name, rule in targets.items():
            w = ope.per_order_terms(log, "late", ope.target_probs(rule, log),
                                    np.zeros((len(log), P.N_ACTIONS)), 20.0)["w"]
            ess.append({"replicate": rep, "policy": name, "ess_fraction": ope.effective_sample_size(w) / len(w)})
    allres = pd.concat(results, ignore_index=True)
    allres["covered"] = (allres.lo <= allres.truth) & (allres.truth <= allres.hi)
    allres["abs_err"] = (allres.estimate - allres.truth).abs()
    return {"truth": truth, "results": allres, "ess": pd.DataFrame(ess)}
