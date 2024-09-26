"""Off-policy evaluation: IPS, self-normalized IPS, direct method and doubly robust estimators.

Data model: one row per logged order = (context x, action a, propensity mu(a|x), reward r, day).
Confidence intervals come from a day-level (cluster) bootstrap, because orders on the same day
share a queue and are not independent.
"""
from __future__ import annotations

from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

CTX_COLS = ["ctx_items", "ctx_store_id", "ctx_hour", "ctx_dow", "ctx_dist_km", "ctx_pred_solo_min",
            "ctx_pending", "ctx_idle", "ctx_load", "ctx_store_recent"]
ESTIMATORS = ("ips", "snips", "dm", "dr")


def ctx_dict(log: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c[4:]: log[c].to_numpy(float) for c in CTX_COLS}


def target_probs(rule: Callable[[Mapping[str, np.ndarray]], np.ndarray], log: pd.DataFrame) -> np.ndarray:
    return rule(ctx_dict(log))


def fit_reward_model(X: np.ndarray, y: np.ndarray, seed: int = 0) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1, random_state=seed)
    return m.fit(X, y)


def crossfit_q(log: pd.DataFrame, reward: str, n_actions: int, folds: int = 3, seed: int = 0) -> np.ndarray:
    """Out-of-fold reward-model predictions q(x, a) for every action (days kept together in folds)."""
    X = log[CTX_COLS].to_numpy(float)
    a = log["action"].to_numpy(float)[:, None]
    y = log[reward].to_numpy(float)
    days = log["day"].to_numpy()
    uniq = np.unique(days)
    fold_of_day = {d: i % folds for i, d in enumerate(np.random.default_rng(seed).permutation(uniq))}
    fold = np.array([fold_of_day[d] for d in days])
    q = np.zeros((len(log), n_actions))
    for f in range(folds):
        tr, te = fold != f, fold == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        m = fit_reward_model(np.hstack([X[tr], a[tr]]), y[tr], seed)
        for k in range(n_actions):
            q[te, k] = m.predict(np.hstack([X[te], np.full((te.sum(), 1), float(k))]))
    return q


def per_order_terms(log: pd.DataFrame, reward: str, probs: np.ndarray, q: np.ndarray, clip: float | None) -> dict:
    a = log["action"].to_numpy(int)
    r = log[reward].to_numpy(float)
    idx = np.arange(len(log))
    w = probs[idx, a] / log["propensity"].to_numpy(float)
    if clip is not None:
        w = np.minimum(w, clip)
    dm = (probs * q).sum(1)
    return {"w": w, "wr": w * r, "dm": dm, "dr": dm + w * (r - q[idx, a])}


def _day_sums(days: np.ndarray, cols: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    uniq, inv = np.unique(days, return_inverse=True)
    out = {k: np.bincount(inv, weights=v, minlength=len(uniq)) for k, v in cols.items()}
    out["n"] = np.bincount(inv, minlength=len(uniq)).astype(float)
    return uniq, out


def _values(s: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    n = s["n"][idx].sum()
    return {
        "ips": s["wr"][idx].sum() / n,
        "snips": s["wr"][idx].sum() / max(s["w"][idx].sum(), 1e-12),
        "dm": s["dm"][idx].sum() / n,
        "dr": s["dr"][idx].sum() / n,
    }


def evaluate_policies(log: pd.DataFrame, reward: str, targets: Mapping[str, Callable], n_actions: int,
                      baseline: str, clip: float | None = 20.0, n_boot: int = 400, seed: int = 0,
                      alpha: float = 0.05) -> pd.DataFrame:
    """Estimate each target policy's mean reward and its lift over `baseline`, with bootstrap CIs.

    Returns long-form rows: policy, estimator, kind ('value' | 'lift'), estimate, lo, hi.
    """
    q = crossfit_q(log, reward, n_actions, seed=seed)
    days = log["day"].to_numpy()
    sums = {}
    for name, rule in targets.items():
        terms = per_order_terms(log, reward, target_probs(rule, log), q, clip)
        uniq, s = _day_sums(days, terms)
        sums[name] = s
    n_days = len(uniq)
    rng = np.random.default_rng(seed + 17)
    boots = [rng.integers(0, n_days, n_days) for _ in range(n_boot)]
    full = np.arange(n_days)
    rows = []
    point = {p: _values(s, full) for p, s in sums.items()}
    draws = {p: [_values(s, b) for b in boots] for p, s in sums.items()}
    for p in targets:
        for e in ESTIMATORS:
            bs = np.array([d[e] for d in draws[p]])
            rows.append((p, e, "value", point[p][e], *np.quantile(bs, [alpha / 2, 1 - alpha / 2])))
            if p != baseline:
                diff = bs - np.array([d[e] for d in draws[baseline]])
                rows.append((p, e, "lift", point[p][e] - point[baseline][e], *np.quantile(diff, [alpha / 2, 1 - alpha / 2])))
    return pd.DataFrame(rows, columns=["policy", "estimator", "kind", "estimate", "lo", "hi"])


def effective_sample_size(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
