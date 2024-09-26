import numpy as np
import pandas as pd

from dispatchlab import ope


def synthetic_log(n=6000, seed=0, days=30):
    """Contextual bandit with known truth. reward = 1 + x + 2*[a==1] - [a==2] + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    probs = np.column_stack([np.full(n, 0.4), 0.3 + 0.2 * (x > 0), 0.3 - 0.2 * (x > 0)])
    a = np.array([rng.choice(3, p=p) for p in probs])
    r = 1 + x + 2 * (a == 1) - (a == 2) + rng.normal(0, 0.5, n)
    cols = {c: np.zeros(n) for c in ope.CTX_COLS}
    cols["ctx_load"] = x
    df = pd.DataFrame(cols)
    df["action"], df["propensity"], df["r"] = a, probs[np.arange(n), a], r
    df["day"] = rng.integers(0, days, n)
    return df, x


def target(ctx):
    p = np.zeros((len(ctx["load"]), 3))
    p[:, 1] = 1.0
    return p


def uniform(ctx):
    return np.full((len(ctx["load"]), 3), 1 / 3)


def test_estimators_recover_known_value():
    log, x = synthetic_log()
    res = ope.evaluate_policies(log, "r", {"t": target, "u": uniform}, 3, "u", n_boot=200)
    v = res[(res.policy == "t") & (res.kind == "value")].set_index("estimator")
    true_t = 1 + 2  # E[x]=0
    for e in ("ips", "snips", "dr"):
        assert v.loc[e, "lo"] <= true_t <= v.loc[e, "hi"], e
    lift = res[(res.policy == "t") & (res.kind == "lift") & (res.estimator == "dr")].iloc[0]
    true_lift = 3 - (1 + (2 - 1) / 3)
    assert lift.lo <= true_lift <= lift.hi


def test_dr_has_lower_variance_than_ips():
    errs = {"ips": [], "dr": []}
    for seed in range(8):
        log, _ = synthetic_log(n=1500, seed=seed)
        res = ope.evaluate_policies(log, "r", {"t": target}, 3, "t", n_boot=20, seed=seed)
        v = res[res.kind == "value"].set_index("estimator").estimate
        for e in errs:
            errs[e].append(v[e] - 3)
    assert np.std(errs["dr"]) < np.std(errs["ips"])


def test_weights_and_clipping():
    log, _ = synthetic_log(n=500)
    q = np.zeros((500, 3))
    t = ope.per_order_terms(log, "r", target(ope.ctx_dict(log)), q, clip=None)
    assert t["w"].max() > 2
    assert ope.per_order_terms(log, "r", target(ope.ctx_dict(log)), q, clip=2.0)["w"].max() <= 2.0
    assert ope.effective_sample_size(np.ones(10)) == 10


def test_on_policy_weights_are_one():
    log, _ = synthetic_log(n=300)
    probs = np.zeros((300, 3))
    probs[np.arange(300), log.action] = log.propensity  # not a distribution; just checks ratio
    t = ope.per_order_terms(log, "r", probs, np.zeros((300, 3)), None)
    np.testing.assert_allclose(t["w"], 1.0)
