import itertools
from types import SimpleNamespace

import numpy as np

from dispatchlab import policies as P
from dispatchlab.simulator import Order, Shopper


def mk_sim():
    import pandas as pd
    stores = pd.DataFrame({"x": [5.0], "y": [5.0]})
    return SimpleNamespace(world=SimpleNamespace(stores=stores), cfg=SimpleNamespace(sla_min=60))


def order(oid, arrival=0.0, items=10):
    return Order(oid, 0, arrival, items, 5, 5, 1.0, 1.0, 1.0, 20.0)


def shopper(sid, x, speed=26.0, skill=1.0):
    return Shopper(sid, x, 5.0, speed, 1.0, skill, skill)


def test_nearest_picks_closest():
    pairs = P.assign_nearest(mk_sim(), [[order(0)]], [shopper(0, 0.0), shopper(1, 4.5)], 0.0)
    assert pairs[0][1].sid == 1


def test_ortools_matches_brute_force_optimum():
    sim = mk_sim()
    batches = [[order(i, arrival=-5.0 * i, items=5 + 8 * i)] for i in range(3)]
    idle = [shopper(0, 0.0, 20, 0.7), shopper(1, 3.0, 32, 1.2), shopper(2, 9.0, 26, 1.0), shopper(3, 5.0, 26, 0.9)]
    pairs = P.assign_ortools(sim, batches, idle, 0.0)
    assert len(pairs) == 3 and len({s.sid for _, s in pairs}) == 3

    def cost(b, s):
        wait = -min(o.arrival for o in b)
        travel = np.hypot(s.x - 5, s.y - 5) * 1.3 / s.speed * 60
        return travel + sum(0.6 * o.items for o in b) / s.skill_obs - wait

    best = min(sum(cost(batches[i], idle[j]) for i, j in enumerate(perm))
               for perm in itertools.permutations(range(4), 3))
    got = sum(cost(b, s) for b, s in pairs)
    assert abs(got - best) < 0.05


def test_ortools_handles_more_batches_than_shoppers():
    pairs = P.assign_ortools(mk_sim(), [[order(i)] for i in range(5)], [shopper(0, 4.0), shopper(1, 6.0)], 0.0)
    assert len(pairs) == 2


def test_cap_rules_return_distributions():
    ctx = {"items": np.ones(4), "load": np.array([0.5, 1, 3, 6]), "pred_solo_min": np.full(4, 20.0),
           "store_recent": np.array([0, 1, 2, 5.0])}
    for rule in (P.never_batch, P.always_batch, P.ml_slack_rule, P.epsilon_mix(P.ml_slack_rule, 0.5)):
        p = rule(ctx)
        np.testing.assert_allclose(p.sum(1), 1)
        assert (p >= 0).all()
    assert (P.epsilon_mix(P.ml_slack_rule, 0.5)(ctx) >= 0.5 / 3).all()
    a = P.ml_slack_rule(ctx).argmax(1)
    assert a[0] == 0 and a[3] >= a[1]
