import numpy as np
import pandas as pd

from dispatchlab import policies as P
from dispatchlab.simulator import Simulator
from dispatchlab.world import sample_day_orders


def run(small, policy, day=40):
    cfg, world, model, _ = small
    return Simulator(world, cfg, policy, model).run(sample_day_orders(world, cfg, day), day), cfg


def test_every_order_is_delivered_after_arrival(small):
    for pol in P.main_policies():
        res, _ = run(small, pol)
        o = res.orders
        assert o.delivered_min.notna().all()
        assert (o.delivered_min > o.dispatched_min).all()
        assert (o.dispatched_min >= o.arrival_min - 1e-9).all()
        assert 0 <= res.metrics["idle_pct"] <= 1
        assert 0 <= res.metrics["late_rate"] <= 1


def test_deterministic_and_paired_demand(small):
    a, _ = run(small, P.greedy_nearest())
    b, _ = run(small, P.greedy_nearest())
    pd.testing.assert_frame_equal(a.orders, b.orders)
    c, _ = run(small, P.ortools_assignment())
    assert (a.orders.order_id == c.orders.order_id).all()
    assert not np.allclose(a.orders.delivered_min, c.orders.delivered_min)


def test_no_batching_policies_never_batch_and_batching_policy_does(small):
    g, _ = run(small, P.greedy_nearest())
    assert (g.orders.batch_size == 1).all()
    b, _ = run(small, P.cap_policies()["always_batch"])
    assert b.orders.batch_size.max() <= 3 and b.orders.batch_size.max() > 1
    assert b.metrics["attributed_cost_per_order"] < g.metrics["attributed_cost_per_order"]


def test_logged_propensities_are_valid(small):
    res, _ = run(small, P.logging_policy())
    o = res.orders
    assert o.propensity.between(1 / 6 - 1e-9, 1.0).all()
    assert set(o.action.unique()) <= {0, 1, 2}
    assert o.action.nunique() > 1


def test_batches_share_a_store(small):
    res, _ = run(small, P.cap_policies()["always_batch"])
    o = res.orders
    for _, g in o[o.batch_size > 1].groupby(["dispatched_min", "store_id"]):
        assert len(g) <= 3
