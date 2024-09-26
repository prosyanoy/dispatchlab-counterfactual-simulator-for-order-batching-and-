from fastapi.testclient import TestClient

from dispatchlab import experiments as E
from dispatchlab import policies as P
from dispatchlab.decision import decide
from dispatchlab.report import build_report


def test_decision_rule_and_report(small):
    cfg, world, model, raw = small
    per_day, orders = E.compare_policies(world, cfg, model)
    assert set(per_day.policy) == {"greedy_nearest", "ml_batching", "ortools_assignment"}
    dec = decide(per_day, 50)
    assert dec["recommended"] in set(per_day.policy)
    ope_out = E.ope_validation(world, cfg, model)
    res = ope_out["results"]
    assert {"ips", "snips", "dm", "dr"} <= set(res.estimator)
    assert (res.lo <= res.hi).all()
    ev = model.evaluate(raw[raw.day >= 11])
    path = build_report(cfg, ev, per_day, orders, ope_out, dec, cfg.output_dir)
    text = path.read_text()
    assert "Recommendation" in text and "CI coverage" in text
    for png in ("policy_comparison", "late_by_hour", "model_calibration", "ope_validation"):
        assert (cfg.output_dir / f"{png}.png").stat().st_size > 1000


def test_decision_keeps_baseline_when_no_gain():
    import pandas as pd
    rows = [{"day": d, "policy": p, "late_rate": 0.2 + 0.01 * (d % 3), "total_cost_per_order": 20.0,
             "attributed_cost_per_order": 12.0, "idle_pct": 0.4, "p90_delivery_min": 60.0}
            for d in range(10) for p in ("greedy_nearest", "other")]
    assert decide(pd.DataFrame(rows), 50)["recommended"] == "greedy_nearest"


def test_api(small, monkeypatch):
    cfg, world, model, _ = small
    from dispatchlab import api
    monkeypatch.setattr(api, "_state", lambda: (cfg, world, model))
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    body = {"store_id": 0, "shopper_id": 1, "items": 20, "leg_dist_km": 3.0, "start_min": 120, "day_of_week": 2}
    r = c.post("/predict", json=body).json()
    assert r["pick_min"] > 0 and abs(r["total_min"] - r["pick_min"] - r["delivery_min"]) < 1e-9
    assert c.post("/predict", json={**body, "store_id": 99}).status_code == 404
    assert c.post("/predict", json={**body, "items": 0}).status_code == 422
    s = c.post("/simulate", json={"policy": "ortools_assignment", "days": 1}).json()
    assert 0 <= s["metrics"]["late_rate"] <= 1
    assert c.post("/simulate", json={"policy": "nope"}).status_code == 404
    assert len(c.get("/policies").json()) == 3
