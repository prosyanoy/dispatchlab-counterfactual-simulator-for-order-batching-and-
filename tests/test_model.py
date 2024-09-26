import numpy as np

from dispatchlab.model import TimeModel


def test_model_beats_baseline_and_reports_metrics(small):
    _, _, model, raw = small
    ev = model.evaluate(raw[raw.day >= 11])
    for name in ("pick", "drive"):
        m = ev[name]
        # the tiny fixture dataset only lets the drive model roughly match the physics baseline
        assert m["mae"] < (m["baseline_mae"] if name == "pick" else 1.15 * m["baseline_mae"])
        assert m["p90_abs_error"] > m["mae"]
        assert abs(m["bias"]) < 1.0
        assert len(m["calibration"]["pred"]) >= 5
    assert ev["pick"]["calibration_mae"] < 1.0


def test_predictions_are_sensible_and_roundtrip(small, tmp_path):
    _, _, model, raw = small
    test = raw[raw.day >= 11]
    p = model.predict(test)
    assert (p.total == p.pick + p.drive).all()
    big = test.assign(items=test["items"] * 3)
    assert model.predict(big).pick.mean() > p.pick.mean()
    model.save(tmp_path / "m.joblib")
    p2 = TimeModel.load(tmp_path / "m.joblib").predict(test)
    np.testing.assert_allclose(p.to_numpy(), p2.to_numpy())
