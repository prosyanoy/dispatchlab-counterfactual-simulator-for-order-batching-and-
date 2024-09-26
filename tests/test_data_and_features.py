import numpy as np
import pandas as pd

from dispatchlab.config import Config
from dispatchlab.datagen import generate_history
from dispatchlab.features import FEATURES, engineer, serving_frame
from dispatchlab.warehouse import Warehouse
from dispatchlab.world import make_world, sample_day_orders


def test_generation_is_deterministic_and_noise_is_configurable():
    cfg = Config(history_days=3, orders_per_day=100)
    w = make_world(cfg)
    a, b = generate_history(w, cfg), generate_history(w, cfg)
    pd.testing.assert_frame_equal(a, b)
    noisy = generate_history(w, Config(history_days=3, orders_per_day=100, noise=3.0))
    assert noisy.pick_min.std() > a.pick_min.std()


def test_demand_has_peaks():
    cfg = Config(orders_per_day=2000)
    o = sample_day_orders(make_world(cfg), cfg, day=3)
    hourly = np.bincount((o.arrival_min // 60).astype(int), minlength=12)
    assert hourly[10] > 2 * hourly[2]  # 18:00 peak vs 10:00


def test_parquet_and_sql(small):
    cfg, _, _, raw = small
    wh = Warehouse(cfg.data_dir)
    n = wh.query("SELECT COUNT(*) AS n FROM orders").n[0]
    assert n == len(raw) > 1000
    by_hour = wh.demand_by_hour()
    assert by_hour.orders_per_day.sum() > 0
    assert (cfg.data_dir / "orders.parquet").exists()


def test_train_serve_feature_parity(small):
    """SQL-join training features must equal the serving path's in-memory lookup features."""
    _, world, _, raw = small
    sample = raw.sample(300, random_state=1)
    train_feats = engineer(sample[[c for c in sample.columns]])
    serve_raw = serving_frame(world, sample.store_id, sample.shopper_id, sample["items"], sample.batch_size,
                              sample.stop_index, sample.leg_dist_km, sample.start_min, sample.dow.to_numpy())
    serve_feats = engineer(serve_raw)
    assert list(train_feats.columns) == FEATURES == list(serve_feats.columns)
    np.testing.assert_allclose(train_feats.to_numpy(), serve_feats.to_numpy(), rtol=1e-9)


def test_single_row_equals_batch_features(small):
    _, world, _, raw = small
    s = raw.head(20)
    bulk = engineer(s)
    for i in range(len(s)):
        row = engineer(s.iloc[[i]])
        np.testing.assert_allclose(row.to_numpy(), bulk.iloc[[i]].to_numpy())
