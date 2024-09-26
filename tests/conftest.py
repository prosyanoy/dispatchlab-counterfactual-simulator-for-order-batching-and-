import pytest

from dispatchlab.config import Config
from dispatchlab.datagen import generate
from dispatchlab.model import TimeModel
from dispatchlab.warehouse import Warehouse


@pytest.fixture(scope="session")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    cfg = Config(data_dir=d, output_dir=d / "out", history_days=14, orders_per_day=120, n_shoppers=10,
                 eval_days=2, ope_log_days=2, ope_replicates=2, ope_truth_days=3, bootstrap=50)
    world = generate(cfg)
    raw = Warehouse(d).raw_training_frame()
    model = TimeModel(0).fit(raw[raw.day < 11])
    return cfg, world, model, raw
