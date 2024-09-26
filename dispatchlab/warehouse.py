"""DuckDB access layer: SQL over the Parquet files."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class Warehouse:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.con = duckdb.connect(":memory:")
        for t in ("orders", "stores", "shoppers"):
            p = (self.dir / f"{t}.parquet").as_posix()
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{p}')")

    def query(self, sql: str, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def write_store_stats(self, train_days: int) -> None:
        """Per-store historical minutes-per-item, from the training days only."""
        out = (self.dir / "store_stats.parquet").as_posix()
        self.con.execute(
            f"""COPY (
                SELECT store_id, AVG(pick_min * batch_size / items) AS store_hist_min_per_item
                FROM orders WHERE day < {int(train_days)} GROUP BY store_id ORDER BY store_id
            ) TO '{out}' (FORMAT PARQUET)"""
        )

    def raw_training_frame(self, day_lo: int = 0, day_hi: int = 10**9) -> pd.DataFrame:
        """Orders joined with store/shopper reference data and store stats (raw features + targets)."""
        stats = (self.dir / "store_stats.parquet").as_posix()
        return self.query(
            f"""
            SELECT o.order_id, o.day, o.dow, o.items, o.batch_size, o.stop_index, o.leg_dist_km,
                   o.start_min, s.store_size_sqft, ss.store_hist_min_per_item,
                   h.shopper_rating, h.shopper_tenure_days, h.shopper_vehicle_kmh,
                   o.store_id, o.shopper_id, o.pick_min, o.drive_min
            FROM orders o
            JOIN stores s USING (store_id)
            JOIN shoppers h USING (shopper_id)
            JOIN read_parquet('{stats}') ss USING (store_id)
            WHERE o.day >= ? AND o.day < ?
            ORDER BY o.order_id""",
            [day_lo, day_hi],
        )

    def demand_by_hour(self) -> pd.DataFrame:
        return self.query(
            """SELECT CAST(8 + start_min / 60 AS INTEGER) AS hour, COUNT(*) / COUNT(DISTINCT day) AS orders_per_day,
                      AVG(pick_min) AS avg_pick_min, AVG(drive_min) AS avg_drive_min
               FROM orders GROUP BY 1 ORDER BY 1"""
        )
