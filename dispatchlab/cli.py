"""Command line: python -m dispatchlab {all,generate,train,serve}."""
from __future__ import annotations

import argparse

from .config import Config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dispatchlab", description=__doc__)
    ap.add_argument("command", choices=["all", "generate", "train", "serve"], default="all", nargs="?")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    cfg = Config.from_env()
    if args.command == "generate":
        from .datagen import generate

        generate(cfg)
        print(f"Wrote Parquet files to {cfg.data_dir}")
    elif args.command == "train":
        from .experiments import prepare

        _, _, ev = prepare(cfg)
        for name, m in ev.items():
            print(f"{name}: MAE={m['mae']:.2f} p90_err={m['p90_abs_error']:.2f} baseline_MAE={m['baseline_mae']:.2f}")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("dispatchlab.api:app", host=args.host, port=args.port)
    else:
        from .report import run_all

        print(f"Report written to {run_all(cfg)}")
