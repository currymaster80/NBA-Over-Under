#!/usr/bin/env python
"""Command-line entry point.

Examples
--------
    python scripts/run_pipeline.py --quick          # ~2 minute smoke run / demo
    python scripts/run_pipeline.py                  # full run (hyper-parameter search, 5 seeds)
    python scripts/run_pipeline.py --models mlp,cf  # only some models
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nbaou.pipeline import ALL_MODELS, Config, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"), help="folder with final_game_data/ and odds_data_processed/")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results"), help="where results are written")
    p.add_argument("--quick", action="store_true", help="tiny grids and fewer seeds (demo mode)")
    p.add_argument("--seeds", type=int, default=None, help="random seeds for the neural models (default 5, 2 with --quick)")
    p.add_argument("--models", default=",".join(ALL_MODELS), help=f"comma list from {ALL_MODELS}")
    p.add_argument("--train-seasons", default="2012-2013,2013-2014,2014-2015,2015-2016")
    p.add_argument("--val-season", default="2016-2017")
    p.add_argument("--test-season", default="2017-2018")
    p.add_argument("--history", type=int, default=3, help="number of previous games per team")
    args = p.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    bad = set(models) - set(ALL_MODELS)
    if bad:
        p.error(f"unknown model(s): {sorted(bad)}")

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        train_seasons=tuple(s.strip() for s in args.train_seasons.split(",")),
        val_season=args.val_season,
        test_season=args.test_season,
        n_history=args.history,
        seeds=args.seeds if args.seeds is not None else (2 if args.quick else 5),
        quick=args.quick,
        models=models,
        max_epochs=60 if args.quick else 150,
    )
    table = run(cfg)
    with open(os.path.join(cfg.out_dir, "results_table.md")) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
