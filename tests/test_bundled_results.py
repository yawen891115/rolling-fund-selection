"""Replay the reference portfolios from the checked-in data and monthly holdings."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundsel.backtest import run_backtest
from fundsel.config import Config
from fundsel.data import load_market_data


def test_bundled_holdings_reproduce_nav_and_fees():
    cfg = Config(raw_dir=str(ROOT / "data" / "raw"))
    market = load_market_data(cfg)
    results = ROOT / "results"
    nav = pd.read_csv(results / "nav_daily.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv(results / "metrics.csv").set_index(["window", "strategy"])
    for model in ("PCA-6", "AE-6"):
        strategy = f"{model} + RP"
        holdings = pd.read_csv(results / f"holdings_{model}.csv", dtype={"code": str}, parse_dates=["date"])
        schedule = {}
        for execution, positions in holdings.groupby("date"):
            signal = market.calendar[market.calendar.get_loc(execution) - cfg.exec_lag]
            schedule[signal] = positions.set_index("code")["weight"]
        replay = run_backtest(strategy, market.R, schedule, cfg.test_start, cfg.holdout_end,
                              cfg.buy_fee, cfg.sell_fee, cfg.exec_lag)
        assert replay.nav.index.equals(nav.index)
        np.testing.assert_allclose(replay.nav, nav[strategy], rtol=1e-10, atol=1e-10)
        main_fees = replay.trades.loc[replay.trades.date <= pd.Timestamp(cfg.test_end), "cost"].sum()
        np.testing.assert_allclose(main_fees, metrics.loc[("2019-2022", strategy), "total_costs"])
