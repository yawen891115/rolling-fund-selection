"""Sensitivity analysis: keep the model scores fixed, vary only the portfolio settings.

The headline result uses settings fixed in advance (first row). This table exists to check
robustness, never to pick the best-looking row after the fact.
"""
from __future__ import annotations

import pandas as pd

from .backtest import build_schedule, run_backtest
from .metrics import perf_stats


def run_sensitivity(scores: pd.DataFrame, R: pd.DataFrame, bench_ret: pd.Series, cfg, end: str) -> pd.DataFrame:
    variants = [
        ("baseline: K=10, buffer 20, monthly", dict(top_k=10, buffer_k=20), "M", 1.0),
        ("K=5", dict(top_k=5, buffer_k=10), "M", 1.0),
        ("K=20", dict(top_k=20, buffer_k=40), "M", 1.0),
        ("no buffer", dict(top_k=10, buffer_k=10), "M", 1.0),
        ("quarterly rebalance", dict(top_k=10, buffer_k=20), "Q", 1.0),
        ("zero fees", dict(top_k=10, buffer_k=20), "M", 0.0),
        ("double fees", dict(top_k=10, buffer_k=20), "M", 2.0),
    ]
    rows = []
    for name, kw, freq, fee_mult in variants:
        sc = scores
        if freq == "Q":
            sc = scores[pd.DatetimeIndex(scores["date"]).month.isin([3, 6, 9, 12])]
        sched = build_schedule(sc, R, cov_lookback=cfg.cov_lookback, **kw)
        bt = run_backtest(name, R, sched, cfg.test_start, end, cfg.buy_fee * fee_mult,
                          cfg.sell_fee * fee_mult, cfg.exec_lag)
        st = perf_stats(bt.returns, bench_ret, cfg.rf)
        rows.append({"setting": name, "cagr": st["cagr"], "excess_pts": st["excess_pts"],
                     "sharpe": st["sharpe"], "max_dd": st["max_dd"],
                     "turnover_per_rebalance": float(bt.trades["sell_turnover"].iloc[1:].mean()),
                     "total_costs": float(bt.trades["cost"].sum())})
    return pd.DataFrame(rows).set_index("setting")
