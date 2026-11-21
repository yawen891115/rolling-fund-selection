"""Unit tests for the core building blocks, each on a tiny example whose answer is known by hand.

    python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fundsel import data as D  # noqa: E402
from fundsel import factors as F  # noqa: E402
from fundsel.backtest import run_backtest  # noqa: E402
from fundsel.metrics import max_drawdown, perf_stats  # noqa: E402
from fundsel.portfolio import risk_contributions, risk_parity_weights, select_funds  # noqa: E402
from fundsel.walkforward import oos_signal_dates, retrain_blocks  # noqa: E402


def test_risk_parity_equalizes_risk_contributions():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))
    cov = np.cov(A.T)
    w = risk_parity_weights(cov)
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()
    assert np.allclose(risk_contributions(w, cov), 1 / 6, atol=1e-4)


def test_risk_parity_diagonal_is_inverse_vol():
    vols = np.array([0.1, 0.2, 0.4])
    w = risk_parity_weights(np.diag(vols ** 2))
    expect = (1 / vols) / (1 / vols).sum()      # uncorrelated assets: ERC = inverse-volatility weights
    assert np.allclose(w, expect, atol=1e-5)


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])         # NAV 1 -> 1.1 -> 0.55 -> 0.66, max DD = 0.55 / 1.1 - 1
    assert abs(max_drawdown(r) - (-0.5)) < 1e-12


def test_factor_helpers_match_naive_pandas():
    rng = np.random.default_rng(1)
    W = rng.normal(0.0005, 0.01, (252, 4))
    nav = pd.DataFrame(np.vstack([np.ones(4), np.cumprod(1 + W, axis=0)]))
    assert np.allclose(F._mdd(W), (nav / nav.cummax() - 1).min().to_numpy())
    x = rng.normal(0.0004, 0.01, 252)
    f = F.factors_at(W, x, rf=0.0)
    beta_naive = [np.cov(W[:, i], x)[0, 1] / np.var(x, ddof=1) for i in range(4)]
    assert np.allclose(f["beta_12m"], beta_naive, atol=1e-10)
    assert len(f) == 30 and set(f) == set(F.FACTOR_NAMES)


def test_forward_return_timing():
    cal = pd.bdate_range("2020-01-01", periods=60)
    R = pd.DataFrame({"A": 0.01, "B": 0.0}, index=cal)
    R.iloc[22, 0] = 0.50                          # a single +50% day at index 22
    sig = pd.DatetimeIndex([cal[20], cal[40]])
    fwd = D.forward_returns(R, sig, lag=1)       # execution day = 21, holding period (21, 41]
    assert abs(fwd.loc[cal[20], "A"] - (1.5 * 1.01 ** 19 - 1)) < 1e-12
    assert np.isnan(fwd.loc[cal[40], "A"])       # the last signal has no next period


def test_calendar_compounds_weekend_and_preserves_missing_quotes():
    cal = pd.to_datetime(["2024-11-01", "2024-11-04", "2024-11-05"])
    quotes = pd.DataFrame({"code": ["000001", "000001", "000001", "000002", "000003"],
                           "date": pd.to_datetime(["2024-11-02", "2024-11-04", "2024-11-05",
                                                   "2024-11-01", "2024-11-07"]),
                           "ret": [0.02, -0.01, np.nan, 0.0, 0.1]})
    result = D.to_calendar(quotes, cal)
    assert list(result.columns) == ["000001", "000002"]
    np.testing.assert_allclose(result["000001"], [np.nan, 1.02 * 0.99 - 1, np.nan], equal_nan=True)
    np.testing.assert_allclose(result["000002"], [0.0, np.nan, np.nan], equal_nan=True)


def test_no_lookahead_in_retrain_blocks():
    cal = pd.bdate_range("2015-01-01", "2023-01-31")
    me = D.month_ends(cal, "2015-12-31", "2022-12-31")
    sig = oos_signal_dates(me, "2019-01-01", "2022-12-31")
    assert sig[0] == pd.Timestamp("2018-12-31") and sig[-1] == pd.Timestamp("2022-11-30")
    blocks = retrain_blocks(sig, cal, lag=1)
    assert list(blocks) == [2019, 2020, 2021, 2022]
    lend = D.label_end_dates(cal, me, lag=1)
    for year, block in blocks.items():
        usable = me[(me < block[0]) & (lend.reindex(me).to_numpy() <= block[0])]
        assert lend[usable[-1]] <= block[0]       # every training label is known at the cutoff


def test_forward_label_requires_both_execution_days():
    cal = pd.bdate_range("2024-10-01", "2024-12-31")
    sig = D.month_ends(cal, "2024-10-01", "2024-12-31")
    ret = pd.DataFrame({"A": 0.01}, index=cal)
    fwd = D.forward_returns(ret, sig, lag=1)
    exits = D.label_end_dates(cal, sig, lag=1)
    assert fwd.iloc[0].notna().all() and exits.iloc[0] == pd.Timestamp("2024-12-02")
    assert fwd.iloc[1:].isna().all().all() and exits.iloc[1:].isna().all()
    assert sig[-1] not in retrain_blocks(sig, cal, 1)[2024]


def test_backtest_costs_and_timing():
    cal = pd.bdate_range("2021-01-01", periods=10)
    R = pd.DataFrame({"A": 0.01, "B": -0.01}, index=cal)
    sched = {cal[0]: pd.Series({"A": 1.0}), cal[4]: pd.Series({"B": 1.0})}
    bt = run_backtest("t", R, sched, str(cal[0].date()), str(cal[-1].date()), buy_fee=0.001, sell_fee=0.005)
    # day 1 (execution): only the buy fee; days 2-4 earn A's +1%; day 5 earns A's +1%, then switches to B
    assert abs(bt.returns.iloc[0] - (-0.001)) < 1e-12
    assert abs(bt.returns.iloc[4] - (1.01 * (1 - 0.006) - 1)) < 1e-12
    assert abs(bt.returns.iloc[5] - (-0.01)) < 1e-12


def test_select_funds_buffer_rule():
    scores = pd.Series({c: s for c, s in zip("abcdefgh", [8, 7, 6, 5, 4, 3, 2, 1])})
    # holding e (rank 5, inside the buffer of 6) is kept; holding h (rank 8) is sold
    assert select_funds(scores, ["e", "h"], top_k=3, buffer_k=6) == ["e", "a", "b"]


def test_backtest_honors_start_and_empty_schedule():
    cal = pd.bdate_range("2024-01-02", periods=7)
    returns = pd.DataFrame({"A": 0.10}, index=cal)
    schedule = {cal[0]: pd.Series({"A": 1.0}), cal[3]: pd.Series({"A": 1.0})}
    bt = run_backtest("bounded", returns, schedule, cal[3], cal[-1], 0.001, 0.005)
    assert list(bt.trades.date) == [cal[4]]
    assert bt.returns.index[0] == cal[4] and np.isclose(bt.nav.iloc[0], 0.999)
    cash = run_backtest("cash", returns, {}, cal[3], cal[-1], 0.001, 0.005)
    assert cash.returns.index.equals(cal[3:]) and (cash.nav == 1.0).all()


def test_perf_stats_constant_return():
    idx = pd.bdate_range("2020-01-01", periods=252)
    st = perf_stats(pd.Series(0.0004, index=idx), None, rf=0.0)
    assert abs(st["cagr"] - (1.0004 ** 252 - 1)) < 1e-12 and st["max_dd"] == 0
