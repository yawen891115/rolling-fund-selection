"""Performance metrics (all from daily return series).

    CAGR              (end NAV / start NAV)^(252 / trading days) - 1
    Volatility        std(daily returns) * sqrt(252)
    Sharpe            (mean daily return - rf/252) / std(daily returns) * sqrt(252)
    Max drawdown      min(NAV / running max NAV - 1)
    Calmar            CAGR / |max drawdown|
    Excess (pts)      strategy CAGR - benchmark CAGR, in percentage points
    Information ratio mean daily active return / its std * sqrt(252)
    Monthly hit rate  share of months in which the strategy beats the benchmark
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def max_drawdown(ret: pd.Series) -> float:
    nav = np.concatenate([[1.0], (1.0 + ret).cumprod().to_numpy()])
    return float((nav / np.maximum.accumulate(nav) - 1.0).min())


def perf_stats(ret: pd.Series, bench: pd.Series | None = None, rf: float = 0.015) -> dict:
    ret = ret.dropna()
    n = len(ret)
    cagr = float((1.0 + ret).prod() ** (ANN / n) - 1.0)
    vol = float(ret.std(ddof=1) * np.sqrt(ANN))
    sharpe = float((ret.mean() - rf / ANN) / ret.std(ddof=1) * np.sqrt(ANN))
    mdd = max_drawdown(ret)
    out = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": mdd,
           "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "days": n,
           "period": f"{ret.index[0].date()} ~ {ret.index[-1].date()}"}
    if bench is not None:
        b = bench.reindex(ret.index).fillna(0.0)
        b_cagr = float((1.0 + b).prod() ** (ANN / n) - 1.0)
        active = ret - b
        te = float(active.std(ddof=1) * np.sqrt(ANN))
        m_s = (1.0 + ret).groupby(ret.index.to_period("M")).prod()
        m_b = (1.0 + b).groupby(b.index.to_period("M")).prod()
        is_bench = te < 1e-10                       # the benchmark compared with itself
        out.update({
            "bench_cagr": b_cagr,
            "excess_pts": (cagr - b_cagr) * 100,
            "tracking_error": te,
            "info_ratio": np.nan if is_bench else float(active.mean() * ANN / te),
            "monthly_hit_rate": np.nan if is_bench else float((m_s > m_b).mean()),
        })
    return out


def yearly_returns(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).groupby(ret.index.year).prod() - 1.0


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    return {"ic_mean": float(ic.mean()), "ic_std": float(ic.std()),
            "icir_annual": float(ic.mean() / ic.std() * np.sqrt(12)) if ic.std() > 0 else np.nan,
            "ic_positive_share": float((ic > 0).mean()), "months": int(len(ic))}
