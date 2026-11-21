"""Daily backtest: hold the scheduled portfolios, charge subscription/redemption fees, track NAV.

Timing (identical to the label definition, so the model is trained on exactly what the backtest earns):
    s  = signal date (month-end close; factors, scores and target weights use data up to s)
    e  = execution date (next trading day): sell old / buy new at day-e NAV
    day e's return still belongs to the old portfolio; the new one earns from e+1 on.
Between rebalances weights drift with returns (no intra-month rebalancing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .portfolio import target_weights


@dataclass
class BacktestResult:
    name: str
    nav: pd.Series                       # daily NAV (starts at 1)
    returns: pd.Series                   # daily return net of costs
    trades: pd.DataFrame                 # per rebalance: date, turnover, cost, number of holdings
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)   # target weights per rebalance


def run_backtest(name: str, R: pd.DataFrame, schedule: dict, start: str, end: str,
                 buy_fee: float, sell_fee: float, lag: int = 1) -> BacktestResult:
    """Start in cash. Apply scheduled trades whose execution falls in the requested window."""
    cal = R.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    exec_map = {}
    for s, w in schedule.items():
        p = cal.searchsorted(s) + lag
        if p < len(cal) and start <= cal[p] <= end:
            exec_map[cal[p]] = w
    first_exec = min(exec_map, default=start)
    days = cal[(cal >= first_exec) & (cal <= end)]

    w = pd.Series(dtype=float)           # current weights (sum to 1; empty = cash)
    nav, navs, rets, trades, hold_rows = 1.0, [], [], [], []
    for d in days:
        # 1) today's return accrues to the portfolio held at the open
        r = R.loc[d, w.index].fillna(0.0) if len(w) else pd.Series(dtype=float)
        port_r = float((w * r).sum()) if len(w) else 0.0
        if len(w):
            w = w * (1.0 + r) / (1.0 + port_r)          # weight drift
        cost = 0.0
        # 2) on an execution day, rebalance at today's NAV
        if d in exec_map:
            tgt = exec_map[d]
            allc = w.index.union(tgt.index)
            old, new = w.reindex(allc, fill_value=0.0), tgt.reindex(allc, fill_value=0.0)
            buy, sell = (new - old).clip(lower=0).sum(), (old - new).clip(lower=0).sum()
            cost = buy * buy_fee + sell * sell_fee
            trades.append({"date": d, "buy_turnover": buy, "sell_turnover": sell,
                           "cost": cost, "n_holdings": int((tgt > 0).sum())})
            hold_rows += [{"date": d, "code": c, "weight": float(v)} for c, v in tgt.items()]
            w = tgt.copy()
        day_ret = (1.0 + port_r) * (1.0 - cost) - 1.0
        nav *= 1.0 + day_ret
        navs.append(nav)
        rets.append(day_ret)

    nav_s = pd.Series(navs, index=days, name=name)
    return BacktestResult(name=name, nav=nav_s, returns=pd.Series(rets, index=days, name=name),
                          trades=pd.DataFrame(trades), holdings=pd.DataFrame(hold_rows))


def build_schedule(scores: pd.DataFrame, R: pd.DataFrame, top_k: int, buffer_k: int,
                   cov_lookback: int, scheme: str = "rp") -> dict:
    """Turn each signal date's scores into target weights (aware of last period's holdings)."""
    schedule, holdings = {}, []
    for s, g in scores.groupby("date"):
        sc = g.set_index("code")["score"].dropna()
        t = R.index.get_loc(s)
        hist = R.iloc[max(0, t - cov_lookback + 1): t + 1]       # data up to the signal date only
        w = target_weights(sc, holdings, hist, top_k, buffer_k, scheme)
        schedule[s] = w
        holdings = list(w.index)
    return schedule


def equal_weight_universe(name: str, R: pd.DataFrame, panel_dates: dict, start: str, end: str,
                          buy_fee: float, sell_fee: float, lag: int = 1) -> BacktestResult:
    """Reference: hold every eligible fund equally, rebalanced monthly (~ the average equity fund)."""
    schedule = {s: pd.Series(1.0 / len(codes), index=codes) for s, codes in panel_dates.items()}
    return run_backtest(name, R, schedule, start, end, buy_fee, sell_fee, lag)


def index_series(name: str, close: pd.Series, start_day: pd.Timestamp, end: str) -> BacktestResult:
    """Wrap an index as a BacktestResult so metrics and charts treat it like a strategy."""
    c = close.loc[(close.index >= start_day) & (close.index <= pd.Timestamp(end))].dropna()
    r = c.pct_change().fillna(0.0)
    return BacktestResult(name=name, nav=c / c.iloc[0], returns=r, trades=pd.DataFrame())
