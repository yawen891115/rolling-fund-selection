"""30 fund factors, computed at every month-end from daily fund returns and benchmark returns only.

Why NAV-only factors? Holdings, fund size or manager tenure must be used in their point-in-time
version (what was published at that date). Public sources only keep the latest few periods, so using
them historically would leak future information. Daily NAVs are public and fully reconstructible.

Implementation: at trading day t take the last-L-days return window W (L x N matrix, N = funds) and
reduce along the time axis, so one numpy expression yields the factor for every fund at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252  # trading days per year

# (name, group, description) -- 30 factors in 5 groups
FACTORS: list[tuple[str, str, str]] = [
    ("ret_1m", "return/momentum", "1-month cumulative return"),
    ("ret_3m", "return/momentum", "3-month cumulative return"),
    ("ret_6m", "return/momentum", "6-month cumulative return"),
    ("ret_12m", "return/momentum", "12-month cumulative return"),
    ("mom_12_1", "return/momentum", "12-1 momentum: past 12 months excluding the latest month"),
    ("ret_24m", "return/momentum", "24-month cumulative return"),
    ("exret_6m", "return/momentum", "6-month return in excess of the benchmark"),
    ("vol_3m", "risk", "3-month annualized volatility"),
    ("vol_12m", "risk", "12-month annualized volatility"),
    ("downvol_12m", "risk", "12-month annualized downside deviation"),
    ("mdd_6m", "risk", "6-month maximum drawdown (negative)"),
    ("mdd_12m", "risk", "12-month maximum drawdown (negative)"),
    ("cvar_12m", "risk", "12-month CVaR: mean of the worst 5% daily returns"),
    ("te_12m", "risk", "12-month tracking error vs the benchmark"),
    ("sharpe_6m", "risk-adjusted", "6-month Sharpe ratio"),
    ("sharpe_12m", "risk-adjusted", "12-month Sharpe ratio"),
    ("sortino_12m", "risk-adjusted", "12-month Sortino ratio"),
    ("calmar_12m", "risk-adjusted", "12-month Calmar ratio: annualized return / |max drawdown|"),
    ("ir_12m", "risk-adjusted", "12-month information ratio: active return / tracking error"),
    ("alpha_12m", "attribution/timing", "CAPM annualized alpha"),
    ("beta_12m", "attribution/timing", "CAPM beta"),
    ("r2_12m", "attribution/timing", "CAPM R^2 (lower = more active)"),
    ("resid_vol_12m", "attribution/timing", "idiosyncratic volatility (CAPM residual)"),
    ("tm_gamma_12m", "attribution/timing", "Treynor-Mazuy market-timing coefficient gamma"),
    ("up_cap_12m", "attribution/timing", "upside capture ratio"),
    ("down_cap_12m", "attribution/timing", "downside capture ratio"),
    ("winrate_12m", "consistency", "share of days beating the benchmark"),
    ("skew_12m", "consistency", "skewness of daily returns"),
    ("kurt_12m", "consistency", "excess kurtosis of daily returns"),
    ("dist_high_12m", "consistency", "distance of current NAV to its 12-month high (<= 0)"),
]
FACTOR_NAMES = [f[0] for f in FACTORS]


# ---------------------------------------------------------------------------
# building blocks (W: L x N returns with missing filled by 0; x: length-L benchmark returns)
# ---------------------------------------------------------------------------
def _cum(W):
    return np.prod(1.0 + W, axis=0) - 1.0


def _ann_ret(W):
    return (1.0 + _cum(W)) ** (ANN / W.shape[0]) - 1.0


def _ann_vol(W):
    return W.std(axis=0, ddof=1) * np.sqrt(ANN)


def _downvol(W):
    return np.sqrt((np.minimum(W, 0.0) ** 2).mean(axis=0) * ANN)


def _nav(W):
    """NAV path starting at 1.0, shape (L+1) x N."""
    return np.vstack([np.ones((1, W.shape[1])), np.cumprod(1.0 + W, axis=0)])


def _mdd(W):
    nav = _nav(W)
    return (nav / np.maximum.accumulate(nav, axis=0) - 1.0).min(axis=0)


def _cvar(W, q=0.05):
    k = max(1, int(np.ceil(q * W.shape[0])))
    return np.sort(W, axis=0)[:k].mean(axis=0)


def _ols(X, Y):
    """All funds share the same design matrix X (L x p): one lstsq call solves N regressions (p x N)."""
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef


def _safe(a):
    a = np.asarray(a, dtype="float64")
    a[~np.isfinite(a)] = np.nan
    return a


# ---------------------------------------------------------------------------
# the 30 factors on one date
# ---------------------------------------------------------------------------
def factors_at(Rw: np.ndarray, bw: np.ndarray, rf: float) -> dict[str, np.ndarray]:
    """Rw: (look-back days x N) returns whose last row is today (NaN = no NAV); bw: benchmark returns.
    Needs at least 252 rows; ret_24m is only computed when 504 rows are available.
    """

    def win(L):
        W = Rw[-L:]
        valid = np.isfinite(W).sum(axis=0) >= 0.9 * L      # >= 90% of days must have data
        return np.nan_to_num(W), np.nan_to_num(bw[-L:]), valid

    rf_d = rf / ANN
    W1, _, _ = win(21)
    W3, _, _ = win(63)
    W6, x6, _ = win(126)
    W, x, _ = win(252)
    f: dict[str, np.ndarray] = {}

    # ---- return / momentum
    f["ret_1m"], f["ret_3m"], f["ret_6m"], f["ret_12m"] = _cum(W1), _cum(W3), _cum(W6), _cum(W)
    f["mom_12_1"] = _cum(W[:-21])
    if Rw.shape[0] >= 504:
        W24, _, ok24 = win(504)
        f["ret_24m"] = np.where(ok24, _cum(W24), np.nan)
    else:
        f["ret_24m"] = np.full(Rw.shape[1], np.nan)
    f["exret_6m"] = _cum(W6) - (np.prod(1.0 + x6) - 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # ---- risk
        ann_ret12, vol12 = _ann_ret(W), _ann_vol(W)
        active = W - x[:, None]
        te = active.std(axis=0, ddof=1) * np.sqrt(ANN)
        f["vol_3m"], f["vol_12m"] = _ann_vol(W3), vol12
        f["downvol_12m"] = _downvol(W)
        f["mdd_6m"], f["mdd_12m"] = _mdd(W6), _mdd(W)
        f["cvar_12m"] = _cvar(W)
        f["te_12m"] = te

        # ---- risk-adjusted return
        f["sharpe_6m"] = (_ann_ret(W6) - rf) / _ann_vol(W6)
        f["sharpe_12m"] = (ann_ret12 - rf) / vol12
        f["sortino_12m"] = (ann_ret12 - rf) / f["downvol_12m"]
        f["calmar_12m"] = ann_ret12 / np.abs(f["mdd_12m"])
        f["ir_12m"] = active.mean(axis=0) * ANN / te

        # ---- attribution & timing:  CAPM  y = a + b*x + e ;  Treynor-Mazuy  y = a + b*x + g*x^2 + e
        y, xe = W - rf_d, x - rf_d
        X1 = np.column_stack([np.ones_like(xe), xe])
        c1 = _ols(X1, y)
        resid = y - X1 @ c1
        sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        f["alpha_12m"] = c1[0] * ANN
        f["beta_12m"] = c1[1]
        f["r2_12m"] = 1.0 - (resid ** 2).sum(axis=0) / sst
        f["resid_vol_12m"] = resid.std(axis=0, ddof=2) * np.sqrt(ANN)
        X2 = np.column_stack([np.ones_like(xe), xe, xe ** 2])
        f["tm_gamma_12m"] = _ols(X2, y)[2]
        up, dn = x > 0, x < 0
        f["up_cap_12m"] = W[up].mean(axis=0) / x[up].mean()
        f["down_cap_12m"] = W[dn].mean(axis=0) / x[dn].mean()

        # ---- consistency
        f["winrate_12m"] = (W > x[:, None]).mean(axis=0)
        f["skew_12m"] = stats.skew(W, axis=0, bias=False)
        f["kurt_12m"] = stats.kurtosis(W, axis=0, fisher=True, bias=False)
        nav = _nav(W)
        f["dist_high_12m"] = nav[-1] / nav.max(axis=0) - 1.0

    return {k: _safe(v) for k, v in f.items()}


# ---------------------------------------------------------------------------
# factor panel over all signal dates
# ---------------------------------------------------------------------------
def build_factor_panel(R: pd.DataFrame, bench: pd.Series, signal_dates: pd.DatetimeIndex,
                       rf: float, min_history: int = 252) -> pd.DataFrame:
    """Long table: one row per (signal date, fund), 30 factor columns.
    Only funds eligible on that date are kept: >= 90% of the last year has a NAV (a full year of
    track record) and a NAV was published within the last 5 days (not liquidated).
    """
    Rv, bv = R.to_numpy(dtype="float64"), bench.to_numpy(dtype="float64")
    codes = R.columns.to_numpy()
    rows = []
    for d in signal_dates:
        t = R.index.get_loc(d)
        if t + 1 < min_history:
            continue
        hist = np.isfinite(Rv[t - min_history + 1: t + 1]).sum(axis=0) >= 0.9 * min_history
        alive = np.isfinite(Rv[t - 4: t + 1]).any(axis=0)
        ok = hist & alive
        if ok.sum() < 30:
            continue
        lo = max(0, t - 504 + 1)                              # longest look-back: 24 months
        f = factors_at(Rv[lo: t + 1][:, ok], bv[lo: t + 1], rf)
        block = pd.DataFrame(f)[FACTOR_NAMES]
        block.insert(0, "code", codes[ok])
        block.insert(0, "date", d)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def cross_sectional_standardize(panel: pd.DataFrame, cols: list[str], q: float) -> pd.DataFrame:
    """Within each date: winsorize at q / 1-q -> z-score -> fill missing with 0 (= the cross-sectional mean).
    Uses only same-day information, so no look-ahead.
    """
    out = panel.copy()

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        lo, hi = g.quantile(q), g.quantile(1 - q)
        g = g.clip(lo, hi, axis=1)
        return (g - g.mean()) / g.std(ddof=0).replace(0, np.nan)

    out[cols] = panel.groupby("date", group_keys=False)[cols].apply(_z).fillna(0.0)
    return out
