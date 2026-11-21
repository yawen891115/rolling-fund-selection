"""Data layer: raw files -> a clean daily-return matrix.

Everything downstream depends only on what this module produces:
    R          DataFrame  rows = trading days, columns = fund codes, values = daily return (0.01 = 1%)
    bench      Series     daily return of the main benchmark on the same calendar
    funds      DataFrame  fund metadata (code, name, type, eligibility flag)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import Config


@dataclass
class MarketData:
    R: pd.DataFrame
    bench: pd.Series
    bench_close: pd.DataFrame
    funds: pd.DataFrame
    calendar: pd.DatetimeIndex
    report: dict


# ---------------------------------------------------------------------------
# 1. raw files
# ---------------------------------------------------------------------------
def _num(s: pd.Series, scale: float) -> np.ndarray:
    """The downloader stores NAVs as scaled integers to keep the file small; undo that (NA -> NaN)."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan) / scale


def load_nav(raw_dir: str | Path, codes=None) -> pd.DataFrame:
    columns = ["code", "date", "unit_nav_e4", "acc_nav_e4", "ret_pct_e2"]
    filters = [("code", "in", codes)] if codes is not None else None
    df = pd.read_parquet(Path(raw_dir) / "fund_nav.parquet", columns=columns, filters=filters,
                         engine="pyarrow", use_threads=False, pre_buffer=False)
    codes = df["code"].astype("category").map(lambda code: str(code).zfill(6)).astype("category")
    dates = df["date"].astype("category")
    decoded = pd.to_datetime(dates.cat.categories.astype(str), format="%Y%m%d")
    return pd.DataFrame({
        "code": codes,
        "date": dates.cat.rename_categories(decoded).astype("datetime64[ns]"),
        "unit_nav": _num(df["unit_nav_e4"], 1e4),
        "acc_nav": _num(df["acc_nav_e4"], 1e4),
        "ret_pct": _num(df["ret_pct_e2"], 1e2),      # Eastmoney "daily growth rate", in percent
    })


def load_funds(raw_dir: str | Path) -> pd.DataFrame:
    f = pd.read_csv(Path(raw_dir) / "fund_list.csv", dtype={"code": str})
    f["code"] = f["code"].str.zfill(6)
    return f


def load_benchmarks(raw_dir: str | Path) -> pd.DataFrame:
    p = Path(raw_dir) / "benchmarks.csv"
    try:
        b = pd.read_csv(p, dtype={"index_code": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["date", "close", "index_code"])
    b["date"] = pd.to_datetime(b["date"])
    return b


# ---------------------------------------------------------------------------
# 2. NAV -> daily returns
# ---------------------------------------------------------------------------
def clean_returns(nav: pd.DataFrame, max_abs: float) -> tuple[pd.DataFrame, dict]:
    """Daily return = Eastmoney's daily growth rate, which already reinvests dividends.
    Where it is missing, back it out of the accumulated NAV:
        r_t = (acc_nav_t - acc_nav_{t-1}) / unit_nav_{t-1}
    A cash dividend d lowers the unit NAV by d but leaves the accumulated NAV unchanged,
    so this formula is dividend-correct by construction (a plain unit-NAV pct_change is not).
    """
    nav = nav.sort_values(["code", "date"], ignore_index=True)
    r = nav["ret_pct"] / 100.0
    filled = 0
    if r.isna().any():
        g = nav.groupby("code", observed=True)
        r_acc = (nav["acc_nav"] - g["acc_nav"].shift()) / g["unit_nav"].shift()
        filled = int((r.isna() & r_acc.notna()).sum())
        r = r.fillna(r_acc)

    bad = r.abs() > max_abs                     # obvious data errors (e.g. an unadjusted share split)
    r = r.mask(bad)
    stats = {"rows": int(len(r)), "filled_from_acc_nav": filled,
             "dropped_abs_gt_max": int(bad.sum()), "missing_after_clean": int(r.isna().sum())}
    return pd.DataFrame({"code": nav["code"], "date": nav["date"], "ret": r}), stats


def to_calendar(ret_long: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Long (fund, date, ret) -> wide (trading day x fund).
    A few funds (e.g. those holding HK stocks via Stock Connect) publish NAVs on days the A-share
    market is closed; those returns are compounded into the next trading day so cumulative
    performance is preserved while every series sits on one calendar with the benchmark.
    """
    pos = calendar.searchsorted(ret_long["date"].to_numpy(), side="left")
    keep = pos < len(calendar)
    if not keep.all():
        ret_long, pos = ret_long.loc[keep], pos[keep]
    fund, codes = pd.factorize(ret_long["code"], sort=True)
    returns = ret_long["ret"].to_numpy(dtype=float)
    valid = (fund >= 0) & np.isfinite(returns)
    cells = (pos[valid], fund[valid])
    totals = np.zeros((len(calendar), len(codes)), dtype=float)
    observed = np.zeros(totals.shape, dtype=bool)
    np.add.at(totals, cells, np.log1p(returns[valid]))
    observed[cells] = True
    np.expm1(totals, out=totals)
    totals[~observed] = np.nan
    return pd.DataFrame(totals, index=calendar, columns=pd.Index(codes.astype(str), name="code"))


# ---------------------------------------------------------------------------
# 3. static universe filter: fund type + name
# ---------------------------------------------------------------------------
def tag_universe(funds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = funds.copy()
    type_ok = f["type"].isin(cfg.universe_types)
    name_ok = ~f["name"].astype(str).str.contains(cfg.exclude_name_pattern, regex=True)
    f["eligible"] = type_ok & name_ok
    f["exclude_reason"] = np.where(~type_ok, "type", np.where(~name_ok, "lock-up / periodic open", ""))
    return f


# ---------------------------------------------------------------------------
# 4. assemble
# ---------------------------------------------------------------------------
def load_market_data(cfg: Config) -> MarketData:
    raw = Path(cfg.raw_dir)
    source = ds.dataset(raw / "fund_nav.parquet", format="parquet")
    fund_codes = set()
    for batch in source.to_batches(columns=["code"], batch_size=65536, use_threads=False):
        fund_codes.update(batch.column(0).unique().to_pylist())
    if not fund_codes or None in fund_codes:
        raise ValueError("The NAV snapshot requires at least one fund and a code for every row.")
    codes = sorted(fund_codes)
    # Keep each fund's complete history in one batch so a missing growth rate can use its previous NAV.
    groups = [codes[i:i + 128] for i in range(0, len(codes), 128)]
    funds = tag_universe(load_funds(raw), cfg)
    bclose = load_benchmarks(raw)

    # Trading calendar: every date on which an index prints, minus weekends and Jan 1. CSIndex's
    # date-range queries add a spurious Jan-1 row carrying the next session's close; Jan 1 and weekends
    # are never A-share trading days. (Funds publish half-year-end NAVs on weekends too; to_calendar
    # folds those into the next trading day.)
    main = bclose[bclose["index_code"] == cfg.benchmark].sort_values("date")
    if len(main) > 250:
        cal = pd.DatetimeIndex(sorted(set(bclose["date"])))
        calendar = cal[(cal.weekday < 5) & ~((cal.month == 1) & (cal.day == 1))]
    else:  # no benchmark file: use days on which at least half of the funds report a NAV
        counts = []
        for group in groups:
            pairs = source.to_table(columns=["code", "date"], filter=ds.field("code").isin(group),
                                    use_threads=False).to_pandas(categories=["code"])
            counts.append(pairs.groupby("date")["code"].nunique())
        cnt = pd.concat(counts, axis=1).sum(axis=1).sort_index()
        cnt.index = pd.to_datetime(cnt.index.astype(str), format="%Y%m%d")
        calendar = pd.DatetimeIndex(cnt[cnt >= 0.5 * cnt.rolling(20, min_periods=1).max()].index).sort_values()

    eligible_codes = set(funds.loc[funds["eligible"], "code"])
    parts, stats = [], {}
    for group in groups:
        ret_long, batch_stats = clean_returns(load_nav(raw, group), cfg.max_daily_abs_ret)
        for key, value in batch_stats.items():
            stats[key] = stats.get(key, 0) + value
        selected = ret_long.loc[ret_long["code"].isin(eligible_codes)]
        if not selected.empty:
            parts.append(to_calendar(selected, calendar))
    if not parts:
        raise ValueError("The NAV snapshot contains no observations for eligible funds.")
    R = pd.concat(parts, axis=1).sort_index(axis=1)

    if len(main) > 250:   # a few sessions are missing for 930950: carry the close forward
        bench = main.set_index("date")["close"].reindex(calendar).ffill().pct_change()
        bench_src = f"csindex:{cfg.benchmark}"
    else:  # fallback: equal-weighted average of the universe (what equity-fund indices approximate)
        bench = R.mean(axis=1, skipna=True)
        bench_src = "fallback: universe equal-weight"

    bench_close = (bclose.pivot_table(index="date", columns="index_code", values="close")
                   .reindex(calendar).ffill() if len(bclose) else pd.DataFrame(index=calendar))
    report = {
        "nav_rows": stats["rows"], "funds_in_file": len(codes),
        "funds_eligible": int(len(eligible_codes)), "funds_in_matrix": int(R.shape[1]),
        "calendar": [str(calendar[0].date()), str(calendar[-1].date()), int(len(calendar))],
        "benchmark_source": bench_src, **stats,
    }
    return MarketData(R=R, bench=bench.rename("bench"), bench_close=bench_close,
                      funds=funds, calendar=calendar, report=report)


# ---------------------------------------------------------------------------
# 5. time helpers: signal dates, execution dates, forward returns (labels)
# ---------------------------------------------------------------------------
def month_ends(calendar: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Last trading day of each month = signal date."""
    s = pd.Series(calendar, index=calendar)
    me = pd.DatetimeIndex(s.groupby([calendar.year, calendar.month]).max().to_numpy())
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def forward_returns(R: pd.DataFrame, signal_dates: pd.DatetimeIndex, lag: int) -> pd.DataFrame:
    """Label: buy at the NAV of the execution day e (= `lag` trading days after signal s),
    sell at the next execution day e'. Holding-period return = prod_{d in (e, e']} (1 + r_d) - 1.
    Day e itself belongs to the *previous* portfolio. The last signal date has no label (NaN row).
    """
    cal = R.index
    pos = cal.searchsorted(signal_dates) + lag
    vals = R.to_numpy()
    out = np.full((len(signal_dates), R.shape[1]), np.nan)
    for k in range(len(signal_dates) - 1):
        if pos[k + 1] >= len(cal):
            break  # the sale has not executed within the available data
        win = vals[pos[k] + 1: pos[k + 1] + 1]
        has = np.isfinite(win).any(axis=0)
        out[k] = np.where(has, np.prod(1.0 + np.nan_to_num(win), axis=0) - 1.0, np.nan)
    return pd.DataFrame(out, index=signal_dates, columns=R.columns)


def label_end_dates(calendar: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex, lag: int) -> pd.Series:
    """The day on which each signal date's label becomes known (= the next execution day).
    A training sample may only be used once its label is known."""
    pos = calendar.searchsorted(signal_dates) + lag
    observed = pos < len(calendar)
    execution = pd.Series(pd.NaT, index=signal_dates, dtype=calendar.dtype)
    execution.iloc[np.flatnonzero(observed)] = calendar[pos[observed]]
    return execution.shift(-1)
