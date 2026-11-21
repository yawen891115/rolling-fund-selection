"""Walk-forward validation (expanding window, yearly retraining).

Timeline for the model that predicts 2020:

    |<---------------- training window (expanding, starts 2015-12) ---------------->|  |<- test: 2020 ->|
    |<---------- train fold ---------->| gap 1m |<------- validation fold: 12m ------>|  | 12 month-ends   |
                                       embargo                               cutoff = last trading day of 2019

Each year:
  1. fit the reducer (PCA / autoencoder) on the train fold; Optuna picks LightGBM hyper-parameters
     by mean Rank IC on the validation fold
  2. refit reducer + LightGBM with those parameters on train + validation
  3. score the next 12 month-ends -> pure out-of-sample predictions
  4. roll forward one year

Hard rule: a training sample is used only if its label was already known at the cutoff
(label_end <= cutoff); otherwise the model would peek at the future.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import FACTOR_NAMES
from .model import make_lgbm, rank_ic_by_date, tune_lgbm
from .reduce import make_reducer


@dataclass
class WFResult:
    method: str
    scores: pd.DataFrame                     # date, code, score (test period only)
    oos_ic: pd.Series                        # Rank IC per out-of-sample signal date
    log: list = field(default_factory=list)  # one record per retrain
    reducers: dict = field(default_factory=dict)


def oos_signal_dates(signal_dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Signal dates whose holding period falls inside [start, end]: from the last month-end before
    `start` up to the month-end before the month of `end`."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first = signal_dates[signal_dates < start].max()
    sel = signal_dates[(signal_dates >= first) & (signal_dates <= end)]
    return sel[~((sel.year == end.year) & (sel.month == end.month))]


def retrain_blocks(sig: pd.DatetimeIndex, calendar: pd.DatetimeIndex, lag: int) -> dict[int, pd.DatetimeIndex]:
    """Group test signals by the calendar year of their execution day; the first signal of each
    group is that retrain's cutoff."""
    pos = calendar.searchsorted(sig) + lag
    observed = pos < len(calendar)
    sig = sig[observed]
    years = calendar[pos[observed]].year
    return {int(y): sig[years == y] for y in sorted(set(years))}


def run_walk_forward(data: pd.DataFrame, label_end: pd.Series, calendar: pd.DatetimeIndex,
                     cfg, method: str, test_end: str, verbose: bool = True) -> WFResult:
    """data: long table [date, code, 30 standardized factors, fwd_ret, y] with y = next-month percentile."""
    all_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    sig = oos_signal_dates(all_dates, cfg.test_start, test_end)
    blocks = retrain_blocks(sig, calendar, cfg.exec_lag)
    res = WFResult(method=method, scores=pd.DataFrame(), oos_ic=pd.Series(dtype=float))
    score_frames, monthly_ics = [], []

    for year, block in blocks.items():
        t0 = time.time()
        cutoff = block[0]
        train_dates = all_dates[(all_dates < cutoff) & (label_end.reindex(all_dates).to_numpy() <= cutoff)]
        valid_dates = train_dates[-cfg.valid_months:]
        sub_dates = train_dates[: -(cfg.valid_months + cfg.embargo_months)]

        labeled = data["y"].notna()
        sub = data[labeled & data["date"].isin(sub_dates)]
        val = data[labeled & data["date"].isin(valid_dates)]
        n_valid_rows = len(val)

        def X(df):
            return df[FACTOR_NAMES].to_numpy(dtype="float32")

        # ---- 1) fit the reducer on the train fold, tune LightGBM on the validation fold
        red = make_reducer(method, cfg).fit(X(sub), X(val))
        params, valid_ic, trials = tune_lgbm(red.transform(X(sub)), sub["y"].to_numpy(),
                                             red.transform(X(val)), val["y"].to_numpy(),
                                             val["date"].to_numpy(), cfg.lgbm_trials, cfg.seed)
        # ---- 2) refit everything on the full training window with the chosen parameters
        ae_params = getattr(red, "params", None) if method == "ae" else None
        del red, sub, val
        full = data[labeled & data["date"].isin(train_dates)]
        red_full = make_reducer(method, cfg, ae_params=ae_params).fit(X(full))
        Z_full = red_full.transform(X(full))
        model = make_lgbm(params, cfg.seed).fit(Z_full, full["y"].to_numpy())

        # ---- 3) score the next 12 month-ends (out-of-sample)
        test = data[data["date"].isin(block)]
        pred = model.predict(red_full.transform(X(test)))
        score_frames.append(pd.DataFrame({"date": test["date"].to_numpy(),
                                          "code": test["code"].to_numpy(), "score": pred}))
        ic = rank_ic_by_date(test["date"], test["fwd_ret"], pred)   # evaluation only
        ic = ic[label_end.reindex(ic.index) <= pd.Timestamp(test_end)]
        monthly_ics.append(ic)

        entry = {
            "method": method, "test_year": year, "cutoff": str(cutoff.date()),
            "train_period": [str(train_dates[0].date()), str(train_dates[-1].date())],
            "n_train_rows": int(len(full)), "n_valid_rows": n_valid_rows, "n_test_rows": int(len(test)),
            "valid_ic": round(valid_ic, 4), "oos_ic_mean": round(float(ic.mean()), 4),
            "best_params": params, "reducer": red_full.info(X(full)),
            "feature_importance_gain": [round(float(v), 1) for v in
                                        model.booster_.feature_importance("gain")],
            "seconds": round(time.time() - t0, 1),
        }
        res.log.append(entry)
        res.reducers[year] = red_full
        del model, Z_full, full, test
        if verbose:
            extra = entry["reducer"].get("cum_explained", entry["reducer"].get("recon_r2", ""))
            print(f"    [{method}] {year}: train {entry['train_period'][0]}..{entry['train_period'][1]} "
                  f"({entry['n_train_rows']} rows)  valid IC {valid_ic:+.3f}  OOS IC {ic.mean():+.3f}  "
                  f"repr {extra}  {entry['seconds']}s", flush=True)

    res.scores = pd.concat(score_frames, ignore_index=True)
    res.oos_ic = pd.concat(monthly_ics)
    return res
