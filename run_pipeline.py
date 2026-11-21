#!/usr/bin/env python3
"""End-to-end run.

    python run_pipeline.py            # full run: walk-forward through 2024, reported on two windows
    python run_pipeline.py --quick    # few Optuna trials, a few minutes: environment check

Windows (all out-of-sample):
    2019-2022   main evaluation window
    2023-2024   later evaluation window with the same annual retraining rule
    2019-2024   both together
The backtest is causal, so the 2019-2022 slice of one continuous run is identical to a run that stops
at the end of 2022.

data -> 30 factors -> labels -> walk-forward (reducer + Optuna-tuned LightGBM) -> top-K + risk parity
-> daily backtest -> metrics / charts.  Everything is written to results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fundsel import backtest as BT  # noqa: E402
from fundsel import data as D  # noqa: E402
from fundsel import factors as F  # noqa: E402
from fundsel import metrics as M  # noqa: E402
from fundsel import plots as P  # noqa: E402
from fundsel import walkforward as WF  # noqa: E402
from fundsel.config import Config  # noqa: E402
from fundsel.model import rank_ic_by_date  # noqa: E402
from fundsel.sensitivity import run_sensitivity  # noqa: E402

NAMES = {"pca": "PCA-6 + RP", "ae": "AE-6 + RP", "raw": "Raw-30 + RP"}
ORDER = ["PCA-6 + RP", "AE-6 + RP", "Raw-30 + RP", "Sharpe-12m + RP", "PCA-6 + EW", "Universe EW",
         "CSI Equity Fund Index", "Equal-weight return proxy", "CSI 300 TR"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few Optuna trials, for a fast smoke run")
    ap.add_argument("--methods", nargs="+", default=["pca", "ae", "raw"], choices=["pca", "ae", "raw"])
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.methods = tuple(args.methods)
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    cfg.out_dir = args.out_dir
    if args.quick:
        cfg.lgbm_trials, cfg.ae_trials, cfg.ae_epochs = 5, 2, 60
    end = cfg.holdout_end
    windows = {"2019-2022": (cfg.test_start, cfg.test_end),
               "2023-2024 hold-out": ((pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end),
               "2019-2024": (cfg.test_start, end)}
    out = Path(cfg.out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    t_start = time.time()

    # ------------------------------------------------------------------ 1. data
    print("[1/6] loading data ...", flush=True)
    md = D.load_market_data(cfg)
    print("      ", json.dumps(md.report, ensure_ascii=False))

    # ------------------------------------------------------------------ 2. factors
    print("[2/6] computing 30 factors at every month-end ...", flush=True)
    signal_dates = D.month_ends(md.calendar, cfg.first_signal, end)
    panel_raw = F.build_factor_panel(md.R, md.bench, signal_dates, cfg.rf, cfg.min_history_days)
    panel_std = F.cross_sectional_standardize(panel_raw, F.FACTOR_NAMES, cfg.winsor_q)
    panel_std = panel_std.astype({name: "float32" for name in F.FACTOR_NAMES})
    n_per_date = panel_raw.groupby("date").size()
    print(f"       {panel_raw['date'].nunique()} month-ends, {n_per_date.min()}-{n_per_date.max()} eligible funds "
          f"per date, {len(panel_raw):,} rows")

    # ------------------------------------------------------------------ 3. labels
    print("[3/6] labels: next-month return percentile ...", flush=True)
    fwd = D.forward_returns(md.R, signal_dates, cfg.exec_lag)
    fwd_long = fwd.stack().rename("fwd_ret").reset_index()
    fwd_long.columns = ["date", "code", "fwd_ret"]
    data = panel_std.merge(fwd_long, on=["date", "code"], how="left")
    del panel_std, fwd
    data["y"] = data.groupby("date")["fwd_ret"].rank(pct=True)
    label_end = D.label_end_dates(md.calendar, signal_dates, cfg.exec_lag)

    # ------------------------------------------------------------------ 4. walk-forward
    print("[4/6] walk-forward: yearly retrain (reducer + Optuna-tuned LightGBM), monthly OOS scoring ...",
          flush=True)
    wf = {m: WF.run_walk_forward(data, label_end, md.calendar, cfg, m, end) for m in args.methods}
    all_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    test_sig = WF.oos_signal_dates(all_dates, cfg.test_start, end)

    # simple non-ML baseline: rank funds by trailing 12-month Sharpe ratio
    base = panel_raw[panel_raw["date"].isin(test_sig)][["date", "code", "sharpe_12m"]]
    base = base.rename(columns={"sharpe_12m": "score"})
    chk = base.merge(fwd_long, on=["date", "code"], how="left")
    base_ic = rank_ic_by_date(chk["date"], chk["fwd_ret"], chk["score"])
    base_ic = base_ic[label_end.reindex(base_ic.index) <= pd.Timestamp(end)]
    ics = {NAMES[m]: res.oos_ic.sort_index() for m, res in wf.items()} | {"Sharpe-12m + RP": base_ic.sort_index()}

    # ------------------------------------------------------------------ 5. portfolios + backtest
    print("[5/6] portfolios (top-10 + risk parity) and daily backtest ...", flush=True)
    bt_kw = dict(start=cfg.test_start, end=end, buy_fee=cfg.buy_fee, sell_fee=cfg.sell_fee, lag=cfg.exec_lag)
    sched_kw = dict(R=md.R, top_k=cfg.top_k, buffer_k=cfg.buffer_k, cov_lookback=cfg.cov_lookback)
    results = {}
    for m, res in wf.items():
        results[NAMES[m]] = BT.run_backtest(NAMES[m], md.R, BT.build_schedule(res.scores, **sched_kw), **bt_kw)
    results["Sharpe-12m + RP"] = BT.run_backtest("Sharpe-12m + RP", md.R, BT.build_schedule(base, **sched_kw), **bt_kw)
    if "pca" in wf:   # ablation: same selection, equal weights instead of risk parity
        results["PCA-6 + EW"] = BT.run_backtest("PCA-6 + EW", md.R,
                                                BT.build_schedule(wf["pca"].scores, scheme="ew", **sched_kw), **bt_kw)
    universe = {s: panel_raw.loc[panel_raw["date"] == s, "code"].tolist() for s in test_sig}
    results["Universe EW"] = BT.equal_weight_universe("Universe EW", md.R, universe, **bt_kw)

    first_day = min(r.nav.index[0] for r in results.values())
    bench_ret = md.bench.loc[first_day:pd.Timestamp(end)].fillna(0.0)
    bench_ret.iloc[0] = 0.0
    benchmark_name = ("Equal-weight return proxy" if md.report["benchmark_source"].startswith("fallback:")
                      else "CSI Equity Fund Index")
    if benchmark_name == "CSI Equity Fund Index":
        results[benchmark_name] = BT.index_series(benchmark_name, md.bench_close[cfg.benchmark], first_day, end)
    else:
        results[benchmark_name] = BT.BacktestResult(benchmark_name, (1 + bench_ret).cumprod(),
                                                    bench_ret, pd.DataFrame())
    if "H00300" in md.bench_close and md.bench_close["H00300"].count() > 250:
        results["CSI 300 TR"] = BT.index_series("CSI 300 TR", md.bench_close["H00300"], first_day, end)

    # ------------------------------------------------------------------ 6. metrics, charts, files
    print("[6/6] metrics, charts, files ...", flush=True)
    rows, ic_rows = [], []
    for wname, (w0, w1) in windows.items():
        sl = slice(pd.Timestamp(w0), pd.Timestamp(w1))
        for name in [n for n in ORDER if n in results]:
            r = results[name]
            st = M.perf_stats(r.returns.loc[sl], bench_ret.loc[sl], cfg.rf)
            if len(r.trades):
                tr = r.trades.set_index("date").loc[sl]
                turnover = tr.iloc[1:] if w0 == cfg.test_start else tr  # initial purchase has no sells
                st["avg_turnover"] = float(turnover["sell_turnover"].mean())
                st["total_costs"] = float(tr["cost"].sum())
            rows.append({"window": wname, "strategy": name, **st})
        for name, ic in ics.items():
            s0, s1 = WF.oos_signal_dates(all_dates, w0, w1)[[0, -1]]
            sample = ic.loc[s0:s1]
            complete = sample[label_end.reindex(sample.index) <= pd.Timestamp(w1)]
            ic_rows.append({"window": wname, "model": name, **M.ic_summary(complete)})
    metrics = pd.DataFrame(rows).set_index(["window", "strategy"])
    ic_tab = pd.DataFrame(ic_rows).set_index(["window", "model"])
    yearly = pd.DataFrame({n: M.yearly_returns(results[n].returns) for n in ORDER if n in results})
    sens = {NAMES[m]: run_sensitivity(res.scores, md.R, bench_ret, cfg, cfg.test_end) for m, res in wf.items()
            if m in ("pca", "ae")}

    metrics.to_csv(out / "metrics.csv")
    ic_tab.to_csv(out / "ic_summary.csv")
    yearly.to_csv(out / "yearly_returns.csv")
    if sens:
        pd.concat(sens, names=["method"]).to_csv(out / "sensitivity.csv")
    pd.DataFrame(ics).to_csv(out / "oos_ic_monthly.csv")
    pd.DataFrame({n: r.nav for n, r in results.items()}).to_csv(out / "nav_daily.csv")
    names = md.funds.set_index("code")["name"]
    for n in ("PCA-6 + RP", "AE-6 + RP"):
        if n in results and len(results[n].holdings):
            h = results[n].holdings.assign(name=results[n].holdings["code"].map(names))
            h.to_csv(out / f"holdings_{n.split()[0]}.csv", index=False, encoding="utf-8-sig")
    with open(out / "walkforward_log.json", "w", encoding="utf-8") as f:
        json.dump({m: res.log for m, res in wf.items()}, f, ensure_ascii=False, indent=2, default=str)

    plot_names = [n for n in ["PCA-6 + RP", "AE-6 + RP", "Raw-30 + RP", "Sharpe-12m + RP",
                              "Universe EW", benchmark_name, "CSI 300 TR"] if n in results]
    main_end = pd.Timestamp(cfg.test_end)

    P.nav_curves({n: results[n].nav.loc[first_day:main_end] for n in plot_names},
                 figs / "nav_2019_2022.png", "Out-of-sample net value, 2019-2022")
    P.nav_curves({n: results[n].nav for n in plot_names}, figs / "nav_2019_2024.png",
                 "Out-of-sample net value, 2019-2024 (shaded: 2023-2024 hold-out)", shade_from="2023-01-01")
    dd_names = [n for n in ["PCA-6 + RP", "AE-6 + RP", benchmark_name] if n in results]
    P.drawdowns({n: results[n].returns for n in dd_names}, figs / "drawdowns.png", "Drawdowns, 2019-2024")
    yb = [n for n in ["PCA-6 + RP", "AE-6 + RP", "Raw-30 + RP", benchmark_name] if n in results]
    P.yearly_bars(yearly[yb], figs / "yearly_returns.png", "Calendar-year returns")
    P.cumulative_ic(ics, figs / "cumulative_ic.png", "Out-of-sample Rank IC, accumulated month by month")
    if "pca" in wf:
        last = max(y for y in wf["pca"].reducers if y <= main_end.year)
        pca = wf["pca"].reducers[last].pca
        load = pd.DataFrame(pca.components_.T, index=F.FACTOR_NAMES,
                            columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
        load.to_csv(out / f"pca_loadings_{last}.csv")
        P.heatmap(load, figs / "pca_loadings.png", f"PCA loadings (model trained for {last})",
                  vmax=float(np.abs(load.values).max()), fmt=".2f", figsize=(6.5, 8.5))
        P.explained_variance(pca.explained_variance_ratio_.tolist(), figs / "pca_explained_variance.png",
                             f"PCA explained variance (model trained for {last})")
    train_part = panel_raw[panel_raw["date"] < pd.Timestamp(cfg.test_start)]
    corr = train_part.groupby("date")[F.FACTOR_NAMES].corr(method="spearman").groupby(level=1).mean()
    corr = corr.loc[F.FACTOR_NAMES, F.FACTOR_NAMES]
    P.heatmap(corr, figs / "factor_corr.png", "Average cross-sectional Spearman correlation (training years)",
              figsize=(9, 8))

    summary = {"config": cfg.to_dict(), "data_report": md.report, "windows": windows,
               "first_trading_day": str(first_day.date()), "minutes": round((time.time() - t_start) / 60, 1)}
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    cols = ["cagr", "excess_pts", "sharpe", "max_dd", "info_ratio", "monthly_hit_rate"]
    for wname in windows:
        print(f"\n================ out-of-sample {wname} ================")
        print(metrics.loc[wname, cols].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\ncalendar-year returns:\n", yearly.to_string(float_format=lambda v: f"{v:+.1%}"))
    print("\nout-of-sample Rank IC:\n", ic_tab.to_string(float_format=lambda v: f"{v:.3f}"))
    if sens:
        print("\nsensitivity, 2019-2022 (scores fixed, portfolio settings varied):\n",
              pd.concat(sens, names=["method"]).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nall results saved to {out.resolve()}  ({summary['minutes']} min)")


if __name__ == "__main__":
    main()
