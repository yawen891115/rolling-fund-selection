"""Generate synthetic data in exactly the format of scripts/download_data.py, for offline smoke tests.

    python tests/make_synthetic_data.py --out data/synthetic
    python run_pipeline.py --quick --raw-dir data/synthetic --out-dir results_synthetic

A little predictability is planted (persistent manager skill + trending sector styles) so the model
has something to learn. The resulting backtest numbers mean nothing; this only exercises the code.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _downloader():
    spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(out: Path, n_funds: int = 600, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2014-01-02", "2024-12-31")
    T, K = len(cal), 6
    month = cal.to_period("M")
    m_ids = pd.factorize(month)[0]
    n_m = m_ids.max() + 1

    mkt = 0.0003 + 0.013 * rng.standard_normal(T)                    # market
    drift = np.repeat(rng.normal(0, 0.0012, (n_m // 6 + 1, K)), 6, axis=0)[:n_m]   # sector trends, regime changes every 6 months
    sect = drift[m_ids] + 0.007 * rng.standard_normal((T, K))

    beta = rng.uniform(0.7, 1.05, n_funds)
    load = rng.normal(0, 0.6, (n_funds, K)) * (rng.random((n_funds, K)) < 0.5)
    a = np.zeros((n_m, n_funds))                                       # manager skill: monthly AR(1)
    a[0] = rng.normal(0, 0.03, n_funds)
    for t in range(1, n_m):
        a[t] = 0.92 * a[t - 1] + rng.normal(0, 0.012, n_funds)
    idio = rng.uniform(0.003, 0.008, n_funds)
    R = (a[m_ids] / 252 + mkt[:, None] * beta + sect @ load.T + rng.standard_normal((T, n_funds)) * idio)

    start = np.where(rng.random(n_funds) < 0.6, 0, rng.integers(0, int(T * 0.75), n_funds))
    end = np.where(rng.random(n_funds) < 0.06, rng.integers(int(T * 0.3), T, n_funds), T)
    types = rng.choice(["混合型-偏股", "股票型", "混合型-灵活"], n_funds, p=[0.55, 0.4, 0.05])

    rows = []
    for i in range(n_funds):
        s, e = start[i], max(end[i], start[i] + 300)
        days, r = cal[s:e], np.round(R[s:e, i] * 100, 2)
        r[0] = 0.0
        keep = rng.random(len(days)) > 0.005                           # 0.5% of days without a NAV
        unit = np.cumprod(1 + r / 100)
        div = np.zeros(len(days))
        if rng.random() < 0.3:                                         # 30% of funds pay one dividend
            k = rng.integers(50, len(days) - 1)
            div[k] = round(0.05 * unit[k], 3)
        acc = unit.copy()
        unit = unit - np.cumsum(div)                                   # dividend: unit NAV drops, accumulated NAV unchanged
        rows.append(pd.DataFrame({
            "code": f"{100000 + i:06d}", "date": days[keep], "unit_nav": np.round(unit[keep], 4),
            "acc_nav": np.round(acc[keep], 4), "ret_pct": r[keep],
            "dividend": np.where(div[keep] > 0, "分红：每份派现金", ""),
        }))
    nav = pd.concat(rows, ignore_index=True)

    out.mkdir(parents=True, exist_ok=True)
    _downloader().save_nav_parquet(nav, out / "fund_nav.parquet")
    names = [f"模拟基金{i:03d}{'持有期' if i % 37 == 0 else ''}混合" for i in range(n_funds)]
    pd.DataFrame({"code": [f"{100000 + i:06d}" for i in range(n_funds)], "name": names, "type": types}) \
        .to_csv(out / "fund_list.csv", index=False, encoding="utf-8-sig")

    ew = pd.DataFrame(R, index=cal).where(nav.pivot(index="date", columns="code", values="ret_pct")
                                          .reindex(cal).notna().to_numpy()).mean(axis=1)
    bench = pd.concat([
        pd.DataFrame({"date": cal, "close": 1000 * np.cumprod(1 + ew.fillna(0)), "index_code": "930950"}),
        pd.DataFrame({"date": cal, "close": 3000 * np.cumprod(1 + mkt), "index_code": "000300"}),
        pd.DataFrame({"date": cal, "close": 4000 * np.cumprod(1 + mkt + 0.0001), "index_code": "H00300"}),
    ])
    bench.to_csv(out / "benchmarks.csv", index=False, encoding="utf-8-sig")
    print(f"synthetic data -> {out}  funds={n_funds} rows={len(nav):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-funds", type=int, default=600)
    a = ap.parse_args()
    make(Path(a.out), a.n_funds)
