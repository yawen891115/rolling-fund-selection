#!/usr/bin/env python3
"""Download the raw data used by the pipeline (public, first-hand sources).

Sources
-------
1. Fund list and fund type : Eastmoney / Tiantian Fund (fund.eastmoney.com/js/fundcode_search.js) via AkShare
2. Daily NAV of every fund : Eastmoney pingzhongdata/{code}.js -- unit NAV, accumulated NAV and the daily
                             growth rate (dividend-reinvested), one request per fund
3. Benchmarks              : China Securities Index Co. (csindex.com.cn) via AkShare
                             930950 CSI Equity Fund Index (main benchmark), 000300 CSI 300, H00300 CSI 300 TR

Usage
-----
    pip install akshare pandas pyarrow requests
    python scripts/download_data.py --limit 30     # smoke test, < 1 minute
    python scripts/download_data.py                # full run, ~10-20 minutes (resumable: re-run after Ctrl+C)

Output (in --out, default data/raw)
-----------------------------------
    fund_list.csv          code, name, type, first/last NAV date, number of observations
    fund_nav.parquet       daily NAVs (compact integer encoding, see save_nav_parquet)
    benchmarks.csv         benchmark index closes
    download_report.json   counts, failed codes, validation
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # avoid UnicodeEncodeError on legacy Windows consoles when printing Chinese fund names
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
UNIVERSE_TYPES = ["股票型", "混合型-偏股"]          # active stock funds + equity-heavy hybrid funds
BENCHMARKS = {
    "930950": "CSI Equity Fund Index (中证偏股型基金指数)",
    "000300": "CSI 300",
    "H00300": "CSI 300 Total Return",
}
PINGZHONG_URL = "https://fund.eastmoney.com/pingzhongdata/{code}.js"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://fund.eastmoney.com/",
}
# Funds launched from 2022 on (codes ~015000+) have no usable history for a 2019-2022 backtest
NEW_CODE_RANGE = ("015000", "099999")


# ----------------------------------------------------------------------------
# 1. fund list
# ----------------------------------------------------------------------------
def split_share_class(name: str) -> tuple[str, str]:
    """'中欧医疗健康混合C' -> ('中欧医疗健康混合', 'C');  '华夏成长混合' -> ('华夏成长混合', '')"""
    m = re.match(r"^(.{2,}?)([A-Z])$", name)
    if m:
        return m.group(1), m.group(2)
    return name, ""


def get_fund_list(types: list[str], include_new: bool) -> pd.DataFrame:
    import akshare as ak

    raw = ak.fund_name_em()
    df = raw.rename(columns={"基金代码": "code", "基金简称": "name", "基金类型": "type"})
    df = df[["code", "name", "type"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["type"].isin(types)].copy()
    # "(后端)" = back-end-load share class: the same fund as its front-end class, no separate NAV page
    df = df[~df["name"].str.contains("后端", regex=False)]

    # A/C/E... share classes of one fund have near-identical returns: keep one (no suffix or A first)
    parts = df["name"].map(split_share_class)
    df["base_name"] = parts.map(lambda p: p[0])
    df["share_class"] = parts.map(lambda p: p[1])
    df["_pref"] = (~df["share_class"].isin(["", "A"])).astype(int)
    df = (df.sort_values(["base_name", "_pref", "share_class", "code"])
            .drop_duplicates("base_name", keep="first")
            .drop(columns="_pref"))

    if not include_new:
        lo, hi = NEW_CODE_RANGE
        df = df[~df["code"].between(lo, hi)]
    return df.sort_values("code").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. one fund's NAV history
# ----------------------------------------------------------------------------
_thread_local = threading.local()


def _session():
    import requests

    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _extract_js_array(text: str, var: str):
    """Pull `var <var> = [...];` out of the pingzhongdata script (the array itself is valid JSON)."""
    m = re.search(r"\bvar\s+" + re.escape(var) + r"\s*=\s*", text)
    if not m:
        return None
    value, _ = json.JSONDecoder().raw_decode(text, m.end())
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array for {var}")
    return value


def _ms_to_cn_date(ms) -> pd.Series:
    # Timestamps are Beijing midnight (= 16:00 UTC the previous day): convert to Asia/Shanghai first,
    # otherwise every date would be off by one day.
    return (pd.to_datetime(pd.Series(ms), unit="ms", utc=True)
              .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize())


def parse_pingzhong(text: str, code: str) -> pd.DataFrame:
    nav = _extract_js_array(text, "Data_netWorthTrend")
    acc = _extract_js_array(text, "Data_ACWorthTrend")
    if not nav:
        return pd.DataFrame()

    nav_df = pd.DataFrame(nav)
    if "equityReturn" not in nav_df:
        nav_df["equityReturn"] = np.nan
    if "unitMoney" not in nav_df:
        nav_df["unitMoney"] = ""
    out = pd.DataFrame({
        "date": _ms_to_cn_date(nav_df["x"]),
        "unit_nav": pd.to_numeric(nav_df["y"], errors="coerce"),
        "ret_pct": pd.to_numeric(nav_df["equityReturn"], errors="coerce"),
        "dividend": nav_df["unitMoney"].fillna("").astype(str),
    })
    if acc:
        acc_df = pd.DataFrame(acc, columns=["x", "acc_nav"])
        acc_df["date"] = _ms_to_cn_date(acc_df["x"])
        acc_df["acc_nav"] = pd.to_numeric(acc_df["acc_nav"], errors="coerce")
        out = out.merge(acc_df[["date", "acc_nav"]], on="date", how="left")
    else:
        out["acc_nav"] = np.nan
    out.insert(0, "code", code)
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_with_akshare(code: str) -> pd.DataFrame:
    """Fallback channel: plain AkShare calls (two requests per fund, but maintained upstream)."""
    import akshare as ak

    nav = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    if nav is None or nav.empty:
        return pd.DataFrame()
    acc = ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
    out = pd.DataFrame({
        "date": pd.to_datetime(nav["净值日期"]),
        "unit_nav": pd.to_numeric(nav["单位净值"], errors="coerce"),
        "ret_pct": pd.to_numeric(nav["日增长率"], errors="coerce"),
        "dividend": "",
    })
    if acc is not None and not acc.empty:
        acc = acc.rename(columns={"净值日期": "date", "累计净值": "acc_nav"})
        acc["date"] = pd.to_datetime(acc["date"])
        out = out.merge(acc[["date", "acc_nav"]], on="date", how="left")
    else:
        out["acc_nav"] = np.nan
    out.insert(0, "code", code)
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_one(code: str, retries: int = 4) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.05, 0.25))           # be polite
            r = _session().get(PINGZHONG_URL.format(code=code), headers=HEADERS, timeout=20)
            if r.status_code == 404:                         # no page (e.g. liquidated)
                return pd.DataFrame()
            r.raise_for_status()
            r.encoding = "utf-8"
            df = parse_pingzhong(r.text, code)
            if not df.empty:
                return df
            if "Data_netWorthTrend" in r.text:               # valid page, no NAVs yet
                return df
            last_err = RuntimeError("unexpected page format, falling back to AkShare")
            break
        except Exception as e:  # network hiccup: exponential back-off
            last_err = e
            time.sleep(2 ** attempt + random.random())
    try:
        return fetch_with_akshare(code)
    except Exception as e:
        raise RuntimeError(f"{code}: {last_err!r} / akshare: {e!r}")


# ----------------------------------------------------------------------------
# 3. benchmarks
# ----------------------------------------------------------------------------
def fetch_benchmarks(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    import akshare as ak

    # AkShare 1.20.44 renamed stock_zh_index_hist_csindex -> index_hist_csindex; support both
    hist_fn = getattr(ak, "index_hist_csindex", None) or getattr(ak, "stock_zh_index_hist_csindex", None)
    sina_fn = getattr(ak, "index_zh_daily_sina", None) or getattr(ak, "stock_zh_index_daily", None)
    frames, notes = [], {}
    y0, y1 = int(start[:4]), int(end[:4])
    for sym, cname in BENCHMARKS.items():
        parts = []
        for y in range(y0, y1 + 1):                      # one request per year
            s = max(start, f"{y}-01-01").replace("-", "")
            e = min(end, f"{y}-12-31").replace("-", "")
            for attempt in range(3):
                try:
                    df = hist_fn(symbol=sym, start_date=s, end_date=e)
                    if df is not None and not df.empty:
                        parts.append(df[["日期", "收盘"]])
                    break
                except Exception as ex:
                    notes[f"{sym}-{y}"] = repr(ex)
                    time.sleep(2 + attempt * 2)
        if parts:
            d = pd.concat(parts).rename(columns={"日期": "date", "收盘": "close"})
            d["source"] = "csindex"
        elif sym == "000300" and sina_fn is not None:    # CSI 300 fallback source: Sina
            try:
                d = sina_fn(symbol="sh000300")[["date", "close"]]
                d["source"] = "sina"
            except Exception as ex:
                notes[sym] = f"all sources failed: {ex!r}"
                continue
        else:
            notes[sym] = "csindex returned nothing"
            continue
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        d["index_code"], d["index_name"] = sym, cname
        frames.append(d.drop_duplicates("date"))
    out = (pd.concat(frames, ignore_index=True) if frames else
           pd.DataFrame(columns=["date", "close", "source", "index_code", "index_name"]))
    return out, notes


# ----------------------------------------------------------------------------
# 4. save
# ----------------------------------------------------------------------------
def save_nav_parquet(nav: pd.DataFrame, path: Path) -> None:
    """Compact integer encoding (divide back when reading):
        unit_nav_e4 = round(unit NAV * 1e4)   acc_nav_e4 = round(accumulated NAV * 1e4)
        ret_pct_e2  = round(daily growth % * 100)   date = yyyymmdd integer
    With DELTA encoding + zstd the file is about 2/3 the size of a plain float layout.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pd.DataFrame({
        "code": nav["code"].astype(str),
        "date": nav["date"].dt.strftime("%Y%m%d").astype("int32"),
        "unit_nav_e4": np.round(nav["unit_nav"] * 1e4).astype("Int32"),
        "acc_nav_e4": np.round(nav["acc_nav"] * 1e4).astype("Int32"),
        "ret_pct_e2": np.round(nav["ret_pct"] * 100).astype("Int32"),
        "dividend": nav["dividend"].astype(str),
    })
    table = pa.Table.from_pandas(t, preserve_index=False)
    pq.write_table(
        table, path, compression="zstd", compression_level=9,
        use_dictionary=["code", "ret_pct_e2", "dividend"],
        column_encoding={"date": "DELTA_BINARY_PACKED",
                         "unit_nav_e4": "DELTA_BINARY_PACKED",
                         "acc_nav_e4": "DELTA_BINARY_PACKED"},
    )


def validate(nav: pd.DataFrame) -> dict:
    """Cross-check Eastmoney's daily growth rate against the return implied by the accumulated NAV."""
    g = nav.sort_values(["code", "date"]).copy()
    prev_unit = g.groupby("code")["unit_nav"].shift(1)
    prev_acc = g.groupby("code")["acc_nav"].shift(1)
    r_acc = (g["acc_nav"] - prev_acc) / prev_unit * 100
    diff = (r_acc - g["ret_pct"]).abs()
    ok = diff.notna()
    return {
        "rows_checked": int(ok.sum()),
        "share_abs_diff_le_0.02pct": float((diff[ok] <= 0.02).mean()) if ok.any() else None,
        "ret_pct_missing_share": float(g["ret_pct"].isna().mean()),
        "dividend_rows": int((g["dividend"].str.len() > 0).sum()),
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download mutual-fund NAVs (Eastmoney) and CSI benchmarks")
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--workers", type=int, default=6, help="parallel requests; lower it if throttled")
    ap.add_argument("--limit", type=int, default=0, help="only the first N funds (smoke test)")
    ap.add_argument("--include-new", action="store_true", help="also download funds launched from 2022 on")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    ap.add_argument("--zip", default="", help="optionally also pack the outputs into this zip file")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "nav_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/4] fund list (Eastmoney) ...", flush=True)
    funds = get_fund_list(UNIVERSE_TYPES, args.include_new)
    if len(funds) < 200:  # type names changed upstream: say so instead of silently using a tiny sample
        import akshare as ak
        print("  !! very few funds matched; current Eastmoney fund types are listed below. "
              "Update UNIVERSE_TYPES at the top of this script:")
        print(ak.fund_name_em()["基金类型"].value_counts().head(25).to_string())
        if not args.limit:
            sys.exit(1)
    if args.limit:
        funds = funds.head(args.limit)
    print(f"      {len(funds)} funds ({' + '.join(UNIVERSE_TYPES)}, one share class per fund)")

    print(f"[2/4] NAV histories, {args.workers} workers, cache: {cache}", flush=True)
    todo = [c for c in funds["code"] if args.refresh or not (cache / f"{c}.parquet").exists()]
    failed, done = {}, len(funds) - len(todo)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, c): c for c in todo}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                df = fut.result()
                df.to_parquet(cache / f"{code}.parquet", index=False)   # write empty frames too (no re-fetch)
            except Exception as e:
                failed[code] = str(e)[:300]
            done += 1
            if done % 50 == 0 or done == len(funds):
                print(f"      {done}/{len(funds)}  failed {len(failed)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    frames = []
    for c in funds["code"]:
        p = cache / f"{c}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if not d.empty:
                frames.append(d)
    if not frames:
        sys.exit("No fund downloaded. Check that https://fund.eastmoney.com is reachable and retry.")
    nav = pd.concat(frames, ignore_index=True)
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav[(nav["date"] >= args.start) & (nav["date"] <= args.end)]

    span = nav.groupby("code")["date"].agg(first_nav_date="min", last_nav_date="max", n_obs="size")
    funds = funds.merge(span, left_on="code", right_index=True, how="left")

    print("[3/4] benchmarks (China Securities Index Co.) ...", flush=True)
    bench, bench_notes = fetch_benchmarks(args.start, args.end)

    print("[4/4] saving ...", flush=True)
    funds.to_csv(out / "fund_list.csv", index=False, encoding="utf-8-sig")
    save_nav_parquet(nav, out / "fund_nav.parquet")
    bench.to_csv(out / "benchmarks.csv", index=False, encoding="utf-8-sig")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "akshare_version": __import__("akshare").__version__,
        "args": vars(args),
        "n_funds_listed": int(len(funds)),
        "n_funds_with_nav": int(nav["code"].nunique()),
        "nav_rows": int(len(nav)),
        "nav_date_range": [str(nav["date"].min().date()), str(nav["date"].max().date())],
        "failed_codes": failed,
        "benchmarks": {k: int((bench["index_code"] == k).sum()) if len(bench) else 0 for k in BENCHMARKS},
        "benchmark_notes": bench_notes,
        "validation": validate(nav),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ["fund_list.csv", "fund_nav.parquet", "benchmarks.csv", "download_report.json"]:
                z.write(out / f, arcname=f"data/raw/{f}")

    print("\ndone.")
    print(f"  {report['n_funds_with_nav']} funds, {report['nav_rows']:,} NAV rows, {report['nav_date_range']}")
    print(f"  failed: {len(failed)} (see download_report.json); benchmark rows: {report['benchmarks']}")
    print(f"  validation: {report['validation']}")


if __name__ == "__main__":
    main()
