#!/usr/bin/env python3
"""把浏览器抓取的文件转换成流水线使用的标准格式（与 download_data.py 的输出完全相同）

    python scripts/convert_browser_data.py fund_nav_eastmoney.ndjson.gz benchmarks_csindex.csv --out data/raw

输入：
  fund_nav_eastmoney.ndjson.gz   scripts/browser/eastmoney_fetch.js 在天天基金页面里生成
  benchmarks_csindex.csv         scripts/browser/csindex_fetch.js 在中证指数官网页面里生成
输出（data/raw/）：fund_list.csv、fund_nav.parquet、benchmarks.csv、download_report.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_NAMES = {"930950": "中证偏股型基金指数", "000300": "沪深300", "H00300": "沪深300全收益"}


def _downloader():
    spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _undelta(xs: list) -> np.ndarray:
    """差分解码：第一个非空值是绝对值，之后每个非空值是相对上一个非空值的增量；None -> NaN"""
    out = np.full(len(xs), np.nan)
    prev = None
    for i, x in enumerate(xs):
        if x is None:
            continue
        prev = x if prev is None else prev + x
        out[i] = prev
    return out


def decode_line(obj: dict) -> pd.DataFrame:
    days = np.cumsum(np.asarray(obj["g"], dtype="int64"))
    d0 = pd.Timestamp(str(obj["d0"]))
    unit_e4 = _undelta(obj["u"])
    acc_minus_unit = _undelta(obj["a"])
    ret = np.array([np.nan if x is None else x for x in obj["r"]], dtype="float64")
    div = np.full(len(days), "", dtype=object)
    for i, text in obj.get("v", []):
        div[i] = text
    return pd.DataFrame({
        "code": obj["c"],
        "date": d0 + pd.to_timedelta(days, unit="D"),
        "unit_nav": unit_e4 / 1e4,
        "acc_nav": (unit_e4 + acc_minus_unit) / 1e4,
        "ret_pct": ret / 100,
        "dividend": div,
    })


def read_ndjson_gz(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        frames = [decode_line(json.loads(line)) for line in f if line.strip()]
    funds = pd.DataFrame(header["funds"], columns=["code", "name", "type"])
    nav = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return header["meta"], funds, nav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fund_file")
    ap.add_argument("bench_file")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    dl = _downloader()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta, funds, nav = read_ndjson_gz(Path(args.fund_file))
    funds = funds[~funds["name"].str.contains("后端", regex=False)].reset_index(drop=True)   # 后端收费重复份额
    parts = funds["name"].map(dl.split_share_class)
    funds["base_name"], funds["share_class"] = parts.map(lambda p: p[0]), parts.map(lambda p: p[1])
    span = nav.groupby("code")["date"].agg(first_nav_date="min", last_nav_date="max", n_obs="size")
    funds = funds.merge(span, left_on="code", right_index=True, how="left")

    b = pd.read_csv(args.bench_file, dtype={"index_code": str, "date": str})
    bench = pd.DataFrame({"date": pd.to_datetime(b["date"], format="%Y%m%d"), "close": b["close"].astype(float),
                          "source": "csindex", "index_code": b["index_code"],
                          "index_name": b["index_code"].map(BENCH_NAMES)})
    bench = bench.drop_duplicates(["index_code", "date"]).sort_values(["index_code", "date"])

    funds.to_csv(out / "fund_list.csv", index=False, encoding="utf-8-sig")
    dl.save_nav_parquet(nav, out / "fund_nav.parquet")
    bench.to_csv(out / "benchmarks.csv", index=False, encoding="utf-8-sig")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "browser fetch (Claude in Chrome) + convert_browser_data.py",
        "browser_meta": {k: v for k, v in meta.items() if k != "failed"},
        "failed_codes": meta.get("failed", {}),
        "n_funds_listed": int(len(funds)), "n_funds_with_nav": int(nav["code"].nunique()),
        "nav_rows": int(len(nav)),
        "nav_date_range": [str(nav["date"].min().date()), str(nav["date"].max().date())],
        "benchmarks": bench.groupby("index_code").size().astype(int).to_dict(),
        "validation": dl.validate(nav),
    }
    (out / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["n_funds_listed", "n_funds_with_nav", "nav_rows", "nav_date_range",
                                             "benchmarks", "validation"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
