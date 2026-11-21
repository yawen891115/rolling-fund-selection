"""Offline test of scripts/download_data.py: fake HTTP / AkShare responses, full run, no network needed.

    python -m pytest tests/test_download_offline.py -q
"""
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_pingzhong_js(code: str, n: int = 30) -> str:
    """Mimics the real structure of Eastmoney pingzhongdata/{code}.js (variable layout, comments, timestamps)."""
    days = pd.bdate_range("2018-09-03", periods=n)
    # Beijing midnight = 16:00 UTC of the previous day, in milliseconds
    ms = [int((d - pd.Timedelta(hours=8)).timestamp() * 1000) for d in days]
    rng = np.random.default_rng(int(code))
    ret = np.round(rng.normal(0.05, 1.2, n), 2)
    ret[0] = 0.0
    unit = np.round(np.cumprod(1 + ret / 100), 4)
    acc = unit + 0.5
    # a 0.05 cash dividend on day 10: unit NAV drops, the growth rate is dividend-adjusted
    unit[10:] = np.round(unit[10:] - 0.05, 4)
    nav_items = []
    for i in range(n):
        money = "分红：每份派现金0.0500元" if i == 10 else ""
        nav_items.append({"x": ms[i], "y": float(unit[i]), "equityReturn": float(ret[i]), "unitMoney": money})
    acc_items = [[ms[i], float(np.round(acc[i], 4))] for i in range(n)]
    acc_items[-1][1] = None  # real data occasionally contains null
    return (
        '/*基金或股票信息*/var ishb=false;var fS_name = "测试基金";var fS_code = "%s";'
        '/*股票仓位测算图*/var Data_fundSharesPositions = [[1715184000000,95.00]];'
        "/*单位净值走势 equityReturn-净值回报 unitMoney-每份派送金*/var Data_netWorthTrend = %s;"
        "/*累计净值走势*/var Data_ACWorthTrend = %s;"
        '/*累计收益率走势*/var Data_grandTotal = [{"name":"x","data":[[1,2],[3,4]]}];'
        % (code, json.dumps(nav_items, ensure_ascii=False), json.dumps(acc_items))
    )


class FakeResp:
    def __init__(self, text, status=200):
        self.text, self.status_code, self.encoding = text, status, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def get(self, url, headers=None, timeout=None):
        code = url.rsplit("/", 1)[-1].split(".")[0]
        if code == "000404":
            return FakeResp("", 404)
        return FakeResp(fake_pingzhong_js(code))


def install_fakes(mod, monkeypatch):
    fake_ak = types.ModuleType("akshare")
    fake_ak.__version__ = "fake"
    fake_ak.fund_name_em = lambda: pd.DataFrame({
        "基金代码": ["000001", "000011", "000012", "110011", "000404", "015001", "000300", "002000", "000155"],
        "拼音缩写": [""] * 9,
        "基金简称": ["华夏成长混合", "某某优选混合A", "某某优选混合C", "易方达优质精选混合",
                 "已清盘混合", "新发基金混合", "某沪深300指数", "某债券A", "某某医疗混合A(后端)"],
        "基金类型": ["混合型-偏股", "混合型-偏股", "混合型-偏股", "混合型-偏股",
                 "混合型-偏股", "混合型-偏股", "指数型-股票", "债券型-长债", "混合型-偏股"],
        "拼音全称": [""] * 9,
    })

    def fake_hist(symbol, start_date, end_date):
        d = pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        return pd.DataFrame({"日期": d.date, "收盘": np.linspace(1000, 1100, len(d))})

    fake_ak.index_hist_csindex = fake_hist
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    fake_requests = types.ModuleType("requests")
    fake_requests.Session = FakeSession
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)


def test_end_to_end(tmp_path, monkeypatch):
    tmp = tmp_path
    mod = load_module()
    install_fakes(mod, monkeypatch)

    # 1) parsing: timestamps must map to Beijing dates
    df = mod.parse_pingzhong(fake_pingzhong_js("000001"), "000001")
    assert df["date"].iloc[0] == pd.Timestamp("2018-09-03"), df["date"].iloc[0]
    assert df["dividend"].str.len().gt(0).sum() == 1
    assert df["acc_nav"].isna().sum() == 1

    # 2) share-class dedupe + type filter + skip new codes + drop back-end classes
    funds = mod.get_fund_list(mod.UNIVERSE_TYPES, include_new=False)
    assert set(funds["code"]) == {"000001", "000011", "110011", "000404"}, funds

    # 3) end to end
    zip_path = tmp / "upload.zip"
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--out", str(tmp / "raw"), "--start", "2018-09-01",
                                     "--end", "2018-12-31", "--workers", "2", "--limit", "10", "--zip", str(zip_path)])
    mod.main()
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    assert {"data/raw/fund_nav.parquet", "data/raw/fund_list.csv",
            "data/raw/benchmarks.csv", "data/raw/download_report.json"} <= names

    nav = pd.read_parquet(tmp / "raw" / "fund_nav.parquet")
    assert nav["code"].nunique() == 3          # 000404 returns 404 -> no data
    assert (nav["unit_nav_e4"] > 0).all()
    report = json.loads((tmp / "raw" / "download_report.json").read_text(encoding="utf-8"))
    assert report["benchmarks"]["930950"] > 0
    print("OK", report["validation"], sorted(names))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
