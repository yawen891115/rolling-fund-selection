# Data

The dataset used for every number in the main README is committed in `data/raw/`:

| File | Content | Source |
|---|---|---|
| [fund_list.csv](raw/fund_list.csv) | 2,237 listed funds with code, name, category, first and last NAV dates | Eastmoney `fund.eastmoney.com/js/fundcode_search.js` |
| [fund_nav.parquet](raw/fund_nav.parquet) | 3,128,021 daily observations for 2,233 funds from 2014-01-02 through 2024-12-31, including NAVs, daily growth rates and dividend notes | Eastmoney `fund.eastmoney.com/pingzhongdata/{code}.js` |
| [benchmarks.csv](raw/benchmarks.csv) | 8,057 daily closes across 930950 CSI Equity Fund Index, 000300 CSI 300 and H00300 CSI 300 Total Return | China Securities Index Co. `csindex.com.cn` |
| [download_report.json](raw/download_report.json) | Acquisition counts and a consistency check against accumulated NAV | Generated during collection |

Four listed funds have no NAV rows in this period. Of the 1,870 funds that pass the static eligibility
rules, 1,866 have observed NAVs. The Parquet key is the pair `(code, date)`. Its SHA-256 checksum is
`1e0430549f2c93ab32ec7f1dc03bbd9767a7f33e9e6284e6c75e560e9edac755`.

`fund_nav.parquet` stores NAVs as scaled integers to keep the file small:
`unit_nav_e4 = round(unit NAV x 1e4)`, `acc_nav_e4 = round(accumulated NAV x 1e4)`,
`ret_pct_e2 = round(daily growth % x 100)`, `date = yyyymmdd`. `src/fundsel/data.py` decodes it.

## How it was collected

The committed files were collected with the browser scripts in `scripts/browser/`, which run the same requests
and selection logic as `scripts/download_data.py` inside a browser tab on each source site. They were then
converted with `scripts/convert_browser_data.py`. `tests/test_browser_roundtrip.py` checks that both paths
decode to identical values.

To rebuild the dataset from scratch:

```bash
pip install akshare pandas pyarrow requests
python scripts/download_data.py          # ~10-20 min, resumable
```

## Universe rules

* Eastmoney types `股票型` (active stock funds) and `混合型-偏股` (equity-heavy hybrid funds).
* One share class per fund: the class without a suffix, or the A class, is preferred. Back-end-load classes
  (`后端`) are dropped.
* The collector excludes codes 015000–099999 as a proxy for recent launches. Actual NAV histories
  determine each fund's eligibility at a given month-end.
* Lock-up and periodic-open funds (持有期 / 定开 / 封闭) are excluded later by the pipeline (`config.py`).

## Known caveats

* **Survivorship bias.** No fund in the list stopped reporting before mid-2024: Eastmoney's list only contains
  funds that still exist, so funds liquidated between 2015 and 2024 are missing.
* **Current classification.** Fund types are Eastmoney's current labels, not point-in-time ones.
* **Calendar artifact.** CSIndex date-range queries return a spurious Jan-1 row. The pipeline's trading calendar
  drops Jan 1 and weekends.
* **Data rights.** Eastmoney and China Securities Index Co. are the sources of the underlying data.
  Downstream use is subject to the respective providers' terms. The repository's MIT license covers the code.
