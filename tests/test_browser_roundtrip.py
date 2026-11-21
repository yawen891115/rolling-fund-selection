"""浏览器抓取通道的往返测试：
用和真实页面同格式的假 pingzhongdata 文本，在 Node 里跑 eastmoney_fetch.js 生成 ndjson.gz，
再用 convert_browser_data.py 解码，结果必须和 download_data.py 直接解析同一文本完全一致。
（需要 node；没有 node 时跳过）
"""
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_download_offline import fake_pingzhong_js, load_module  # noqa: E402

NODE_RUNNER = r"""
const fs = require('fs');
const dir = process.argv[2];
globalThis.fetch = async (url) => {
  const p = url.startsWith('/js/') ? dir + '/fundcode_search.js' : dir + '/pz_' + url.split('/').pop();
  if (!fs.existsSync(p)) return new Response('', { status: 404 });
  return new Response(fs.readFileSync(p, 'utf8'), { status: 200 });
};
const src = fs.readFileSync(process.argv[3], 'utf8');
const res = eval(src);
(async () => {
  const job = globalThis.__fundJob;
  await job.promise;
  if (job.stage !== 'ready') { console.error(job.error || job.stage); process.exit(1); }
  fs.writeFileSync(dir + '/out.ndjson.gz', Buffer.from(await job.blob.arrayBuffer()));
  console.log(JSON.stringify({ res, total: job.total, ok: job.ok, empty: job.empty, rows: job.rows,
                               failed: job.failed, gz: job.gzBytes }));
})();
"""


def test_browser_roundtrip(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser collector round trip")
    tmp = tmp_path
    funds = [["000001", "HXCZHH", "华夏成长混合", "混合型-偏股", ""],
             ["000011", "", "某某优选混合A", "混合型-偏股", ""], ["000012", "", "某某优选混合C", "混合型-偏股", ""],
             ["110011", "", "易方达优质精选混合", "股票型", ""], ["000404", "", "已清盘混合", "混合型-偏股", ""],
             ["015001", "", "新发基金混合", "混合型-偏股", ""], ["000300", "", "某沪深300", "指数型-股票", ""],
             ["000155", "", "某某医疗混合A(后端)", "混合型-偏股", ""]]
    (tmp / "fundcode_search.js").write_text("var r = " + json.dumps(funds, ensure_ascii=False) + ";", encoding="utf-8")
    texts = {}
    for code in ["000001", "000011", "110011"]:
        texts[code] = fake_pingzhong_js(code, n=40)
        (tmp / f"pz_{code}.js").write_text(texts[code], encoding="utf-8")
    (tmp / "runner.js").write_text(NODE_RUNNER, encoding="utf-8")

    js = str(ROOT / "scripts" / "browser" / "eastmoney_fetch.js")
    js_tmp = tmp / "eastmoney_fetch_test.js"
    # 测试时把起止日期放宽到假数据的区间
    js_tmp.write_text(Path(js).read_text(encoding="utf-8").replace("start: 20140101", "start: 20180101"),
                      encoding="utf-8")
    out = subprocess.run(["node", str(tmp / "runner.js"), str(tmp), str(js_tmp)], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout)
    assert info["total"] == 4 and info["ok"] == 3 and info["empty"] == 1, info   # 000404 -> 404 -> empty

    import convert_browser_data as conv
    meta, flist, nav = conv.read_ndjson_gz(tmp / "out.ndjson.gz")
    assert list(flist["code"]) == ["000001", "000011", "000404", "110011"]
    dl = load_module()
    for code, text in texts.items():
        truth = dl.parse_pingzhong(text, code).reset_index(drop=True)
        got = nav[nav["code"] == code].reset_index(drop=True)
        assert len(truth) == len(got)
        assert (truth["date"].to_numpy() == got["date"].to_numpy()).all()
        for col in ["unit_nav", "acc_nav", "ret_pct"]:
            a, b = truth[col].to_numpy(dtype=float), got[col].to_numpy(dtype=float)
            assert np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9), atol=1e-9), (code, col)
        assert list(truth["dividend"]) == list(got["dividend"])
    print("OK", info)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
