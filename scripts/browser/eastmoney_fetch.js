// 在浏览器里（fund.eastmoney.com 的任意页面）运行：抓取「股票型 + 混合型-偏股」基金的历史净值，
// 打包成 fund_nav_eastmoney.ndjson.gz。逻辑与 scripts/download_data.py 完全一致，只是换成浏览器的 fetch。
//
// 文件格式（每行一个 JSON）：
//   第 1 行  {"meta": {...}, "funds": [[代码, 简称, 类型], ...]}
//   之后每行 {"c": 代码, "d0": 首日yyyymmdd, "g": [日期间隔(天)], "r": [日增长率%×100],
//            "u": [单位净值×1e4 的差分], "a": [(累计−单位)净值×1e4 的差分], "v": [[行号, 分红说明], ...]}
//   差分编码：数组第一个非空值是绝对值，之后是相对上一个非空值的增量；null 表示缺失。
(() => {
  const CFG = { start: 20140101, end: 20241231, types: ["股票型", "混合型-偏股"], workers: 6,
                newLo: "015000", newHi: "099999" };
  const job = { stage: "starting", total: 0, done: 0, ok: 0, empty: 0, rows: 0, failed: {},
                startedAt: Date.now(), blob: null };
  globalThis.__fundJob = job;

  const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
  const splitShare = (name) => { const m = /^(.{2,}?)([A-Z])$/.exec(name); return m ? [m[1], m[2]] : [name, ""]; };
  // 时间戳是北京时间 0 点；加 8 小时后取 UTC 日期 = 北京日期
  const cnDate = (ms) => { const d = new Date(ms + 8 * 3600e3);
    return d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate(); };
  const epochDay = (ymd) => Date.UTC(Math.floor(ymd / 10000), Math.floor(ymd / 100) % 100 - 1, ymd % 100) / 864e5;
  const getArr = (text, name) => {
    const m = new RegExp("var\\s+" + name + "\\s*=\\s*(\\[[\\s\\S]*?\\]);").exec(text);
    if (!m) return null;
    try { return JSON.parse(m[1]); } catch (e) { return null; }
  };

  function selectFunds(raw) {
    let f = raw.filter((x) => CFG.types.includes(x[3]) && !x[2].includes("后端"))   // 后端收费份额 = 重复份额
               .map((x) => ({ code: x[0], name: x[2], type: x[3] }));
    f.forEach((x) => { const [b, c] = splitShare(x.name); x.base = b; x.cls = c; x.pref = c === "" || c === "A" ? 0 : 1; });
    f.sort((a, b) => cmp(a.base, b.base) || a.pref - b.pref || cmp(a.cls, b.cls) || cmp(a.code, b.code));
    const seen = new Set();
    f = f.filter((x) => (seen.has(x.base) ? false : (seen.add(x.base), true)));
    f = f.filter((x) => !(x.code >= CFG.newLo && x.code <= CFG.newHi));
    return f.sort((a, b) => cmp(a.code, b.code));
  }

  function encodeFund(code, text) {
    const nav = getArr(text, "Data_netWorthTrend");
    if (!nav) return text.includes("Data_netWorthTrend") ? "" : null;   // "" 正常但无数据；null 格式异常
    const accMap = new Map();
    for (const p of getArr(text, "Data_ACWorthTrend") || []) {
      if (p && p[1] != null && Number.isFinite(Number(p[1]))) accMap.set(cnDate(p[0]), Math.round(Number(p[1]) * 1e4));
    }
    const g = [], r = [], u = [], a = [], v = [], seen = new Set();
    let d0 = null, prevDay = null, prevU = null, prevA = null;
    for (const e of nav) {
      if (!e || e.x == null) continue;
      const ymd = cnDate(e.x);
      if (ymd < CFG.start || ymd > CFG.end || seen.has(ymd)) continue;
      seen.add(ymd);
      const day = epochDay(ymd);
      g.push(d0 === null ? 0 : day - prevDay);
      if (d0 === null) d0 = ymd;
      prevDay = day;
      const ret = e.equityReturn === "" || e.equityReturn == null ? NaN : Math.round(Number(e.equityReturn) * 100);
      r.push(Number.isFinite(ret) ? ret : null);
      const ue = e.y == null || e.y === "" ? NaN : Math.round(Number(e.y) * 1e4);
      if (!Number.isFinite(ue)) u.push(null); else { u.push(prevU === null ? ue : ue - prevU); prevU = ue; }
      const ae = Number.isFinite(ue) && accMap.has(ymd) ? accMap.get(ymd) - ue : null;
      if (ae === null) a.push(null); else { a.push(prevA === null ? ae : ae - prevA); prevA = ae; }
      if (e.unitMoney) v.push([g.length - 1, String(e.unitMoney)]);
    }
    if (d0 === null) return "";
    job.rows += g.length;
    return JSON.stringify({ c: code, d0, g, r, u, a, v });
  }

  async function fetchText(url, tries = 4) {
    let err;
    for (let i = 0; i < tries; i++) {
      try {
        const resp = await fetch(url, { cache: "no-store" });
        if (resp.status === 404) return null;
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return await resp.text();
      } catch (e) { err = e; await sleep(800 * 2 ** i); }
    }
    throw err;
  }

  async function run() {
    job.stage = "list";
    const listText = await fetchText("/js/fundcode_search.js");
    const raw = JSON.parse(listText.slice(listText.indexOf("["), listText.lastIndexOf("]") + 1));
    const funds = selectFunds(raw);
    job.total = funds.length;
    const lines = new Array(funds.length).fill(null);
    let next = 0;
    job.stage = "nav";
    const worker = async () => {
      while (next < funds.length) {
        const i = next++;
        const f = funds[i];
        try {
          const t = await fetchText(`/pingzhongdata/${f.code}.js`);
          const enc = t == null ? "" : encodeFund(f.code, t);
          if (enc === null) job.failed[f.code] = "unexpected page format";
          else if (enc === "") job.empty++;
          else { lines[i] = enc; job.ok++; }
        } catch (e) { job.failed[f.code] = String(e).slice(0, 120); }
        job.done++;
      }
    };
    await Promise.all(Array.from({ length: CFG.workers }, worker));
    job.stage = "pack";
    const meta = { source: "fund.eastmoney.com/pingzhongdata", fund_list: "fund.eastmoney.com/js/fundcode_search.js",
                   start: CFG.start, end: CFG.end, types: CFG.types, generated_at: new Date().toISOString(),
                   n_list_raw: raw.length, n_selected: funds.length, n_ok: job.ok, n_empty: job.empty,
                   failed: job.failed, rows: job.rows, user_agent: (globalThis.navigator || {}).userAgent };
    const parts = [JSON.stringify({ meta, funds: funds.map((f) => [f.code, f.name, f.type]) }) + "\n"];
    for (const l of lines) if (l) parts.push(l + "\n");
    const blob = new Blob(parts, { type: "application/x-ndjson" });
    job.rawBytes = blob.size;
    job.blob = await new Response(blob.stream().pipeThrough(new CompressionStream("gzip"))).blob();
    job.gzBytes = job.blob.size;
    job.stage = "ready";
    job.minutes = +((Date.now() - job.startedAt) / 60000).toFixed(1);
  }

  globalThis.__fundHelpers = { selectFunds, encodeFund, cnDate };   // 便于测试
  job.promise = run().catch((e) => { job.stage = "error"; job.error = String((e && e.stack) || e); });
  return "started";
})();
