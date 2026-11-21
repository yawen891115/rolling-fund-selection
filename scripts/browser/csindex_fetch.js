// 在浏览器里（www.csindex.com.cn 的任意页面）运行：按年分段抓取基准指数日收盘价，生成 benchmarks_csindex.csv
// 930950 中证偏股型基金指数 · 000300 沪深300 · H00300 沪深300全收益
(async () => {
  const codes = ["930950", "000300", "H00300"];
  const rows = ["date,close,index_code"];
  const notes = {};
  for (const c of codes) {
    for (let y = 2014; y <= 2024; y++) {
      const url = `/csindex-home/perf/index-perf?indexCode=${c}&startDate=${y}0101&endDate=${y}1231`;
      try {
        const j = await (await fetch(url, { cache: "no-store" })).json();
        const data = j.data || [];
        if (!data.length) notes[`${c}-${y}`] = "empty";
        for (const d of data) rows.push(`${d.tradeDate},${d.close},${c}`);
      } catch (e) { notes[`${c}-${y}`] = String(e).slice(0, 100); }
    }
  }
  globalThis.__benchCsv = rows.join("\n") + "\n";
  const count = {};
  for (const r of rows.slice(1)) { const c = r.split(",")[2]; count[c] = (count[c] || 0) + 1; }
  return { rows: rows.length - 1, count, notes, first: rows[1], last: rows[rows.length - 1],
           bytes: globalThis.__benchCsv.length };
})();
