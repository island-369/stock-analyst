# 示例：特发信息 (000070.SZ) · 2026-06-30

完整分析产物目录。所有数据均来自 `fetch_all.py` 实时采集。

## 文件清单

| 文件 | 说明 |
|------|------|
| `report.md` | 详细 Markdown 报告（模板见 `references/report_template.md`）|
| `report.html` | 由 `md2html.py` 生成的专业排版网页 |
| `summary.json` | 数据摘要（核心字段）|
| `kline_em.png` | 东方财富日 K 线图 |
| `pe_band.png` | 3 年价格水位图（PE 亏损时退化）|
| `financials_trend.png` | 季度营收/净利润趋势 |
| `macro.png` | 宏观环境快照 |

## 快速结论

- **总评级：** 🔴 回避
- **置信度：** 中（数据命中 6/8）
- **核心风险：** 基本面 vs 股价严重背离 — 2026Q1 亏损扩大、PB 18 倍（同业 9.64），3 年价格 87.9% 高位，但 6/24 仍创历史新高 26.05 元
- **关键事件：** 6/26-6/29 连续两根跌停、员工持股 5/19 清仓、中邮证券目标价 18.03 低于现价

## 复现命令

```bash
bash scripts/setup.sh
source scripts/venv/bin/activate
python scripts/fetch_all.py 000070.SZ
```

详细分析流程见仓库根目录的 [SKILL.md](../../SKILL.md)。
