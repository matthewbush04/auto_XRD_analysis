from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_V1_TRAIN = PROJECT_ROOT / "results" / "mixed_phase" / "mixed_phase_cnn_v1_pretrained" / "metrics_summary.json"
DEFAULT_V1_HARD = PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "v1_presence_on_mixed_v2_hard_eval" / "metrics_summary.json"
DEFAULT_V3_TRAIN = PROJECT_ROOT / "results" / "mixed_phase" / "mixed_phase_cnn_v3_hard_aug_presence_50ep" / "metrics_summary.json"
DEFAULT_V3_HARD = PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "v3_hard_aug_presence_on_mixed_v2_hard_eval" / "metrics_summary.json"

DEFAULT_OUTPUT = ROOT / "mixed_phase_dashboard.html"
DEFAULT_SUMMARY_CSV = ROOT / "mixed_phase_dashboard_summary.csv"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def pct_delta(new: float, old: float) -> str:
    return f"{(new - old) * 100:+.1f} pp"


def h(text: Any) -> str:
    return html.escape(str(text), quote=True)


def train_metric(metrics: dict[str, Any], split: str, key: str) -> float:
    return float(metrics[split][key])


def hard_metric(metrics: dict[str, Any], split: str, key: str, mode: str = "fixed") -> float:
    return float(metrics["splits"][split][mode]["metrics"][key])


def hard_group_rows(hard_metrics_path: Path, split: str, group: str) -> list[dict[str, str]]:
    return read_csv(hard_metrics_path.parent / f"{split}_fixed_{group}_metrics.csv")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bar(label: str, value: float, maximum: float = 1.0, accent: str = "blue", detail: str = "") -> str:
    width = max(0.0, min(100.0, value / maximum * 100.0))
    return f"""
      <div class="bar-row">
        <div class="bar-label"><span>{h(label)}</span><b>{fmt(value)}</b></div>
        <div class="bar-track"><div class="bar-fill {accent}" style="width:{width:.2f}%"></div></div>
        <div class="bar-detail">{h(detail)}</div>
      </div>
    """


def comparison_bar(label: str, v1: float, v3: float, key: str) -> str:
    return f"""
      <tr>
        <th>{h(label)}</th>
        <td>{fmt(v1)}</td>
        <td class="strong">{fmt(v3)}</td>
        <td class="delta">{h(pct_delta(v3, v1))}</td>
        <td>{bar('', v1, accent='gray')} {bar('', v3, accent=key)}</td>
      </tr>
    """


def rows_by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if key in row}


def display_group_name(name: str) -> str:
    names = {
        "minor_5pct": "5% 少量副相",
        "minor_10pct": "10% 少量副相",
        "minor_15pct": "15% 少量副相",
        "minor_gt15pct": ">15% 副相",
        "top_1pct": "峰形最相似 Top 1%",
        "top_5pct": "峰形最相似 Top 5%",
        "top_10pct": "峰形最相似 Top 10%",
        "cathode_reference": "正极材料 + 参考/副产物相",
        "anode_reference": "负极材料 + 参考/副产物相",
        "solid_electrolyte_reference": "固态电解质 + 参考/副产物相",
        "cathode_solid_electrolyte_reference": "正极 + 固态电解质 + 参考相",
        "anode_solid_electrolyte_reference": "负极 + 固态电解质 + 参考相",
    }
    return names.get(name, name)


def group_table(title: str, key: str, v1_rows: list[dict[str, str]], v3_rows: list[dict[str, str]], order: list[str]) -> str:
    v1 = rows_by_key(v1_rows, key)
    v3 = rows_by_key(v3_rows, key)
    body = []
    for name in order:
        if name not in v1 or name not in v3:
            continue
        old = float(v1[name]["minor_phase_recall"])
        new = float(v3[name]["minor_phase_recall"])
        body.append(
            f"""
            <tr>
              <th>{h(display_group_name(name))}</th>
              <td>{fmt(float(v1[name]["micro_f1"]))}</td>
              <td>{fmt(float(v3[name]["micro_f1"]))}</td>
              <td>{fmt(old)}</td>
              <td class="strong">{fmt(new)}</td>
              <td class="delta">{h(pct_delta(new, old))}</td>
              <td>{fmt(float(v3[name]["avg_false_positives_per_sample"]))}</td>
            </tr>
            """
        )
    return f"""
      <section class="panel">
        <h2>{h(title)}</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>分组</th>
                <th>v1 micro-F1</th>
                <th>v3 micro-F1</th>
                <th>v1 副相召回率</th>
                <th>v3 副相召回率</th>
                <th>召回率变化</th>
                <th>v3 平均误报相数/样本</th>
              </tr>
            </thead>
            <tbody>{''.join(body)}</tbody>
          </table>
        </div>
      </section>
    """


def build_dashboard(args: argparse.Namespace) -> str:
    v1_train = read_json(args.v1_train_metrics)
    v1_hard = read_json(args.v1_hard_metrics)
    v3_train = read_json(args.v3_train_metrics)
    v3_hard = read_json(args.v3_hard_metrics)

    v1_normal_f1 = train_metric(v1_train, "test_normal", "micro_f1")
    v3_normal_f1 = train_metric(v3_train, "test_normal", "micro_f1")
    v1_hard_minor = train_metric(v1_train, "test_hard", "minor_phase_recall")
    v3_hard_minor = train_metric(v3_train, "test_hard", "minor_phase_recall")

    hard_splits = [
        ("低含量副相测试", "test_minor"),
        ("高峰重叠测试", "test_overlap"),
        ("电池相关混相测试", "test_battery_relevant"),
    ]
    summary_rows: list[dict[str, Any]] = []
    for label, split in hard_splits:
        summary_rows.append(
            {
                "benchmark": label,
                "split": split,
                "v1_micro_f1": fmt(hard_metric(v1_hard, split, "micro_f1")),
                "v3_micro_f1": fmt(hard_metric(v3_hard, split, "micro_f1")),
                "micro_f1_change_pp": pct_delta(hard_metric(v3_hard, split, "micro_f1"), hard_metric(v1_hard, split, "micro_f1")),
                "v1_minor_recall": fmt(hard_metric(v1_hard, split, "minor_phase_recall")),
                "v3_minor_recall": fmt(hard_metric(v3_hard, split, "minor_phase_recall")),
                "minor_recall_change_pp": pct_delta(
                    hard_metric(v3_hard, split, "minor_phase_recall"),
                    hard_metric(v1_hard, split, "minor_phase_recall"),
                ),
            }
        )
    write_summary_csv(args.summary_csv, summary_rows)

    benchmark_rows = "".join(
        comparison_bar(
            label,
            hard_metric(v1_hard, split, "minor_phase_recall"),
            hard_metric(v3_hard, split, "minor_phase_recall"),
            "green" if split == "test_battery_relevant" else "blue",
        )
        for label, split in hard_splits
    )

    micro_rows = "".join(
        comparison_bar(label, hard_metric(v1_hard, split, "micro_f1"), hard_metric(v3_hard, split, "micro_f1"), "blue")
        for label, split in hard_splits
    )

    minor_group = group_table(
        "低含量副相分组结果",
        "minor_fraction_bin",
        hard_group_rows(args.v1_hard_metrics, "test_minor", "minor_fraction_bin"),
        hard_group_rows(args.v3_hard_metrics, "test_minor", "minor_fraction_bin"),
        ["minor_5pct", "minor_10pct", "minor_15pct", "minor_gt15pct"],
    )
    overlap_group = group_table(
        "峰重叠分组结果",
        "overlap_bin",
        hard_group_rows(args.v1_hard_metrics, "test_overlap", "overlap_bin"),
        hard_group_rows(args.v3_hard_metrics, "test_overlap", "overlap_bin"),
        ["top_1pct", "top_5pct", "top_10pct"],
    )
    battery_group = group_table(
        "电池相关场景分组结果",
        "battery_scenario",
        hard_group_rows(args.v1_hard_metrics, "test_battery_relevant", "battery_scenario"),
        hard_group_rows(args.v3_hard_metrics, "test_battery_relevant", "battery_scenario"),
        [
            "cathode_reference",
            "anode_reference",
            "solid_electrolyte_reference",
            "cathode_solid_electrolyte_reference",
            "anode_solid_electrolyte_reference",
        ],
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>混相 XRD 识别 Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #10202a;
      --muted: #52616b;
      --line: #c8d4dc;
      --soft: #f5f8fa;
      --blue: #1769aa;
      --green: #1f8a5b;
      --amber: #b7791f;
      --rose: #b42318;
      --gray: #8796a1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 34px 26px 54px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 9px;
      font-size: 31px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 21px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
      max-width: 1060px;
      margin: 8px 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 15px;
      background: var(--soft);
      min-height: 108px;
    }}
    .metric b {{
      display: block;
      font-size: 25px;
      color: var(--blue);
      margin-bottom: 5px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .callout {{
      border-left: 5px solid var(--green);
      background: #eefaf3;
      padding: 12px 16px;
      margin: 18px 0 24px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin: 16px 0;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 17px;
      margin: 16px 0;
      background: #ffffff;
    }}
    .panel.tint {{
      background: #fbfcfd;
    }}
    .explain {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .explain div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }}
    .explain b {{
      display: block;
      margin-bottom: 5px;
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.35;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 9px;
      text-align: left;
      vertical-align: middle;
    }}
    thead th {{
      background: #e8eef2;
      color: #23313a;
      font-size: 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .strong {{
      color: var(--green);
      font-weight: 700;
    }}
    .delta {{
      color: var(--green);
      font-weight: 700;
      white-space: nowrap;
    }}
    .bar-row {{
      min-width: 210px;
    }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      margin: 2px 0 3px;
    }}
    .bar-track {{
      width: 100%;
      height: 9px;
      background: #e1e8ec;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-fill.blue {{ background: var(--blue); }}
    .bar-fill.green {{ background: var(--green); }}
    .bar-fill.gray {{ background: var(--gray); }}
    .bar-detail {{
      color: var(--muted);
      font-size: 11px;
      min-height: 3px;
    }}
    .workflow {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .step {{
      border-top: 4px solid var(--blue);
      background: var(--soft);
      border-radius: 8px;
      padding: 13px;
      min-height: 120px;
    }}
    .step:nth-child(2) {{ border-top-color: var(--amber); }}
    .step:nth-child(3) {{ border-top-color: var(--green); }}
    code {{
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--line);
      margin-top: 26px;
      padding-top: 14px;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 24px 16px 42px; }}
      .summary, .grid, .explain, .workflow {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 25px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>混相 XRD 识别 Dashboard</h1>
      <p>本页面总结 XRD 自动表征平台中的混相识别模块。相比单相模型只预测一个 phase，混相任务是多标签识别：一条 XRD pattern 可能同时包含主相、少量杂相、副产物或分解产物。</p>
    </header>

    <div class="summary">
      <div class="metric"><b>{fmt(v3_normal_f1)}</b><span>v3 在普通随机混相测试集上的 micro-F1</span></div>
      <div class="metric"><b>{fmt(hard_metric(v3_hard, "test_battery_relevant", "micro_f1"))}</b><span>v3 在电池相关困难混相测试上的 micro-F1</span></div>
      <div class="metric"><b>{fmt(hard_metric(v3_hard, "test_battery_relevant", "minor_phase_recall"))}</b><span>v3 在电池相关混相中的副相召回率</span></div>
      <div class="metric"><b>{fmt(train_metric(v3_train, "test_hard", "minor_phase_recall"))}</b><span>v3 在 mixed_v1 hard split 上的副相召回率</span></div>
    </div>

    <div class="callout">
      <strong>推荐的最终混相模型：</strong><code>mixed_phase_cnn_v3_hard_aug_presence_50ep</code>。
      它在保持普通混相识别性能的同时，提升了低含量副相、峰重叠和电池相关混相场景下的鲁棒性。
    </div>

    <section class="panel tint">
      <h2>从 v1 到 v3 改进了什么？</h2>
      <div class="workflow">
        <div class="step"><h3>v1 baseline</h3><p>使用随机主相/副相混合数据训练，验证 sigmoid 多标签 CNN 可以从一条 XRD pattern 中识别多个 phase。</p></div>
        <div class="step"><h3>困难场景诊断</h3><p>额外构建困难测试集：低含量副相、高峰重叠相对、电池相关物相组合，用来定位模型弱点。</p></div>
        <div class="step"><h3>v3 targeted augmentation</h3><p>在 train/val 中加入有针对性的 hard samples，同时保持 test_normal/test_hard 不变，以验证数据增强是否有效。</p></div>
      </div>
    </section>

    <section class="panel">
      <h2>指标解释</h2>
      <div class="explain">
        <div><b>micro-F1</b><span>把所有 phase 的预测一起统计得到的综合多标签识别分数，是本模块最主要的整体指标。</span></div>
        <div><b>minor recall</b><span>副相召回率，即真实存在的非主相有多少被模型找出来；它直接对应杂相/副产物识别能力。</span></div>
        <div><b>top3 all-hit</b><span>真实 phase 集合是否全部出现在模型置信度最高的 3 个候选中，适合描述候选推荐能力。</span></div>
        <div><b>false positives/sample</b><span>平均每个样本多报了多少不存在的 phase。召回率提升时，这个值可能略微增加。</span></div>
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>普通混相测试</h2>
        {bar("v1 随机混相基线 micro-F1", v1_normal_f1, accent="gray")}
        {bar("v3 困难样本增强后 micro-F1", v3_normal_f1, accent="blue", detail=f"变化：{pct_delta(v3_normal_f1, v1_normal_f1)}")}
        {bar("v1 困难测试副相召回率", v1_hard_minor, accent="gray")}
        {bar("v3 困难测试副相召回率", v3_hard_minor, accent="green", detail=f"变化：{pct_delta(v3_hard_minor, v1_hard_minor)}")}
      </section>

      <section class="panel">
        <h2>展示时最适合讲的结论</h2>
        <p>在电池相关困难基准测试上，针对性困难样本增强将 micro-F1 从 <strong>{fmt(hard_metric(v1_hard, "test_battery_relevant", "micro_f1"))}</strong> 提升到 <strong>{fmt(hard_metric(v3_hard, "test_battery_relevant", "micro_f1"))}</strong>，副相召回率从 <strong>{fmt(hard_metric(v1_hard, "test_battery_relevant", "minor_phase_recall"))}</strong> 提升到 <strong>{fmt(hard_metric(v3_hard, "test_battery_relevant", "minor_phase_recall"))}</strong>。</p>
        <p>代价是平均误报相数略有增加，但对于高通量筛选流程来说是可以接受的：模型先给出候选 phase，后续可以由自动 workflow 或人工进一步确认。</p>
      </section>
    </div>

    <section class="panel">
      <h2>困难基准测试：副相召回率提升</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>测试场景</th><th>v1</th><th>v3</th><th>变化</th><th>可视化对比</th></tr></thead>
          <tbody>{benchmark_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>困难基准测试：micro-F1 提升</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>测试场景</th><th>v1</th><th>v3</th><th>变化</th><th>可视化对比</th></tr></thead>
          <tbody>{micro_rows}</tbody>
        </table>
      </div>
    </section>

    {battery_group}
    {minor_group}
    {overlap_group}

    <section class="panel tint">
      <h2>如何解读这些结果？</h2>
      <p>v3 的结果说明：混相模块已经足够作为单相 XRD 平台的稳定扩展。模型并没有完全解决 5% 极低含量杂相识别问题，但 hard augmentation 证明了平台可以发现并改善真实电池材料样品中常见的 failure mode。</p>
      <p>正式展示时建议重点讲电池相关场景的提升，并把低含量副相作为仍然存在的挑战。峰重叠结果可以作为补充证据，不需要在一分钟主线里展开。</p>
    </section>

    <footer>
      数据来源：<br>
      v1 训练结果：<code>{h(args.v1_train_metrics)}</code><br>
      v1 hard eval：<code>{h(args.v1_hard_metrics)}</code><br>
      v3 训练结果：<code>{h(args.v3_train_metrics)}</code><br>
      v3 hard eval：<code>{h(args.v3_hard_metrics)}</code>
    </footer>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static dashboard for mixed-phase XRD recognition results.")
    parser.add_argument("--v1-train-metrics", type=Path, default=DEFAULT_V1_TRAIN)
    parser.add_argument("--v1-hard-metrics", type=Path, default=DEFAULT_V1_HARD)
    parser.add_argument("--v3-train-metrics", type=Path, default=DEFAULT_V3_TRAIN)
    parser.add_argument("--v3-hard-metrics", type=Path, default=DEFAULT_V3_HARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_dashboard(args)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Saved dashboard: {args.output}")
    print(f"Saved summary CSV: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
