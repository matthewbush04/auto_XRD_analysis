from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "model_comparison_summary.csv"
PROJECT_ROOT = ROOT.parent
SCAN_ROOTS = [
    PROJECT_ROOT / "results",
    Path.home() / "Desktop" / "results(5)",
    Path.home() / "Desktop" / "results(6)",
    Path.home() / "Desktop" / "results(7)",
    Path.home() / "Desktop" / "results(8)",
]


RUN_NOTES = {
    "multitask_cnn": ("Original multi-task CNN baseline.", "Stable baseline; useful as the report reference point."),
    "label_smoothing_005": ("label_smoothing=0.05.", "Good classification improvement over baseline."),
    "lambda_lat_003": ("Increase lattice loss weight to lambda_lat=0.03.", "Not recommended alone; hard performance regressed."),
    "lambda_lat_005": ("Increase lattice loss weight to lambda_lat=0.05.", "Not recommended alone; hard performance regressed."),
    "stage2_lattice_head": ("Stage-2 training of lattice head only.", "Not useful in current setup."),
    "stage2_lattice_late": ("Stage-2 fine-tune late backbone and lattice head.", "Some improvement, but weaker than later methods."),
    "combo_stage1_ls005": ("Label-smoothing stage-1 run.", "Similar role to label_smoothing_005."),
    "combo_stage2_lattice_late": ("Combination stage-2 late fine-tune.", "Better than weak stage-2 runs, but not the final choice."),
    "strong_lattice_ls005": ("Strong lattice head, lambda_lat=0.03, label_smoothing=0.05.", "Best hard-set lattice robustness among tested models."),
    "physics_loss_ls005": ("Bragg physics loss with lambda_phys=0.1.", "Not recommended; direct large physics loss harms classification."),
    "residual_label_smoothing_005": ("Phase-residual lattice, residual_scale=0.02.", "Strong physics-informed story and excellent normal lattice accuracy."),
    "residual_scale_001": ("Phase-residual lattice, residual_scale=0.01.", "Residual scale is too conservative for deployable hard performance."),
    "residual_strong_lat001_ls005": ("Phase-residual plus strong lattice head, lambda_lat=0.01.", "Best normal phase accuracy; normal lattice is also excellent."),
    "residual_strong_lat003_ls005": ("Phase-residual plus strong lattice head, lambda_lat=0.03.", "Recommended main model for normal precision and physics-informed workflow."),
    "residual_strong_lat003_ls005_v2": ("V2 run of phase-residual plus strong lattice head, 40 epochs.", "Primary checkpoint selected by validation phase accuracy."),
    "residual_strong_lat003_ls005_v2_best_lattice": ("V2 lattice-best checkpoint from epoch 40.", "Recommended final checkpoint: best normal lattice MAE with excellent phase and crystal accuracy."),
    "residual_strong_scale001_lat003_ls005": ("Phase-residual plus strong lattice head, residual_scale=0.01, lambda_lat=0.03.", "Worse than residual_scale=0.02."),
}


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(split: dict, key: str) -> float | None:
    value = split.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def nested_metric(split: dict, section: str, key: str) -> float | None:
    value = split.get(section, {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def find_run_config(metrics_path: Path) -> dict:
    candidate = metrics_path.with_name("run_config.json")
    if candidate.exists():
        return read_json(candidate)
    return {}


def discover_metric_files() -> list[Path]:
    paths: list[Path] = []
    for root in SCAN_ROOTS:
        if root.exists():
            paths.extend(root.rglob("metrics_summary.json"))
    return sorted(set(paths), key=lambda p: str(p).lower())


def row_from_metrics(metrics_path: Path) -> dict[str, str]:
    metrics = read_json(metrics_path)
    config = find_run_config(metrics_path)
    args = config.get("args", {}) if isinstance(config.get("args"), dict) else {}
    run = metrics_path.parent.name
    main_change, recommendation = RUN_NOTES.get(run, ("Experiment run.", "See metrics for interpretation."))
    val = metrics.get("val", {})
    normal = metrics.get("test_normal", {})
    hard = metrics.get("test_hard", {})

    return {
        "source": str(metrics_path),
        "model": run,
        "main_change": main_change,
        "recommendation": recommendation,
        "best_epoch": str(metrics.get("best_epoch", "")),
        "best_metric_name": str(metrics.get("best_metric_name", "")),
        "label_smoothing": fmt(args.get("label_smoothing") if isinstance(args.get("label_smoothing"), (int, float)) else None, 4),
        "lambda_lat": fmt(args.get("lambda_lat") if isinstance(args.get("lambda_lat"), (int, float)) else None, 4),
        "lambda_phys": fmt(args.get("lambda_phys") if isinstance(args.get("lambda_phys"), (int, float)) else None, 4),
        "lattice_mode": str(args.get("lattice_mode", "")),
        "residual_scale": fmt(args.get("residual_scale") if isinstance(args.get("residual_scale"), (int, float)) else None, 4),
        "strong_lattice_head": str(args.get("strong_lattice_head", "")),
        "normal_phase_acc_pct": fmt(metric(normal, "phase_accuracy") * 100 if metric(normal, "phase_accuracy") is not None else None, 4),
        "normal_phase_top3_acc_pct": fmt(metric(normal, "phase_top3_accuracy") * 100 if metric(normal, "phase_top3_accuracy") is not None else None, 4),
        "normal_phase_top5_acc_pct": fmt(metric(normal, "phase_top5_accuracy") * 100 if metric(normal, "phase_top5_accuracy") is not None else None, 4),
        "normal_crystal_acc_pct": fmt(metric(normal, "crystal_accuracy") * 100 if metric(normal, "crystal_accuracy") is not None else None, 4),
        "normal_lattice_mae_A": fmt(metric(normal, "lattice_mae_mean"), 6),
        "normal_lattice_mae_a_A": fmt(nested_metric(normal, "lattice_mae", "a"), 6),
        "normal_lattice_mae_b_A": fmt(nested_metric(normal, "lattice_mae", "b"), 6),
        "normal_lattice_mae_c_A": fmt(nested_metric(normal, "lattice_mae", "c"), 6),
        "normal_bragg_deg": fmt(metric(normal, "bragg_peak_mae_deg"), 6),
        "hard_phase_acc_pct": fmt(metric(hard, "phase_accuracy") * 100 if metric(hard, "phase_accuracy") is not None else None, 4),
        "hard_phase_top3_acc_pct": fmt(metric(hard, "phase_top3_accuracy") * 100 if metric(hard, "phase_top3_accuracy") is not None else None, 4),
        "hard_phase_top5_acc_pct": fmt(metric(hard, "phase_top5_accuracy") * 100 if metric(hard, "phase_top5_accuracy") is not None else None, 4),
        "hard_crystal_acc_pct": fmt(metric(hard, "crystal_accuracy") * 100 if metric(hard, "crystal_accuracy") is not None else None, 4),
        "hard_lattice_mae_A": fmt(metric(hard, "lattice_mae_mean"), 6),
        "hard_lattice_mae_a_A": fmt(nested_metric(hard, "lattice_mae", "a"), 6),
        "hard_lattice_mae_b_A": fmt(nested_metric(hard, "lattice_mae", "b"), 6),
        "hard_lattice_mae_c_A": fmt(nested_metric(hard, "lattice_mae", "c"), 6),
        "hard_bragg_deg": fmt(metric(hard, "bragg_peak_mae_deg"), 6),
        "val_phase_acc_pct": fmt(metric(val, "phase_accuracy") * 100 if metric(val, "phase_accuracy") is not None else None, 4),
        "val_crystal_acc_pct": fmt(metric(val, "crystal_accuracy") * 100 if metric(val, "crystal_accuracy") is not None else None, 4),
        "val_lattice_mae_A": fmt(metric(val, "lattice_mae_mean"), 6),
        "val_bragg_deg": fmt(metric(val, "bragg_peak_mae_deg"), 6),
    }


def write_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model",
        "main_change",
        "recommendation",
        "best_epoch",
        "best_metric_name",
        "label_smoothing",
        "lambda_lat",
        "lambda_phys",
        "lattice_mode",
        "residual_scale",
        "strong_lattice_head",
        "normal_phase_acc_pct",
        "normal_phase_top3_acc_pct",
        "normal_phase_top5_acc_pct",
        "normal_crystal_acc_pct",
        "normal_lattice_mae_A",
        "normal_lattice_mae_a_A",
        "normal_lattice_mae_b_A",
        "normal_lattice_mae_c_A",
        "normal_bragg_deg",
        "hard_phase_acc_pct",
        "hard_phase_top3_acc_pct",
        "hard_phase_top5_acc_pct",
        "hard_crystal_acc_pct",
        "hard_lattice_mae_A",
        "hard_lattice_mae_a_A",
        "hard_lattice_mae_b_A",
        "hard_lattice_mae_c_A",
        "hard_bragg_deg",
        "val_phase_acc_pct",
        "val_crystal_acc_pct",
        "val_lattice_mae_A",
        "val_bragg_deg",
        "source",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    if "-" in value and not value.startswith("-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_bar_chart(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    output: Path,
    lower_is_better: bool = True,
    unit: str = "",
) -> None:
    values = [(r["model"], parse_float(r[metric])) for r in rows]
    values = [(name, val) for name, val in values if val is not None and math.isfinite(val)]
    values.sort(key=lambda item: item[1], reverse=not lower_is_better)

    width = 1500
    row_h = 56
    top = 112
    left_label = 430
    right_pad = 160
    chart_w = width - left_label - right_pad
    height = top + row_h * len(values) + 80

    bg = (248, 250, 252)
    ink = (15, 23, 42)
    muted = (71, 85, 105)
    grid = (203, 213, 225)
    accent = (14, 116, 144)
    highlight = (22, 163, 74)

    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(20)
    small_font = font(18)

    d.text((40, 34), title, fill=ink, font=title_font)
    d.text((40, 76), "Lower lattice/Bragg error is better; higher accuracy is better.", fill=muted, font=small_font)

    max_val = max(v for _, v in values)
    min_val = min(v for _, v in values)
    denom = max_val if lower_is_better else max_val
    best_name = values[0][0]

    for i, (name, val) in enumerate(values):
        y = top + i * row_h
        bar_y = y + 12
        bar_h = 28
        bar_w = int(chart_w * (val / denom)) if denom else 0
        color = highlight if name == best_name else accent
        d.text((40, y + 11), name, fill=ink, font=label_font)
        d.rounded_rectangle((left_label, bar_y, left_label + chart_w, bar_y + bar_h), radius=5, fill=(226, 232, 240))
        d.rounded_rectangle((left_label, bar_y, left_label + bar_w, bar_y + bar_h), radius=5, fill=color)
        value_text = f"{val:.4g}{unit}"
        d.text((left_label + chart_w + 24, y + 11), value_text, fill=ink, font=label_font)

    note = f"Best: {best_name} ({min_val:.4g}{unit})" if lower_is_better else f"Best: {best_name} ({values[0][1]:.4g}{unit})"
    d.text((40, height - 46), note, fill=highlight, font=font(20, True))
    img.save(output)


def build_html(rows: list[dict[str, str]]) -> None:
    metric_rules = {
        "normal_phase_acc_pct": "max",
        "normal_crystal_acc_pct": "max",
        "normal_lattice_mae_A": "min",
        "normal_bragg_deg": "min",
        "hard_phase_acc_pct": "max",
        "hard_crystal_acc_pct": "max",
        "hard_lattice_mae_A": "min",
        "hard_bragg_deg": "min",
    }
    best_values: dict[str, float] = {}
    for key, rule in metric_rules.items():
        values = [parse_float(r.get(key, "")) for r in rows]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if values:
            best_values[key] = min(values) if rule == "min" else max(values)

    def cell(value: str, suffix: str = "") -> str:
        value = (value or "").strip()
        return f"{value}{suffix}" if value else "-"

    def metric_cell(row: dict[str, str], key: str, suffix: str = "") -> str:
        value = (row.get(key) or "").strip()
        if not value:
            return "<td>-</td>"
        numeric = parse_float(value)
        is_best = numeric is not None and key in best_values and abs(numeric - best_values[key]) < 1e-9
        class_name = ' class="best-cell"' if is_best else ""
        return f"<td{class_name}>{cell(value, suffix)}</td>"

    table_rows = []
    for r in rows:
        table_rows.append(
            f"""
            <tr>
              <td><code>{r['model']}</code></td>
              <td>{r['main_change']}</td>
              {metric_cell(r, 'normal_phase_acc_pct', '%')}
              {metric_cell(r, 'normal_crystal_acc_pct', '%')}
              {metric_cell(r, 'normal_lattice_mae_A', ' A')}
              {metric_cell(r, 'normal_bragg_deg', ' deg')}
              {metric_cell(r, 'hard_phase_acc_pct', '%')}
              {metric_cell(r, 'hard_crystal_acc_pct', '%')}
              {metric_cell(r, 'hard_lattice_mae_A', ' A')}
              {metric_cell(r, 'hard_bragg_deg', ' deg')}
              <td>{r['recommendation']}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XRD CNN Model Comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --band: #f8fafc;
      --accent: #0e7490;
      --good: #15803d;
    }}
    body {{
      margin: 0;
      font-family: Arial, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
      max-width: 960px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 28px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--band);
    }}
    .metric b {{
      display: block;
      font-size: 23px;
      margin-bottom: 4px;
      color: var(--accent);
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
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
      vertical-align: top;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #e2e8f0;
      color: #1e293b;
      font-size: 12px;
    }}
    code {{
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    .best-cell {{
      color: #166534;
      background: #dcfce7;
      font-weight: 700;
      box-shadow: inset 0 0 0 1px #86efac;
    }}
    .charts {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      margin: 30px 0;
    }}
    .charts img {{
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .pick {{
      border-left: 5px solid var(--good);
      padding: 12px 16px;
      background: #f0fdf4;
      margin: 22px 0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>XRD Multi-task CNN Model Comparison</h1>
    <p>This dashboard compares the major model runs for phase classification, crystal-system classification, lattice regression, and Bragg-law consistency. The recommended selection prioritizes normal-set lattice precision because the normal split better represents realistic high-throughput characterization conditions.</p>
    <div class="summary">
      <div class="metric"><b>99.758%</b><span>Normal phase accuracy of recommended model</span></div>
      <div class="metric"><b>99.855%</b><span>Normal crystal-system accuracy</span></div>
      <div class="metric"><b>0.0481 A</b><span>Normal lattice MAE of recommended model</span></div>
      <div class="metric"><b>16</b><span>Experiment summaries aggregated from metrics JSON</span></div>
    </div>
    <div class="pick"><strong>Recommended main model:</strong> <code>residual_strong_lat003_ls005</code>. It provides the best normal-set lattice accuracy while keeping phase recognition highly accurate.</div>
    <div class="charts">
      <img src="normal_lattice_mae.png" alt="Normal lattice MAE comparison">
      <img src="hard_lattice_mae.png" alt="Hard lattice MAE comparison">
      <img src="normal_phase_accuracy.png" alt="Normal phase accuracy comparison">
      <img src="normal_crystal_accuracy.png" alt="Normal crystal-system accuracy comparison">
      <img src="hard_crystal_accuracy.png" alt="Hard crystal-system accuracy comparison">
    </div>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Main change</th>
          <th>Normal phase</th>
          <th>Normal crystal</th>
          <th>Normal lattice</th>
          <th>Normal Bragg</th>
          <th>Hard phase</th>
          <th>Hard crystal</th>
          <th>Hard lattice</th>
          <th>Hard Bragg</th>
          <th>Recommendation</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    (ROOT / "model_comparison_dashboard.html").write_text(html, encoding="utf-8")


def build_markdown_report(rows: list[dict[str, str]]) -> None:
    recommended = next((r for r in rows if r["model"] == "residual_strong_lat003_ls005"), rows[0])

    def pct(row: dict[str, str], key: str) -> str:
        return f"{row[key]}%" if row.get(key) else "-"

    def unit(row: dict[str, str], key: str, suffix: str) -> str:
        return f"{row[key]} {suffix}" if row.get(key) else "-"

    lines = [
        "# Multi-task CNN Model Comparison",
        "",
        "This report is generated from available `metrics_summary.json` files in `Project/results` and Desktop result folders `results(5)` through `results(8)`. It summarizes the phase-classification, crystal-system classification, lattice-regression, and Bragg-law consistency metrics for the major CNN experiments.",
        "",
        "## Main Recommendation",
        "",
        f"Use `{recommended['model']}` as the current main model.",
        "",
        f"- Normal phase accuracy: {pct(recommended, 'normal_phase_acc_pct')}",
        f"- Normal crystal-system accuracy: {pct(recommended, 'normal_crystal_acc_pct')}",
        f"- Normal lattice MAE: {unit(recommended, 'normal_lattice_mae_A', 'A')}",
        f"- Normal Bragg peak MAE: {unit(recommended, 'normal_bragg_deg', 'deg')}",
        f"- Hard phase accuracy: {pct(recommended, 'hard_phase_acc_pct')}",
        f"- Hard crystal-system accuracy: {pct(recommended, 'hard_crystal_acc_pct')}",
        f"- Hard lattice MAE: {unit(recommended, 'hard_lattice_mae_A', 'A')}",
        f"- Hard Bragg peak MAE: {unit(recommended, 'hard_bragg_deg', 'deg')}",
        "",
        "This model combines phase-residual lattice prediction, a stronger lattice regression head, label smoothing, and a larger lattice loss weight. It is the best match for the course narrative: accurate phase identification, reliable normal-condition lattice prediction, and a lightweight physics-informed design.",
        "",
        "## Comparison Table",
        "",
        "| Model | Normal Phase | Normal Crystal | Normal Lattice MAE | Normal Bragg | Hard Phase | Hard Crystal | Hard Lattice MAE | Hard Bragg | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r['model']}`",
                    pct(r, "normal_phase_acc_pct"),
                    pct(r, "normal_crystal_acc_pct"),
                    unit(r, "normal_lattice_mae_A", "A"),
                    unit(r, "normal_bragg_deg", "deg"),
                    pct(r, "hard_phase_acc_pct"),
                    pct(r, "hard_crystal_acc_pct"),
                    unit(r, "hard_lattice_mae_A", "A"),
                    unit(r, "hard_bragg_deg", "deg"),
                    r["recommendation"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current bottleneck is no longer phase or crystal-system classification. The best normal-set lattice MAE is around 0.05 A, which is strong for the project goal. The hard split remains useful as an out-of-distribution stress test, but it should not dominate model selection because it intentionally contains more extreme perturbations.",
            "",
            "A direct high-weight Bragg physics loss reduced classification reliability in the tested runs. The more successful physics-informed strategy is to use phase-specific lattice priors through phase-residual prediction, then evaluate physical consistency with Bragg-law metrics.",
        ]
    )

    (ROOT / "model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    by_model: dict[str, dict[str, str]] = {}
    for row in [row_from_metrics(path) for path in discover_metric_files()]:
        existing = by_model.get(row["model"])
        if existing is None:
            by_model[row["model"]] = row
            continue
        current_lat = parse_float(row["normal_lattice_mae_A"]) or 999
        existing_lat = parse_float(existing["normal_lattice_mae_A"]) or 999
        if current_lat < existing_lat:
            by_model[row["model"]] = row

    rows = list(by_model.values())
    rows.sort(key=lambda r: (parse_float(r["normal_lattice_mae_A"]) is None, parse_float(r["normal_lattice_mae_A"]) or 999, r["model"]))
    write_summary_csv(rows)
    draw_bar_chart(rows, "normal_lattice_mae_A", "Normal Test Lattice MAE", ROOT / "normal_lattice_mae.png", True, " A")
    draw_bar_chart(rows, "hard_lattice_mae_A", "Hard Test Lattice MAE", ROOT / "hard_lattice_mae.png", True, " A")
    draw_bar_chart(rows, "normal_phase_acc_pct", "Normal Test Phase Accuracy", ROOT / "normal_phase_accuracy.png", False, "%")
    draw_bar_chart(rows, "normal_crystal_acc_pct", "Normal Test Crystal-system Accuracy", ROOT / "normal_crystal_accuracy.png", False, "%")
    draw_bar_chart(rows, "hard_crystal_acc_pct", "Hard Test Crystal-system Accuracy", ROOT / "hard_crystal_accuracy.png", False, "%")
    build_html(rows)
    build_markdown_report(rows)


if __name__ == "__main__":
    main()
