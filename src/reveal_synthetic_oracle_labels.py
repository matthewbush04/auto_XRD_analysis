from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "active_learning"


def read_selected_indices(path: Path) -> list[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_index" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no sample_index column.")
        return [int(row["sample_index"]) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def component_list(values: np.ndarray, n_phases: int) -> str:
    return ";".join(str(values[idx]) for idx in range(n_phases))


def make_rows(pool_path: Path, indices: list[int]) -> list[dict[str, Any]]:
    data = np.load(pool_path, allow_pickle=True)
    labels = np.asarray(data["labels"]).astype(str)
    rows: list[dict[str, Any]] = []
    for rank, sample_idx in enumerate(indices, start=1):
        n_phases = int(data["n_phases"][sample_idx])
        phase_indices = np.asarray(data["component_phase_indices"][sample_idx, :n_phases], dtype=np.int64)
        fractions = np.asarray(data["component_fractions"][sample_idx, :n_phases], dtype=np.float32)
        row: dict[str, Any] = {
            "oracle_rank": rank,
            "sample_index": int(sample_idx),
            "n_phases": n_phases,
            "phase_indices": component_list(phase_indices, n_phases),
            "phase_labels": component_list(labels[phase_indices], n_phases),
            "component_fractions": component_list(fractions, n_phases),
        }
        for key in ("hard_eval_type", "minor_fraction_bin", "overlap_bin", "battery_scenario"):
            if key in data.files:
                row[key] = str(data[key][sample_idx])
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reveal hidden synthetic labels for selected active-learning candidates."
    )
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.selected_csv.with_name("oracle_labels_active.csv")
    if output.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing oracle label file: {output}", file=sys.stderr)
        return 1

    try:
        indices = read_selected_indices(args.selected_csv)
        rows = make_rows(args.candidate_pool, indices)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, rows)
        summary_output = args.summary_output or output.with_name("oracle_label_summary.json")
        summary = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "candidate_pool": str(args.candidate_pool),
            "selected_csv": str(args.selected_csv),
            "output": str(output),
            "revealed_count": len(rows),
            "note": "This synthetic oracle step is the first point where hidden ground-truth labels are read.",
        }
        write_json(summary_output, summary)
        print(f"Revealed synthetic oracle labels: {output}")
    except Exception as exc:
        print(f"Synthetic oracle reveal failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
