from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed" / "single_phase"

SPLITS = ("train", "val", "test_normal", "test_hard")


def load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    df = pd.read_csv(path)
    required = {
        "record_id",
        "material_id",
        "phase_label",
        "crystal_system",
        "spacegroup_symbol",
        "spacegroup_number",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["crystal_system"] = df["crystal_system"].fillna("unknown").astype(str)
    crystal_system_labels = sorted(df["crystal_system"].unique().tolist())
    crystal_to_idx = {label: idx for idx, label in enumerate(crystal_system_labels)}
    records = {
        str(row.record_id): {
            "material_id": str(row.material_id),
            "phase_label": str(row.phase_label),
            "crystal_system": str(row.crystal_system),
            "crystal_system_idx": crystal_to_idx[str(row.crystal_system)],
            "spacegroup_symbol": str(row.spacegroup_symbol),
            "spacegroup_number": -1 if pd.isna(row.spacegroup_number) else int(row.spacegroup_number),
        }
        for row in df.itertuples(index=False)
    }
    return df, records, crystal_to_idx


def enrich_split(path: Path, records: dict[str, dict[str, Any]], crystal_to_idx: dict[str, int]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    data = np.load(path, allow_pickle=False)
    try:
        arrays = {key: data[key] for key in data.files}
    finally:
        data.close()
    if "sample_record_ids" not in arrays:
        raise ValueError(f"{path} is missing sample_record_ids")

    sample_record_ids = arrays["sample_record_ids"].astype(str)
    missing = sorted({record_id for record_id in sample_record_ids if record_id not in records})
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{path} has record IDs not found in metadata: {preview}")

    sample_phase_labels = []
    sample_crystal_systems = []
    crystal_system_y = []
    sample_spacegroup_symbols = []
    sample_spacegroup_numbers = []
    for record_id in sample_record_ids:
        item = records[str(record_id)]
        sample_phase_labels.append(item["phase_label"])
        sample_crystal_systems.append(item["crystal_system"])
        crystal_system_y.append(item["crystal_system_idx"])
        sample_spacegroup_symbols.append(item["spacegroup_symbol"])
        sample_spacegroup_numbers.append(item["spacegroup_number"])

    arrays.update(
        {
            "sample_phase_labels": np.asarray(sample_phase_labels),
            "sample_crystal_systems": np.asarray(sample_crystal_systems),
            "crystal_system_y": np.asarray(crystal_system_y, dtype=np.int64),
            "crystal_system_labels": np.asarray(list(crystal_to_idx.keys())),
            "sample_spacegroup_symbols": np.asarray(sample_spacegroup_symbols),
            "sample_spacegroup_numbers": np.asarray(sample_spacegroup_numbers, dtype=np.int64),
        }
    )

    tmp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)

    return {
        "split": path.stem,
        "path": str(path),
        "sample_count": int(len(sample_record_ids)),
        "crystal_system_counts": {
            label: int((arrays["sample_crystal_systems"] == label).sum())
            for label in crystal_to_idx
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add crystal-system and spacegroup metadata to existing NPZ splits.")
    parser.add_argument("--metadata", default=str(METADATA_DIR / "materials_index.csv"))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata)
    processed_dir = Path(args.processed_dir)
    _, records, crystal_to_idx = load_metadata(metadata_path)

    summaries = []
    for split in SPLITS:
        summaries.append(enrich_split(processed_dir / f"{split}.npz", records, crystal_to_idx))

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata": str(metadata_path),
        "processed_dir": str(processed_dir),
        "crystal_system_labels": list(crystal_to_idx.keys()),
        "split_summaries": summaries,
    }
    (processed_dir / "metadata_enrichment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Added crystal-system metadata to {processed_dir}")
    for item in summaries:
        print(f"- {item['split']}: {item['sample_count']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
