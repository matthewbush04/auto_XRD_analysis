from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v3_hard_augmented"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "mixed_phase"
DEFAULT_DATASET_NAME = "mixed_v4_active_round1"
RANDOM_SEED = 42


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_selected_indices(path: Path | None) -> list[int]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_index" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no sample_index column.")
        return [int(row["sample_index"]) for row in reader]


def choose_random_indices(pool_path: Path, count: int, seed: int) -> list[int]:
    data = np.load(pool_path, allow_pickle=True)
    n_samples = int(data["X"].shape[0])
    if count <= 0 or count > n_samples:
        raise ValueError(f"--random-count must be in [1, {n_samples}].")
    rng = np.random.default_rng(seed)
    return sorted(int(idx) for idx in rng.choice(n_samples, size=count, replace=False))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def is_sample_array(value: np.ndarray, count: int) -> bool:
    return value.ndim > 0 and int(value.shape[0]) == count


def concat_or_copy(base: dict[str, np.ndarray], pool: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    base_count = int(base["X"].shape[0])
    pool_count = int(pool["X"].shape[0])
    merged: dict[str, np.ndarray] = {}
    for key, base_value in base.items():
        base_array = np.asarray(base_value)
        if is_sample_array(base_array, base_count):
            if key in pool and is_sample_array(np.asarray(pool[key]), pool_count):
                pool_values = np.asarray(pool[key])[indices]
                merged[key] = np.concatenate([base_array, pool_values], axis=0)
            else:
                merged[key] = base_array
        else:
            merged[key] = base_array

    # Preserve useful pool-only sample metadata when available.
    for key, pool_value in pool.items():
        if key in merged:
            continue
        pool_array = np.asarray(pool_value)
        if is_sample_array(pool_array, pool_count):
            filler_shape = (base_count,) + pool_array.shape[1:]
            if np.issubdtype(pool_array.dtype, np.integer):
                filler = np.full(filler_shape, -1, dtype=pool_array.dtype)
            elif np.issubdtype(pool_array.dtype, np.floating):
                filler = np.full(filler_shape, np.nan, dtype=pool_array.dtype)
            else:
                filler = np.full(filler_shape, "", dtype=pool_array.dtype)
            merged[key] = np.concatenate([filler, pool_array[indices]], axis=0)
    return merged


def copy_eval_splits(base_data_dir: Path, output_dir: Path) -> None:
    for split_name in ("val", "test_normal", "test_hard"):
        src = base_data_dir / f"{split_name}.npz"
        dst = output_dir / f"{split_name}.npz"
        if src.exists():
            shutil.copy2(src, dst)


def unique_counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(keys, counts)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new mixed-phase training dataset for one synthetic-oracle active-learning round."
    )
    parser.add_argument("--base-data-dir", type=Path, default=DEFAULT_BASE_DATA_DIR)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, default=None)
    parser.add_argument("--random-count", type=int, default=None)
    parser.add_argument("--selection-method", choices=["active", "random"], default="active")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.output_root / args.dataset_name
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing active-learning round dataset. "
            "Choose a new --dataset-name or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.selection_method == "active":
            if args.selected_csv is None:
                raise ValueError("--selected-csv is required for active selection.")
            selected_indices = read_selected_indices(args.selected_csv)
        else:
            if args.random_count is None:
                raise ValueError("--random-count is required for random selection.")
            selected_indices = choose_random_indices(args.candidate_pool, int(args.random_count), int(args.seed))

        if not selected_indices:
            raise ValueError("No selected indices were provided.")

        base_train = load_npz(args.base_data_dir / "train.npz")
        pool = load_npz(args.candidate_pool)
        indices = np.asarray(selected_indices, dtype=np.int64)
        merged_train = concat_or_copy(base_train, pool, indices)
        train_path = output_dir / "train.npz"
        np.savez_compressed(train_path, **merged_train)
        copy_eval_splits(args.base_data_dir, output_dir)

        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_name": args.dataset_name,
            "selection_method": args.selection_method,
            "base_dataset": str(args.base_data_dir),
            "candidate_pool": str(args.candidate_pool),
            "selected_csv": str(args.selected_csv) if args.selected_csv else None,
            "random_count": int(args.random_count) if args.random_count is not None else None,
            "seed": int(args.seed),
            "selected_count": int(len(indices)),
            "base_train_count": int(base_train["X"].shape[0]),
            "train_count": int(merged_train["X"].shape[0]),
            "train_path": str(train_path),
            "test_sets_are_not_included_in_selection": True,
            "fixed_eval_splits_copied_from_base": ["val", "test_normal", "test_hard"],
        }
        if "hard_eval_type" in pool:
            manifest["selected_hard_eval_type_counts"] = unique_counts(np.asarray(pool["hard_eval_type"])[indices])
        if "minor_fraction_bin" in pool:
            manifest["selected_minor_fraction_bin_counts"] = unique_counts(np.asarray(pool["minor_fraction_bin"])[indices])
        if "overlap_bin" in pool:
            manifest["selected_overlap_bin_counts"] = unique_counts(np.asarray(pool["overlap_bin"])[indices])
        if "battery_scenario" in pool:
            manifest["selected_battery_scenario_counts"] = unique_counts(np.asarray(pool["battery_scenario"])[indices])
        write_json(output_dir / "dataset_manifest.json", manifest)
        print(f"Saved active-learning round dataset: {output_dir}")
        print(f"Train samples: {manifest['base_train_count']} + {manifest['selected_count']} = {manifest['train_count']}")
    except Exception as exc:
        print(f"Active-learning round build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
