from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from generate_mixed_phase_dataset import MAX_COMPONENTS, load_single_split
from generate_mixed_phase_hard_eval import (
    generate_test_battery_relevant,
    generate_test_minor,
    generate_test_overlap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_PHASE_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "mixed_phase"
DEFAULT_DATASET_NAME = "mixed_v4_active_pool_round1"
DEFAULT_SOURCE_SPLIT = "train"
RANDOM_SEED = 9001


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def sample_count(payload: dict[str, np.ndarray]) -> int:
    return int(np.asarray(payload["X"]).shape[0])


def is_sample_array(value: np.ndarray, count: int) -> bool:
    return value.ndim > 0 and int(value.shape[0]) == count


def concatenate_payloads(parts: list[dict[str, np.ndarray]], seed: int) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("At least one pool part is required.")
    counts = [sample_count(part) for part in parts]
    keys: set[str] = set()
    for part in parts:
        keys.update(part)

    merged: dict[str, np.ndarray] = {}
    reference = parts[0]
    for key in sorted(keys):
        values: list[np.ndarray] = []
        all_sample_level = True
        for part, count in zip(parts, counts):
            if key not in part or not is_sample_array(np.asarray(part[key]), count):
                all_sample_level = False
                break
            values.append(np.asarray(part[key]))
        if all_sample_level:
            merged[key] = np.concatenate(values, axis=0)
        elif key in reference:
            merged[key] = np.asarray(reference[key])

    total = sample_count(merged)
    order = np.random.default_rng(seed).permutation(total)
    for key, value in list(merged.items()):
        if is_sample_array(np.asarray(value), total):
            merged[key] = np.asarray(value)[order]

    if "y_multi" in merged:
        merged["y_presence"] = merged["y_multi"]
    if "component_effective_fractions" in merged:
        merged["component_fractions"] = merged["component_effective_fractions"]
    if "component_nominal_fractions" in merged:
        merged["component_mixing_coefficients"] = merged["component_nominal_fractions"]
    if "component_phase_labels" in merged:
        merged["component_labels"] = merged["component_phase_labels"]
    if "component_phase_indices" in merged:
        merged["component_label_indices"] = merged["component_phase_indices"]
    if "n_phases" in merged:
        merged["num_phases"] = merged["n_phases"]
    merged["pool_sample_index"] = np.arange(total, dtype=np.int64)
    merged["generation_seed"] = np.asarray(seed, dtype=np.int64)
    merged["pool_part_counts"] = np.asarray(counts, dtype=np.int64)
    return merged


def split_counts(pool_size: int, minor_fraction: float, overlap_fraction: float) -> dict[str, int]:
    minor = int(round(pool_size * minor_fraction))
    overlap = int(round(pool_size * overlap_fraction))
    battery = int(pool_size - minor - overlap)
    if min(minor, overlap, battery) < 0:
        raise ValueError("Pool fractions produced a negative count.")
    return {"minor": minor, "overlap": overlap, "battery": battery}


def unique_counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(keys, counts)}


def summarize_pool(payload: dict[str, np.ndarray], part_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": sample_count(payload),
        "X_shape": list(np.asarray(payload["X"]).shape),
        "part_summaries": part_summaries,
    }
    for key in ("n_phases", "hard_eval_type", "minor_fraction_bin", "overlap_bin", "battery_scenario"):
        if key in payload:
            summary[f"{key}_counts"] = unique_counts(np.asarray(payload[key]))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hidden-label mixed-phase candidate pool for synthetic-oracle active learning."
    )
    parser.add_argument("--single-phase-dir", type=Path, default=DEFAULT_SINGLE_PHASE_DIR)
    parser.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pool-size", type=int, default=20_000)
    parser.add_argument("--minor-fraction", type=float, default=0.50)
    parser.add_argument("--overlap-fraction", type=float, default=0.30)
    parser.add_argument("--battery-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-component-shift-bins", type=int, default=1)
    parser.add_argument("--max-global-shift-bins", type=int, default=1)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.03)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.012)
    parser.add_argument("--background-order", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.output_root / args.dataset_name
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing active-learning pool directory. "
            "Choose a new --dataset-name or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_fraction = args.minor_fraction + args.overlap_fraction + args.battery_fraction
        if not np.isclose(total_fraction, 1.0, atol=1e-6):
            raise ValueError("Pool fractions must sum to 1.")
        if args.pool_size <= 0:
            raise ValueError("--pool-size must be positive.")

        source_path = args.single_phase_dir / f"{args.source_split}.npz"
        source = load_single_split(source_path)
        counts = split_counts(int(args.pool_size), float(args.minor_fraction), float(args.overlap_fraction))
        mix_kwargs = {
            "max_component_shift_bins": int(args.max_component_shift_bins),
            "max_global_shift_bins": int(args.max_global_shift_bins),
            "background_range": (float(args.background_min), float(args.background_max)),
            "noise_range": (float(args.noise_min), float(args.noise_max)),
            "background_order": int(args.background_order),
        }

        with tempfile.TemporaryDirectory(prefix="active_pool_parts_") as temp_name:
            temp_dir = Path(temp_name)
            part_summaries: list[dict[str, Any]] = []
            parts: list[dict[str, np.ndarray]] = []

            if counts["minor"] > 0:
                path = temp_dir / "pool_minor.npz"
                summary = generate_test_minor(source, path, counts["minor"], int(args.seed) + 101, mix_kwargs)
                summary["pool_part"] = "minor"
                part_summaries.append(summary)
                parts.append(load_npz(path))
            if counts["overlap"] > 0:
                path = temp_dir / "pool_overlap.npz"
                summary = generate_test_overlap(source, path, counts["overlap"], int(args.seed) + 202, mix_kwargs)
                summary["pool_part"] = "overlap"
                part_summaries.append(summary)
                parts.append(load_npz(path))
            if counts["battery"] > 0:
                path = temp_dir / "pool_battery_relevant.npz"
                summary = generate_test_battery_relevant(source, path, counts["battery"], int(args.seed) + 303, mix_kwargs)
                summary["pool_part"] = "battery"
                part_summaries.append(summary)
                parts.append(load_npz(path))

            payload = concatenate_payloads(parts, int(args.seed) + 404)

        pool_path = output_dir / "pool.npz"
        np.savez_compressed(pool_path, **payload)
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_name": args.dataset_name,
            "purpose": "hidden-label candidate pool for synthetic-oracle active learning simulation",
            "source_path": str(source_path),
            "source_split": args.source_split,
            "output_dir": str(output_dir),
            "pool_path": str(pool_path),
            "seed": int(args.seed),
            "max_components": MAX_COMPONENTS,
            "pool_policy": {
                "pool_size": int(args.pool_size),
                "counts": counts,
                "fractions": {
                    "minor": float(args.minor_fraction),
                    "overlap": float(args.overlap_fraction),
                    "battery": float(args.battery_fraction),
                },
                "note": "Ground-truth labels are stored but must not be used during candidate scoring.",
                "augmentation": {
                    "max_component_shift_bins": int(args.max_component_shift_bins),
                    "max_global_shift_bins": int(args.max_global_shift_bins),
                    "background_range": [float(args.background_min), float(args.background_max)],
                    "noise_range": [float(args.noise_min), float(args.noise_max)],
                    "background_order": int(args.background_order),
                },
            },
            "pool_summary": summarize_pool(payload, part_summaries),
        }
        write_json(output_dir / "dataset_manifest.json", manifest)
        print(f"Saved active-learning candidate pool: {pool_path}")
        print(f"Samples: {sample_count(payload)}")
    except Exception as exc:
        print(f"Active-learning pool generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
