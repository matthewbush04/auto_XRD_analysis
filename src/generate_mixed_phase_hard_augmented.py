from __future__ import annotations

import argparse
import json
import shutil
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
    minor_bin_from_fractions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_PHASE_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
DEFAULT_BASELINE_MIXED_DIR = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "mixed_phase"
DEFAULT_DATASET_NAME = "mixed_v3_hard_augmented"
RANDOM_SEED = 3031


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing npz file: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def sample_count(payload: dict[str, np.ndarray]) -> int:
    return int(payload["X"].shape[0])


def infer_minor_bins(component_fractions: np.ndarray) -> np.ndarray:
    bins = np.full(component_fractions.shape[0], "none", dtype="<U16")
    for row_idx, fractions in enumerate(component_fractions):
        bins[row_idx] = minor_bin_from_fractions(np.asarray(fractions, dtype=np.float32))
    return bins


def default_for_key(key: str, count: int, reference: np.ndarray | None = None) -> np.ndarray:
    if key == "hard_eval_type":
        return np.full(count, "baseline_random", dtype="<U32")
    if key == "minor_fraction_bin":
        raise KeyError("minor_fraction_bin should be inferred from component_fractions.")
    if key == "component_pair_similarity":
        return np.full(count, np.nan, dtype=np.float32)
    if key == "overlap_rank":
        return np.full(count, -1, dtype=np.int64)
    if key == "overlap_percentile":
        return np.full(count, np.nan, dtype=np.float32)
    if key == "overlap_bin":
        return np.full(count, "none", dtype="<U24")
    if key == "battery_scenario":
        return np.full(count, "none", dtype="<U64")
    if reference is None:
        raise KeyError(f"No default rule for missing key: {key}")
    shape = (count,) + tuple(reference.shape[1:])
    if np.issubdtype(reference.dtype, np.integer):
        fill = -1 if key.startswith("component_") else 0
        return np.full(shape, fill, dtype=reference.dtype)
    if np.issubdtype(reference.dtype, np.floating):
        fill = np.nan if key.startswith("component_") else 0.0
        return np.full(shape, fill, dtype=reference.dtype)
    return np.full(shape, "", dtype=reference.dtype)


def is_per_sample_array(value: np.ndarray, count: int) -> bool:
    return value.ndim >= 1 and int(value.shape[0]) == count


def get_per_sample(payload: dict[str, np.ndarray], key: str, reference: np.ndarray | None = None) -> np.ndarray:
    count = sample_count(payload)
    if key == "minor_fraction_bin" and key not in payload:
        return infer_minor_bins(np.asarray(payload["component_fractions"], dtype=np.float32))
    if key in payload and is_per_sample_array(payload[key], count):
        return payload[key]
    return default_for_key(key, count, reference)


def concatenate_payloads(base: dict[str, np.ndarray], hard_parts: list[dict[str, np.ndarray]], seed: int) -> dict[str, np.ndarray]:
    all_payloads = [base] + hard_parts
    counts = [sample_count(payload) for payload in all_payloads]
    keys: set[str] = set()
    for payload in all_payloads:
        for key, value in payload.items():
            if is_per_sample_array(value, sample_count(payload)):
                keys.add(key)
    keys.update(
        {
            "hard_eval_type",
            "minor_fraction_bin",
            "component_pair_similarity",
            "overlap_rank",
            "overlap_percentile",
            "overlap_bin",
            "battery_scenario",
        }
    )

    merged: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        if key in {"labels", "two_theta"}:
            continue
        reference = next((payload[key] for payload in all_payloads if key in payload and is_per_sample_array(payload[key], sample_count(payload))), None)
        arrays = [get_per_sample(payload, key, reference) for payload in all_payloads]
        merged[key] = np.concatenate(arrays, axis=0)

    merged["labels"] = base["labels"]
    merged["two_theta"] = base["two_theta"]
    if "y_multi" in merged:
        merged["y_presence"] = merged["y_multi"]
    if "component_phase_indices" in merged:
        merged["component_label_indices"] = merged["component_phase_indices"]
    if "component_effective_fractions" in merged:
        merged["component_fractions"] = merged["component_effective_fractions"]
    if "component_nominal_fractions" in merged:
        merged["component_mixing_coefficients"] = merged["component_nominal_fractions"]
    if "component_phase_labels" in merged:
        merged["component_labels"] = merged["component_phase_labels"]
    if "n_phases" in merged:
        merged["num_phases"] = merged["n_phases"]
    merged["generation_seed"] = np.asarray(seed, dtype=np.int64)
    merged["augmentation_source_counts"] = np.asarray(counts, dtype=np.int64)
    return merged


def split_counts(total: int, minor_fraction: float, overlap_fraction: float) -> dict[str, int]:
    minor = int(round(total * minor_fraction))
    overlap = int(round(total * overlap_fraction))
    battery = int(total - minor - overlap)
    if min(minor, overlap, battery) < 0:
        raise ValueError("Hard sample proportions produced a negative count.")
    return {"minor": minor, "overlap": overlap, "battery": battery}


def generate_hard_parts(
    single_phase_dir: Path,
    source_split: str,
    output_dir: Path,
    counts: dict[str, int],
    seed: int,
    mix_kwargs: dict[str, Any],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    source = load_single_split(single_phase_dir / f"{source_split}.npz")
    hard_parts: list[dict[str, np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    if counts["minor"] > 0:
        path = output_dir / f"{source_split}_hard_minor.npz"
        summary = generate_test_minor(source, path, counts["minor"], seed + 11, mix_kwargs)
        summary["path"] = f"embedded:{source_split}_hard_minor"
        summaries.append(summary)
        hard_parts.append(load_npz(path))
    if counts["overlap"] > 0:
        path = output_dir / f"{source_split}_hard_overlap.npz"
        summary = generate_test_overlap(source, path, counts["overlap"], seed + 22, mix_kwargs)
        summary["path"] = f"embedded:{source_split}_hard_overlap"
        summaries.append(summary)
        hard_parts.append(load_npz(path))
    if counts["battery"] > 0:
        path = output_dir / f"{source_split}_hard_battery_relevant.npz"
        summary = generate_test_battery_relevant(source, path, counts["battery"], seed + 33, mix_kwargs)
        summary["path"] = f"embedded:{source_split}_hard_battery_relevant"
        summaries.append(summary)
        hard_parts.append(load_npz(path))
    return hard_parts, summaries


def save_augmented_split(
    split_name: str,
    baseline_path: Path,
    single_phase_dir: Path,
    output_path: Path,
    hard_count: int,
    proportions: dict[str, float],
    seed: int,
    mix_kwargs: dict[str, Any],
    temp_dir: Path,
) -> dict[str, Any]:
    base = load_npz(baseline_path)
    counts = split_counts(hard_count, proportions["minor"], proportions["overlap"])
    hard_parts, hard_summaries = generate_hard_parts(single_phase_dir, split_name, temp_dir, counts, seed, mix_kwargs)
    merged = concatenate_payloads(base, hard_parts, seed)
    np.savez_compressed(output_path, **merged)
    n_phases = np.asarray(merged["n_phases"])
    hard_eval_type = np.asarray(merged["hard_eval_type"]).astype(str)
    return {
        "split": split_name,
        "path": str(output_path),
        "baseline_path": str(baseline_path),
        "baseline_count": sample_count(base),
        "hard_added_count": int(hard_count),
        "sample_count": int(sample_count(merged)),
        "X_shape": list(merged["X"].shape),
        "n_phase_counts": {str(k): int(v) for k, v in zip(*np.unique(n_phases, return_counts=True))},
        "hard_eval_type_counts": {str(k): int(v) for k, v in zip(*np.unique(hard_eval_type, return_counts=True))},
        "hard_component_counts": counts,
        "hard_component_summaries": hard_summaries,
    }


def copy_baseline_test_split(split_name: str, baseline_path: Path, output_path: Path) -> dict[str, Any]:
    shutil.copy2(baseline_path, output_path)
    payload = load_npz(output_path)
    return {
        "split": split_name,
        "path": str(output_path),
        "baseline_path": str(baseline_path),
        "sample_count": sample_count(payload),
        "X_shape": list(payload["X"].shape),
        "copied_from_baseline": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mixed_v1 plus targeted hard-augmentation training dataset.")
    parser.add_argument("--single-phase-dir", type=Path, default=DEFAULT_SINGLE_PHASE_DIR)
    parser.add_argument("--baseline-mixed-dir", type=Path, default=DEFAULT_BASELINE_MIXED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-hard-count", type=int, default=12_000)
    parser.add_argument("--val-hard-count", type=int, default=1_600)
    parser.add_argument("--minor-proportion", type=float, default=0.50)
    parser.add_argument("--overlap-proportion", type=float, default=0.35)
    parser.add_argument("--battery-proportion", type=float, default=0.15)
    parser.add_argument("--max-component-shift-bins", type=int, default=1)
    parser.add_argument("--max-global-shift-bins", type=int, default=1)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.03)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.012)
    parser.add_argument("--background-order", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.output_root / args.dataset_name
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing hard-augmented dataset directory. "
            "Choose a new --dataset-name or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_proportion = args.minor_proportion + args.overlap_proportion + args.battery_proportion
        if not np.isclose(total_proportion, 1.0, atol=1e-6):
            raise ValueError("Hard augmentation proportions must sum to 1.")
        proportions = {
            "minor": float(args.minor_proportion),
            "overlap": float(args.overlap_proportion),
            "battery": float(args.battery_proportion),
        }
        mix_kwargs = {
            "max_component_shift_bins": int(args.max_component_shift_bins),
            "max_global_shift_bins": int(args.max_global_shift_bins),
            "background_range": (float(args.background_min), float(args.background_max)),
            "noise_range": (float(args.noise_min), float(args.noise_max)),
            "background_order": int(args.background_order),
        }
        summaries: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="mixed_v3_hard_parts_") as temp_name:
            temp_dir = Path(temp_name)
            summaries.append(
                save_augmented_split(
                    "train",
                    args.baseline_mixed_dir / "train.npz",
                    args.single_phase_dir,
                    output_dir / "train.npz",
                    int(args.train_hard_count),
                    proportions,
                    int(args.seed + 100),
                    mix_kwargs,
                    temp_dir,
                )
            )
            summaries.append(
                save_augmented_split(
                    "val",
                    args.baseline_mixed_dir / "val.npz",
                    args.single_phase_dir,
                    output_dir / "val.npz",
                    int(args.val_hard_count),
                    proportions,
                    int(args.seed + 200),
                    mix_kwargs,
                    temp_dir,
                )
            )
        for split_name in ("test_normal", "test_hard"):
            summaries.append(copy_baseline_test_split(split_name, args.baseline_mixed_dir / f"{split_name}.npz", output_dir / f"{split_name}.npz"))

        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_name": args.dataset_name,
            "single_phase_dir": str(args.single_phase_dir),
            "baseline_mixed_dir": str(args.baseline_mixed_dir),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "max_components": MAX_COMPONENTS,
            "augmentation_policy": {
                "train_hard_count": int(args.train_hard_count),
                "val_hard_count": int(args.val_hard_count),
                "hard_proportions": proportions,
                "minor_nominal_scenarios": ["95/5", "90/10", "85/15", "80/10/10", "85/10/5"],
                "overlap_pair_selection": "cosine similarity among per-phase mean spectra, sampled from top 10 percent",
                "battery_scenarios": [
                    "cathode_reference",
                    "anode_reference",
                    "solid_electrolyte_reference",
                    "cathode_solid_electrolyte_reference",
                    "anode_solid_electrolyte_reference",
                ],
                "augmentation": {
                    "max_component_shift_bins": int(args.max_component_shift_bins),
                    "max_global_shift_bins": int(args.max_global_shift_bins),
                    "background_range": [float(args.background_min), float(args.background_max)],
                    "noise_range": [float(args.noise_min), float(args.noise_max)],
                    "background_order": int(args.background_order),
                },
            },
            "split_summaries": summaries,
        }
        write_json(output_dir / "dataset_manifest.json", manifest)
        print(f"Saved hard-augmented mixed-phase dataset to {output_dir}")
        for summary in summaries:
            print(f"  {summary['split']}: {summary['X_shape']} -> {summary['path']}")
    except Exception as exc:
        print(f"Hard-augmented dataset generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
