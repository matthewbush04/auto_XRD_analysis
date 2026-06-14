from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_PHASE_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "mixed_phase"
DEFAULT_DATASET_NAME = "mixed_v1"
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "val", "test_normal", "test_hard")
DEFAULT_SPLIT_COUNTS = {
    "train": 31_050,
    "val": 4_140,
    "test_normal": 4_140,
    "test_hard": 4_140,
}
MAX_COMPONENTS = 3
REQUIRED_FIELDS = ["X", "y", "labels", "two_theta"]
FRACTION_LEVEL_LOW_MAX = 0.25
FRACTION_LEVEL_MEDIUM_MAX = 0.60
OPTIONAL_COMPONENT_FIELDS = {
    "sample_material_ids": "component_material_ids",
    "sample_record_ids": "component_record_ids",
    "sample_categories": "component_categories",
    "sample_subclasses": "component_subclasses",
    "sample_cif_paths": "component_cif_paths",
    "sample_crystal_systems": "component_crystal_systems",
    "sample_spacegroup_symbols": "component_spacegroup_symbols",
    "sample_spacegroup_numbers": "component_spacegroup_numbers",
}


def normalize_max(x: np.ndarray) -> np.ndarray:
    max_value = float(np.max(x))
    if max_value <= 1e-8:
        return x.astype(np.float32)
    return (x / max_value).astype(np.float32)


def random_polynomial_background(rng: np.random.Generator, length: int, max_level: float, order: int) -> np.ndarray:
    if max_level <= 0:
        return np.zeros(length, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    coeffs = rng.normal(0.0, 1.0, size=order + 1).astype(np.float32)
    coeffs[0] = abs(coeffs[0]) + 0.5
    background = np.polynomial.polynomial.polyval(x, coeffs).astype(np.float32)
    background = background - np.min(background)
    peak = float(np.max(background))
    if peak > 1e-8:
        background = background / peak
    return (background * max_level).astype(np.float32)


def shift_spectrum(spectrum: np.ndarray, shift_bins: int) -> np.ndarray:
    if shift_bins == 0:
        return spectrum.astype(np.float32)
    shifted = np.roll(spectrum, shift_bins).astype(np.float32)
    if shift_bins > 0:
        shifted[:shift_bins] = 0.0
    else:
        shifted[shift_bins:] = 0.0
    return shifted


def sample_fractions(
    rng: np.random.Generator,
    n_phases: int,
    major_range: tuple[float, float],
    minor_alpha: float,
    min_fraction: float,
) -> np.ndarray:
    if n_phases == 1:
        return np.ones(1, dtype=np.float32)

    max_major_allowed = 1.0 - (n_phases - 1) * min_fraction
    low = max(float(major_range[0]), 1.0 / n_phases)
    high = min(float(major_range[1]), max_major_allowed)
    if low > high:
        raise ValueError(
            f"Incompatible fraction settings: n_phases={n_phases}, "
            f"major_range={major_range}, min_fraction={min_fraction}"
        )

    for _ in range(1000):
        major = float(rng.uniform(low, high))
        remainder = 1.0 - major
        minor = rng.dirichlet(np.full(n_phases - 1, minor_alpha, dtype=np.float32)) * remainder
        fractions = np.concatenate([[major], minor]).astype(np.float32)
        fractions = fractions / np.sum(fractions)
        if float(np.min(fractions)) >= min_fraction:
            return fractions
    raise RuntimeError(f"Could not sample {n_phases}-phase fractions with min_fraction={min_fraction}.")


def fraction_level(fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    if fraction <= FRACTION_LEVEL_LOW_MAX:
        return 1
    if fraction <= FRACTION_LEVEL_MEDIUM_MAX:
        return 2
    return 3


def sample_phase_count(rng: np.random.Generator, p_single: float, p_two_phase: float) -> int:
    draw = float(rng.random())
    if draw < p_single:
        return 1
    if draw < p_single + p_two_phase:
        return 2
    return 3


def choose_component_samples(
    rng: np.random.Generator,
    class_to_indices: dict[int, np.ndarray],
    num_classes: int,
    n_phases: int,
) -> tuple[np.ndarray, np.ndarray]:
    available_classes = np.array(sorted(class_to_indices), dtype=np.int64)
    if len(available_classes) < n_phases:
        raise ValueError(f"Need at least {n_phases} classes, found {len(available_classes)}.")
    phase_indices = rng.choice(available_classes, size=n_phases, replace=False)
    source_indices = np.array(
        [rng.choice(class_to_indices[int(phase_idx)]) for phase_idx in phase_indices],
        dtype=np.int64,
    )
    if np.any(phase_indices < 0) or np.any(phase_indices >= num_classes):
        raise ValueError("Sampled phase index is out of bounds.")
    return phase_indices.astype(np.int64), source_indices


def make_mixture(
    X_source: np.ndarray,
    source_indices: np.ndarray,
    fractions: np.ndarray,
    rng: np.random.Generator,
    max_component_shift_bins: int,
    max_global_shift_bins: int,
    background_range: tuple[float, float],
    noise_range: tuple[float, float],
    background_order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    length = int(X_source.shape[1])
    mixed = np.zeros(length, dtype=np.float32)
    component_shift_bins: list[int] = []
    component_scales: list[float] = []
    for source_idx, fraction in zip(source_indices, fractions):
        component = normalize_max(X_source[int(source_idx)])
        component_shift = 0
        if max_component_shift_bins > 0:
            component_shift = int(rng.integers(-max_component_shift_bins, max_component_shift_bins + 1))
            component = shift_spectrum(component, component_shift)
        intensity_scale = float(rng.uniform(0.90, 1.10))
        mixed += float(fraction) * intensity_scale * component
        component_shift_bins.append(component_shift)
        component_scales.append(intensity_scale)

    global_shift = 0
    if max_global_shift_bins > 0:
        global_shift = int(rng.integers(-max_global_shift_bins, max_global_shift_bins + 1))
        mixed = shift_spectrum(mixed, global_shift)

    background_level = float(rng.uniform(*background_range))
    mixed += random_polynomial_background(rng, length, background_level, background_order)

    noise_level = float(rng.uniform(*noise_range))
    if noise_level > 0:
        mixed += rng.normal(0.0, noise_level, size=length).astype(np.float32)

    params = {
        "component_shift_bins": component_shift_bins,
        "component_intensity_scales": component_scales,
        "global_shift_bins": global_shift,
        "background_level": background_level,
        "noise_level": noise_level,
    }
    return normalize_max(np.clip(mixed, 0.0, None)), params


def load_single_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    data = np.load(path, allow_pickle=True)
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    loaded: dict[str, Any] = {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "labels": np.asarray(data["labels"]),
        "two_theta": data["two_theta"].astype(np.float32),
    }
    for field in OPTIONAL_COMPONENT_FIELDS:
        if field in data.files:
            loaded[field] = np.asarray(data[field])
    return loaded


def make_component_field_storage(source: dict[str, Any], count: int) -> dict[str, np.ndarray]:
    storage: dict[str, np.ndarray] = {}
    for source_field, output_field in OPTIONAL_COMPONENT_FIELDS.items():
        if source_field not in source:
            continue
        values = np.asarray(source[source_field])
        if np.issubdtype(values.dtype, np.integer):
            storage[output_field] = np.full((count, MAX_COMPONENTS), -1, dtype=values.dtype)
        elif np.issubdtype(values.dtype, np.floating):
            storage[output_field] = np.full((count, MAX_COMPONENTS), np.nan, dtype=values.dtype)
        else:
            max_len = max(1, max(len(str(value)) for value in values))
            storage[output_field] = np.full((count, MAX_COMPONENTS), "", dtype=f"<U{max_len}")
    return storage


def fill_component_field_storage(
    storage: dict[str, np.ndarray],
    source: dict[str, Any],
    sample_idx: int,
    source_indices: np.ndarray,
) -> None:
    n_phases = len(source_indices)
    for source_field, output_field in OPTIONAL_COMPONENT_FIELDS.items():
        if source_field not in source or output_field not in storage:
            continue
        storage[output_field][sample_idx, :n_phases] = np.asarray(source[source_field])[source_indices]


def build_class_index(y: np.ndarray) -> dict[int, np.ndarray]:
    class_to_indices: dict[int, np.ndarray] = {}
    for class_idx in np.unique(y):
        class_to_indices[int(class_idx)] = np.flatnonzero(y == int(class_idx)).astype(np.int64)
    return class_to_indices


def generate_split(
    split_name: str,
    source_path: Path,
    output_path: Path,
    count: int,
    seed: int,
    p_single: float,
    p_two_phase: float,
    major_fraction_range: tuple[float, float],
    minor_alpha: float,
    min_binary_fraction: float,
    min_ternary_fraction: float,
    max_component_shift_bins: int,
    max_global_shift_bins: int,
    background_range: tuple[float, float],
    noise_range: tuple[float, float],
    background_order: int,
) -> dict[str, Any]:
    source = load_single_split(source_path)
    X_source = source["X"]
    y_source = source["y"]
    labels = source["labels"]
    two_theta = source["two_theta"]
    num_classes = int(len(labels))
    length = int(X_source.shape[1])
    class_to_indices = build_class_index(y_source)
    rng = np.random.default_rng(seed)

    X_mixed = np.zeros((count, length), dtype=np.float32)
    y_multi = np.zeros((count, num_classes), dtype=np.float32)
    y_fraction = np.zeros((count, num_classes), dtype=np.float32)
    y_fraction_level = np.zeros((count, num_classes), dtype=np.int64)
    component_phase_indices = np.full((count, MAX_COMPONENTS), -1, dtype=np.int64)
    component_source_indices = np.full((count, MAX_COMPONENTS), -1, dtype=np.int64)
    component_nominal_fractions = np.zeros((count, MAX_COMPONENTS), dtype=np.float32)
    component_effective_fractions = np.zeros((count, MAX_COMPONENTS), dtype=np.float32)
    component_phase_labels = np.full((count, MAX_COMPONENTS), "", dtype=f"<U{max(len(str(label)) for label in labels)}")
    component_shift_bins = np.zeros((count, MAX_COMPONENTS), dtype=np.int64)
    component_intensity_scales = np.ones((count, MAX_COMPONENTS), dtype=np.float32)
    component_metadata = make_component_field_storage(source, count)
    global_shift_bins = np.zeros(count, dtype=np.int64)
    background_levels = np.zeros(count, dtype=np.float32)
    noise_levels = np.zeros(count, dtype=np.float32)
    n_phases_array = np.zeros(count, dtype=np.int64)
    major_phase_index = np.zeros(count, dtype=np.int64)
    major_fraction = np.zeros(count, dtype=np.float32)

    for sample_idx in range(count):
        accepted = False
        for _ in range(200):
            n_phases = sample_phase_count(rng, p_single, p_two_phase)
            phase_indices, source_indices = choose_component_samples(rng, class_to_indices, num_classes, n_phases)
            min_fraction = min_binary_fraction if n_phases == 2 else min_ternary_fraction
            fractions = sample_fractions(rng, n_phases, major_fraction_range, minor_alpha, min_fraction)

            order = np.argsort(-fractions)
            phase_indices = phase_indices[order]
            source_indices = source_indices[order]
            fractions = fractions[order]

            mixed_spectrum, augmentation_params = make_mixture(
                X_source,
                source_indices,
                fractions,
                rng,
                max_component_shift_bins=max_component_shift_bins,
                max_global_shift_bins=max_global_shift_bins,
                background_range=background_range,
                noise_range=noise_range,
                background_order=background_order,
            )
            scales = np.asarray(augmentation_params["component_intensity_scales"], dtype=np.float32)
            effective_fractions = fractions * scales
            effective_fractions = (effective_fractions / np.sum(effective_fractions)).astype(np.float32)
            if float(np.min(effective_fractions)) >= min_fraction:
                accepted = True
                break
        if not accepted:
            raise RuntimeError(
                f"Could not generate valid effective fractions for sample {sample_idx} "
                f"after repeated attempts."
            )

        X_mixed[sample_idx] = mixed_spectrum
        effective_order = np.argsort(-effective_fractions)
        phase_indices = phase_indices[effective_order]
        source_indices = source_indices[effective_order]
        nominal_fractions = fractions[effective_order]
        effective_fractions = effective_fractions[effective_order]
        component_shifts = np.asarray(augmentation_params["component_shift_bins"], dtype=np.int64)[effective_order]
        scales = scales[effective_order]

        y_multi[sample_idx, phase_indices] = 1.0
        y_fraction[sample_idx, phase_indices] = effective_fractions
        for phase_idx, fraction in zip(phase_indices, effective_fractions):
            y_fraction_level[sample_idx, int(phase_idx)] = fraction_level(float(fraction))
        component_phase_indices[sample_idx, :n_phases] = phase_indices
        component_source_indices[sample_idx, :n_phases] = source_indices
        component_nominal_fractions[sample_idx, :n_phases] = nominal_fractions
        component_effective_fractions[sample_idx, :n_phases] = effective_fractions
        component_phase_labels[sample_idx, :n_phases] = labels[phase_indices].astype(str)
        component_shift_bins[sample_idx, :n_phases] = component_shifts
        component_intensity_scales[sample_idx, :n_phases] = scales
        fill_component_field_storage(component_metadata, source, sample_idx, source_indices)
        global_shift_bins[sample_idx] = int(augmentation_params["global_shift_bins"])
        background_levels[sample_idx] = float(augmentation_params["background_level"])
        noise_levels[sample_idx] = float(augmentation_params["noise_level"])
        n_phases_array[sample_idx] = n_phases
        major_phase_index[sample_idx] = int(phase_indices[0])
        major_fraction[sample_idx] = float(effective_fractions[0])

    save_payload = {
        "X": X_mixed,
        "y_multi": y_multi,
        "y_presence": y_multi,
        "y_fraction": y_fraction,
        "y_fraction_level": y_fraction_level,
        "labels": labels,
        "two_theta": two_theta,
        "component_phase_indices": component_phase_indices,
        "component_label_indices": component_phase_indices,
        "component_source_indices": component_source_indices,
        "component_fractions": component_effective_fractions,
        "component_effective_fractions": component_effective_fractions,
        "component_nominal_fractions": component_nominal_fractions,
        "component_mixing_coefficients": component_nominal_fractions,
        "component_phase_labels": component_phase_labels,
        "component_labels": component_phase_labels,
        "component_shift_bins": component_shift_bins,
        "component_intensity_scales": component_intensity_scales,
        "global_shift_bins": global_shift_bins,
        "background_levels": background_levels,
        "noise_levels": noise_levels,
        "n_phases": n_phases_array,
        "num_phases": n_phases_array,
        "major_phase_index": major_phase_index,
        "major_fraction": major_fraction,
        "source_split": np.asarray([split_name] * count),
        "generation_seed": np.asarray(seed, dtype=np.int64),
        "mixing_seed": np.asarray(seed, dtype=np.int64),
    }
    save_payload.update(component_metadata)
    np.savez_compressed(output_path, **save_payload)

    minor_values = component_effective_fractions[:, 1:][component_effective_fractions[:, 1:] > 0]
    nominal_minor_values = component_nominal_fractions[:, 1:][component_nominal_fractions[:, 1:] > 0]
    return {
        "split": split_name,
        "path": str(output_path),
        "source_path": str(source_path),
        "sample_count": int(count),
        "X_shape": list(X_mixed.shape),
        "num_classes": num_classes,
        "n_phase_counts": {str(k): int(v) for k, v in zip(*np.unique(n_phases_array, return_counts=True))},
        "mean_major_effective_fraction": float(np.mean(major_fraction)),
        "mean_minor_effective_fraction": float(np.mean(minor_values)) if minor_values.size > 0 else 0.0,
        "mean_major_nominal_fraction": float(np.mean(component_nominal_fractions[:, 0])),
        "mean_minor_nominal_fraction": float(np.mean(nominal_minor_values)) if nominal_minor_values.size > 0 else 0.0,
        "component_metadata_fields": sorted(component_metadata.keys()),
    }


def parse_count_overrides(values: list[str] | None) -> dict[str, int]:
    counts = dict(DEFAULT_SPLIT_COUNTS)
    if not values:
        return counts
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --split-count value: {item}. Expected split=count.")
        split, raw_count = item.split("=", 1)
        split = split.strip()
        if split not in counts:
            raise ValueError(f"Unknown split in --split-count: {split}")
        counts[split] = int(raw_count)
    return counts


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed synthetic mixed-phase XRD datasets.")
    parser.add_argument("--single-phase-dir", type=Path, default=DEFAULT_SINGLE_PHASE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--split-count", action="append", default=None, help="Override count, e.g. train=10000")
    parser.add_argument("--p-single", type=float, default=0.10)
    parser.add_argument("--p-two-phase", type=float, default=0.60)
    parser.add_argument("--major-fraction-min", type=float, default=0.60)
    parser.add_argument("--major-fraction-max", type=float, default=0.95)
    parser.add_argument("--minor-alpha", type=float, default=1.0)
    parser.add_argument("--min-binary-fraction", type=float, default=0.10)
    parser.add_argument("--min-ternary-fraction", type=float, default=0.08)
    parser.add_argument("--max-component-shift-bins", type=int, default=0)
    parser.add_argument("--max-global-shift-bins", type=int, default=1)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.02)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.01)
    parser.add_argument("--background-order", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.output_root / args.dataset_name
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing mixed-phase dataset directory. "
            "Choose a new --dataset-name or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        split_counts = parse_count_overrides(args.split_count)
        major_range = (float(args.major_fraction_min), float(args.major_fraction_max))
        if not (0.0 < major_range[0] <= major_range[1] < 1.0):
            raise ValueError("Major fraction range must satisfy 0 < min <= max < 1.")
        if args.p_single < 0 or args.p_two_phase < 0 or args.p_single + args.p_two_phase > 1:
            raise ValueError("--p-single and --p-two-phase must be nonnegative and sum to <= 1.")
        background_range = (float(args.background_min), float(args.background_max))
        noise_range = (float(args.noise_min), float(args.noise_max))

        summaries = []
        for split_offset, split_name in enumerate(SPLIT_NAMES):
            summaries.append(
                generate_split(
                    split_name=split_name,
                    source_path=args.single_phase_dir / f"{split_name}.npz",
                    output_path=output_dir / f"{split_name}.npz",
                    count=int(split_counts[split_name]),
                    seed=int(args.seed + split_offset * 1009),
                    p_single=float(args.p_single),
                    p_two_phase=float(args.p_two_phase),
                    major_fraction_range=major_range,
                    minor_alpha=float(args.minor_alpha),
                    min_binary_fraction=float(args.min_binary_fraction),
                    min_ternary_fraction=float(args.min_ternary_fraction),
                    max_component_shift_bins=int(args.max_component_shift_bins),
                    max_global_shift_bins=int(args.max_global_shift_bins),
                    background_range=background_range,
                    noise_range=noise_range,
                    background_order=int(args.background_order),
                )
            )

        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_name": args.dataset_name,
            "single_phase_dir": str(args.single_phase_dir),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "max_components": MAX_COMPONENTS,
            "mixture_policy": {
                "p_single": float(args.p_single),
                "p_two_phase": float(args.p_two_phase),
                "p_ternary": float(1.0 - args.p_single - args.p_two_phase),
                "major_fraction_range": list(major_range),
                "minor_alpha": float(args.minor_alpha),
                "min_binary_fraction": float(args.min_binary_fraction),
                "min_ternary_fraction": float(args.min_ternary_fraction),
                "max_component_shift_bins": int(args.max_component_shift_bins),
                "max_global_shift_bins": int(args.max_global_shift_bins),
                "background_range": list(background_range),
                "noise_range": list(noise_range),
                "background_order": int(args.background_order),
                "fraction_level_bins": {
                    "absent": 0,
                    "low": [0.0, FRACTION_LEVEL_LOW_MAX],
                    "medium": [FRACTION_LEVEL_LOW_MAX, FRACTION_LEVEL_MEDIUM_MAX],
                    "high": [FRACTION_LEVEL_MEDIUM_MAX, 1.0],
                },
            },
            "split_summaries": summaries,
        }
        write_json(output_dir / "dataset_manifest.json", manifest)
        print(f"Saved mixed-phase dataset to {output_dir}")
        for summary in summaries:
            print(f"  {summary['split']}: {summary['X_shape']} -> {summary['path']}")
    except Exception as exc:
        print(f"Dataset generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
