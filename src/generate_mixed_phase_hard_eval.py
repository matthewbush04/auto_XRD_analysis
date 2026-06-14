from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from generate_mixed_phase_dataset import (
    FRACTION_LEVEL_LOW_MAX,
    FRACTION_LEVEL_MEDIUM_MAX,
    MAX_COMPONENTS,
    build_class_index,
    fill_component_field_storage,
    fraction_level,
    load_single_split,
    make_component_field_storage,
    make_mixture,
    normalize_max,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_PHASE_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "mixed_phase"
DEFAULT_DATASET_NAME = "mixed_v2_hard_eval"
DEFAULT_SOURCE_SPLIT = "test_hard"
RANDOM_SEED = 2026
DEFAULT_TEST_COUNT = 4140
TEST_NAMES = ("test_minor", "test_overlap", "test_battery_relevant")


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_source_indices(
    rng: np.random.Generator,
    class_to_indices: dict[int, np.ndarray],
    phase_indices: np.ndarray,
) -> np.ndarray:
    return np.asarray([rng.choice(class_to_indices[int(phase_idx)]) for phase_idx in phase_indices], dtype=np.int64)


def make_category_to_classes(source: dict[str, Any]) -> dict[str, np.ndarray]:
    if "sample_categories" not in source:
        return {}
    y = np.asarray(source["y"], dtype=np.int64)
    categories = np.asarray(source["sample_categories"]).astype(str)
    category_to_classes: dict[str, list[int]] = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        classes = sorted(int(value) for value in np.unique(y[mask]))
        category_to_classes[str(category)] = classes
    return {key: np.asarray(value, dtype=np.int64) for key, value in category_to_classes.items()}


def sample_classes_from_categories(
    rng: np.random.Generator,
    category_to_classes: dict[str, np.ndarray],
    categories: list[str],
) -> np.ndarray:
    selected: list[int] = []
    for category in categories:
        choices = category_to_classes.get(category)
        if choices is None or len(choices) == 0:
            raise ValueError(f"No classes available for category '{category}'.")
        for _ in range(100):
            phase_idx = int(rng.choice(choices))
            if phase_idx not in selected:
                selected.append(phase_idx)
                break
        else:
            raise RuntimeError(f"Could not sample a unique class for categories {categories}.")
    return np.asarray(selected, dtype=np.int64)


def compute_phase_prototypes(source: dict[str, Any]) -> np.ndarray:
    X = np.asarray(source["X"], dtype=np.float32)
    y = np.asarray(source["y"], dtype=np.int64)
    num_classes = int(len(source["labels"]))
    prototypes = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    for phase_idx in range(num_classes):
        mask = y == phase_idx
        if not np.any(mask):
            continue
        prototypes[phase_idx] = normalize_max(np.mean(X[mask], axis=0))
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norms, 1e-8)


def compute_overlap_pairs(source: dict[str, Any]) -> list[dict[str, Any]]:
    prototypes = compute_phase_prototypes(source)
    pairs: list[dict[str, Any]] = []
    num_classes = prototypes.shape[0]
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            similarity = float(np.sum(prototypes[i] * prototypes[j], dtype=np.float64))
            pairs.append({"phase_a": i, "phase_b": j, "similarity": similarity})
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    total = len(pairs)
    for rank, item in enumerate(pairs, start=1):
        percentile = rank / max(total, 1)
        if percentile <= 0.01:
            overlap_bin = "top_1pct"
        elif percentile <= 0.05:
            overlap_bin = "top_5pct"
        elif percentile <= 0.10:
            overlap_bin = "top_10pct"
        else:
            overlap_bin = "below_top_10pct"
        item["overlap_rank"] = rank
        item["overlap_percentile"] = float(percentile)
        item["overlap_bin"] = overlap_bin
    return pairs


def make_storage(source: dict[str, Any], count: int) -> dict[str, Any]:
    labels = np.asarray(source["labels"])
    label_width = max(1, max(len(str(label)) for label in labels))
    storage: dict[str, Any] = {
        "X": np.zeros((count, source["X"].shape[1]), dtype=np.float32),
        "y_multi": np.zeros((count, len(labels)), dtype=np.float32),
        "y_fraction": np.zeros((count, len(labels)), dtype=np.float32),
        "y_fraction_level": np.zeros((count, len(labels)), dtype=np.int64),
        "component_phase_indices": np.full((count, MAX_COMPONENTS), -1, dtype=np.int64),
        "component_source_indices": np.full((count, MAX_COMPONENTS), -1, dtype=np.int64),
        "component_nominal_fractions": np.zeros((count, MAX_COMPONENTS), dtype=np.float32),
        "component_effective_fractions": np.zeros((count, MAX_COMPONENTS), dtype=np.float32),
        "component_phase_labels": np.full((count, MAX_COMPONENTS), "", dtype=f"<U{label_width}"),
        "component_shift_bins": np.zeros((count, MAX_COMPONENTS), dtype=np.int64),
        "component_intensity_scales": np.ones((count, MAX_COMPONENTS), dtype=np.float32),
        "global_shift_bins": np.zeros(count, dtype=np.int64),
        "background_levels": np.zeros(count, dtype=np.float32),
        "noise_levels": np.zeros(count, dtype=np.float32),
        "n_phases": np.zeros(count, dtype=np.int64),
        "major_phase_index": np.zeros(count, dtype=np.int64),
        "major_fraction": np.zeros(count, dtype=np.float32),
        "minor_fraction_bin": np.full(count, "none", dtype="<U16"),
        "hard_eval_type": np.full(count, "", dtype="<U32"),
        "component_pair_similarity": np.full(count, np.nan, dtype=np.float32),
        "overlap_rank": np.full(count, -1, dtype=np.int64),
        "overlap_percentile": np.full(count, np.nan, dtype=np.float32),
        "overlap_bin": np.full(count, "none", dtype="<U24"),
        "battery_scenario": np.full(count, "none", dtype="<U64"),
    }
    storage.update(make_component_field_storage(source, count))
    return storage


def minor_bin_from_fractions(fractions: np.ndarray) -> str:
    positive = fractions[fractions > 0]
    if len(positive) <= 1:
        return "none"
    minor = float(np.min(positive))
    if minor <= 0.075:
        return "minor_5pct"
    if minor <= 0.125:
        return "minor_10pct"
    if minor <= 0.175:
        return "minor_15pct"
    return "minor_gt15pct"


def save_sample(
    storage: dict[str, Any],
    source: dict[str, Any],
    sample_idx: int,
    split_name: str,
    phase_indices: np.ndarray,
    source_indices: np.ndarray,
    nominal_fractions: np.ndarray,
    mixed_spectrum: np.ndarray,
    augmentation_params: dict[str, Any],
    hard_eval_type: str,
    overlap_info: dict[str, Any] | None = None,
    battery_scenario: str = "none",
) -> None:
    labels = np.asarray(source["labels"])
    scales = np.asarray(augmentation_params["component_intensity_scales"], dtype=np.float32)
    effective_fractions = nominal_fractions.astype(np.float32) * scales
    effective_fractions = effective_fractions / np.sum(effective_fractions)
    order = np.argsort(-effective_fractions)

    phase_indices = phase_indices[order]
    source_indices = source_indices[order]
    nominal_fractions = nominal_fractions[order]
    effective_fractions = effective_fractions[order]
    component_shift_bins = np.asarray(augmentation_params["component_shift_bins"], dtype=np.int64)[order]
    scales = scales[order]
    n_phases = len(phase_indices)

    storage["X"][sample_idx] = mixed_spectrum
    storage["y_multi"][sample_idx, phase_indices] = 1.0
    storage["y_fraction"][sample_idx, phase_indices] = effective_fractions
    for phase_idx, fraction in zip(phase_indices, effective_fractions):
        storage["y_fraction_level"][sample_idx, int(phase_idx)] = fraction_level(float(fraction))
    storage["component_phase_indices"][sample_idx, :n_phases] = phase_indices
    storage["component_source_indices"][sample_idx, :n_phases] = source_indices
    storage["component_nominal_fractions"][sample_idx, :n_phases] = nominal_fractions
    storage["component_effective_fractions"][sample_idx, :n_phases] = effective_fractions
    storage["component_phase_labels"][sample_idx, :n_phases] = labels[phase_indices].astype(str)
    storage["component_shift_bins"][sample_idx, :n_phases] = component_shift_bins
    storage["component_intensity_scales"][sample_idx, :n_phases] = scales
    storage["global_shift_bins"][sample_idx] = int(augmentation_params["global_shift_bins"])
    storage["background_levels"][sample_idx] = float(augmentation_params["background_level"])
    storage["noise_levels"][sample_idx] = float(augmentation_params["noise_level"])
    storage["n_phases"][sample_idx] = n_phases
    storage["major_phase_index"][sample_idx] = int(phase_indices[0])
    storage["major_fraction"][sample_idx] = float(effective_fractions[0])
    storage["minor_fraction_bin"][sample_idx] = minor_bin_from_fractions(effective_fractions)
    storage["hard_eval_type"][sample_idx] = hard_eval_type
    storage["battery_scenario"][sample_idx] = battery_scenario
    if overlap_info is not None:
        storage["component_pair_similarity"][sample_idx] = float(overlap_info["similarity"])
        storage["overlap_rank"][sample_idx] = int(overlap_info["overlap_rank"])
        storage["overlap_percentile"][sample_idx] = float(overlap_info["overlap_percentile"])
        storage["overlap_bin"][sample_idx] = str(overlap_info["overlap_bin"])
    fill_component_field_storage(storage, source, sample_idx, source_indices)


def finalize_payload(storage: dict[str, Any], source: dict[str, Any], split_name: str, seed: int) -> dict[str, Any]:
    payload = {
        "X": storage["X"],
        "y_multi": storage["y_multi"],
        "y_presence": storage["y_multi"],
        "y_fraction": storage["y_fraction"],
        "y_fraction_level": storage["y_fraction_level"],
        "labels": np.asarray(source["labels"]),
        "two_theta": np.asarray(source["two_theta"], dtype=np.float32),
        "component_phase_indices": storage["component_phase_indices"],
        "component_label_indices": storage["component_phase_indices"],
        "component_source_indices": storage["component_source_indices"],
        "component_fractions": storage["component_effective_fractions"],
        "component_effective_fractions": storage["component_effective_fractions"],
        "component_nominal_fractions": storage["component_nominal_fractions"],
        "component_mixing_coefficients": storage["component_nominal_fractions"],
        "component_phase_labels": storage["component_phase_labels"],
        "component_labels": storage["component_phase_labels"],
        "component_shift_bins": storage["component_shift_bins"],
        "component_intensity_scales": storage["component_intensity_scales"],
        "global_shift_bins": storage["global_shift_bins"],
        "background_levels": storage["background_levels"],
        "noise_levels": storage["noise_levels"],
        "n_phases": storage["n_phases"],
        "num_phases": storage["n_phases"],
        "major_phase_index": storage["major_phase_index"],
        "major_fraction": storage["major_fraction"],
        "minor_fraction_bin": storage["minor_fraction_bin"],
        "hard_eval_type": storage["hard_eval_type"],
        "component_pair_similarity": storage["component_pair_similarity"],
        "overlap_rank": storage["overlap_rank"],
        "overlap_percentile": storage["overlap_percentile"],
        "overlap_bin": storage["overlap_bin"],
        "battery_scenario": storage["battery_scenario"],
        "source_split": np.asarray([split_name] * storage["X"].shape[0]),
        "generation_seed": np.asarray(seed, dtype=np.int64),
        "mixing_seed": np.asarray(seed, dtype=np.int64),
    }
    for key, value in storage.items():
        if key.startswith("component_") and key not in payload:
            payload[key] = value
    return payload


def generate_test_minor(
    source: dict[str, Any],
    output_path: Path,
    count: int,
    seed: int,
    mix_kwargs: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(source["y"], dtype=np.int64)
    labels = np.asarray(source["labels"])
    class_to_indices = build_class_index(y)
    available_classes = np.asarray(sorted(class_to_indices), dtype=np.int64)
    scenarios = [
        np.asarray([0.95, 0.05], dtype=np.float32),
        np.asarray([0.90, 0.10], dtype=np.float32),
        np.asarray([0.85, 0.15], dtype=np.float32),
        np.asarray([0.80, 0.10, 0.10], dtype=np.float32),
        np.asarray([0.85, 0.10, 0.05], dtype=np.float32),
    ]
    storage = make_storage(source, count)

    for sample_idx in range(count):
        nominal = scenarios[sample_idx % len(scenarios)].copy()
        phase_indices = rng.choice(available_classes, size=len(nominal), replace=False).astype(np.int64)
        source_indices = choose_source_indices(rng, class_to_indices, phase_indices)
        mixed, params = make_mixture(source["X"], source_indices, nominal, rng, **mix_kwargs)
        save_sample(
            storage,
            source,
            sample_idx,
            "test_minor",
            phase_indices,
            source_indices,
            nominal,
            mixed,
            params,
            hard_eval_type="minor_fraction",
        )

    np.savez_compressed(output_path, **finalize_payload(storage, source, "test_minor", seed))
    return summarize_split("test_minor", output_path, storage, len(labels))


def generate_test_overlap(
    source: dict[str, Any],
    output_path: Path,
    count: int,
    seed: int,
    mix_kwargs: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(source["y"], dtype=np.int64)
    labels = np.asarray(source["labels"])
    class_to_indices = build_class_index(y)
    pairs = [pair for pair in compute_overlap_pairs(source) if pair["overlap_bin"] != "below_top_10pct"]
    if not pairs:
        raise RuntimeError("No overlap pairs found in the top 10 percent.")
    by_bin = {
        "top_1pct": [pair for pair in pairs if pair["overlap_bin"] == "top_1pct"],
        "top_5pct": [pair for pair in pairs if pair["overlap_bin"] == "top_5pct"],
        "top_10pct": [pair for pair in pairs if pair["overlap_bin"] == "top_10pct"],
    }
    bins = [key for key, values in by_bin.items() if values]
    storage = make_storage(source, count)

    for sample_idx in range(count):
        overlap_bin = bins[sample_idx % len(bins)]
        pair = by_bin[overlap_bin][int(rng.integers(0, len(by_bin[overlap_bin])))]
        phase_indices = np.asarray([pair["phase_a"], pair["phase_b"]], dtype=np.int64)
        if rng.random() < 0.5:
            phase_indices = phase_indices[::-1].copy()
        major = float(rng.uniform(0.70, 0.90))
        nominal = np.asarray([major, 1.0 - major], dtype=np.float32)
        source_indices = choose_source_indices(rng, class_to_indices, phase_indices)
        mixed, params = make_mixture(source["X"], source_indices, nominal, rng, **mix_kwargs)
        save_sample(
            storage,
            source,
            sample_idx,
            "test_overlap",
            phase_indices,
            source_indices,
            nominal,
            mixed,
            params,
            hard_eval_type="peak_overlap",
            overlap_info=pair,
        )

    np.savez_compressed(output_path, **finalize_payload(storage, source, "test_overlap", seed))
    return summarize_split("test_overlap", output_path, storage, len(labels))


def generate_test_battery_relevant(
    source: dict[str, Any],
    output_path: Path,
    count: int,
    seed: int,
    mix_kwargs: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(source["labels"])
    class_to_indices = build_class_index(np.asarray(source["y"], dtype=np.int64))
    category_to_classes = make_category_to_classes(source)
    scenarios = [
        ("cathode_reference", ["cathode", "reference"], np.asarray([0.85, 0.15], dtype=np.float32)),
        ("anode_reference", ["anode", "reference"], np.asarray([0.85, 0.15], dtype=np.float32)),
        (
            "solid_electrolyte_reference",
            ["solid_electrolyte", "reference"],
            np.asarray([0.85, 0.15], dtype=np.float32),
        ),
        (
            "cathode_solid_electrolyte_reference",
            ["cathode", "solid_electrolyte", "reference"],
            np.asarray([0.75, 0.15, 0.10], dtype=np.float32),
        ),
        ("anode_solid_electrolyte_reference", ["anode", "solid_electrolyte", "reference"], np.asarray([0.75, 0.15, 0.10], dtype=np.float32)),
    ]
    available = [scenario for scenario in scenarios if all(category in category_to_classes for category in scenario[1])]
    if not available:
        raise RuntimeError("No battery-relevant category scenarios can be built from the source metadata.")
    storage = make_storage(source, count)

    for sample_idx in range(count):
        scenario_name, categories, nominal = available[sample_idx % len(available)]
        phase_indices = sample_classes_from_categories(rng, category_to_classes, categories)
        source_indices = choose_source_indices(rng, class_to_indices, phase_indices)
        mixed, params = make_mixture(source["X"], source_indices, nominal, rng, **mix_kwargs)
        save_sample(
            storage,
            source,
            sample_idx,
            "test_battery_relevant",
            phase_indices,
            source_indices,
            nominal,
            mixed,
            params,
            hard_eval_type="battery_relevant",
            battery_scenario=scenario_name,
        )

    np.savez_compressed(output_path, **finalize_payload(storage, source, "test_battery_relevant", seed))
    return summarize_split("test_battery_relevant", output_path, storage, len(labels))


def summarize_split(split_name: str, output_path: Path, storage: dict[str, Any], num_classes: int) -> dict[str, Any]:
    n_phases = storage["n_phases"]
    minor_bins = storage["minor_fraction_bin"]
    overlap_bins = storage["overlap_bin"]
    battery_scenarios = storage["battery_scenario"]
    positive_minor = storage["component_effective_fractions"][:, 1:]
    positive_minor = positive_minor[positive_minor > 0]
    return {
        "split": split_name,
        "path": str(output_path),
        "sample_count": int(storage["X"].shape[0]),
        "X_shape": list(storage["X"].shape),
        "num_classes": int(num_classes),
        "n_phase_counts": {str(k): int(v) for k, v in zip(*np.unique(n_phases, return_counts=True))},
        "minor_fraction_bin_counts": {str(k): int(v) for k, v in zip(*np.unique(minor_bins, return_counts=True))},
        "overlap_bin_counts": {str(k): int(v) for k, v in zip(*np.unique(overlap_bins, return_counts=True))},
        "battery_scenario_counts": {str(k): int(v) for k, v in zip(*np.unique(battery_scenarios, return_counts=True))},
        "mean_major_effective_fraction": float(np.mean(storage["major_fraction"])),
        "mean_minor_effective_fraction": float(np.mean(positive_minor)) if positive_minor.size > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate targeted hard-evaluation mixed-phase XRD test sets.")
    parser.add_argument("--single-phase-dir", type=Path, default=DEFAULT_SINGLE_PHASE_DIR)
    parser.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--minor-count", type=int, default=None)
    parser.add_argument("--overlap-count", type=int, default=None)
    parser.add_argument("--battery-count", type=int, default=None)
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
            "Refusing to overwrite an existing hard-eval dataset directory. "
            "Choose a new --dataset-name or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_path = args.single_phase_dir / f"{args.source_split}.npz"
        source = load_single_split(source_path)
        mix_kwargs = {
            "max_component_shift_bins": int(args.max_component_shift_bins),
            "max_global_shift_bins": int(args.max_global_shift_bins),
            "background_range": (float(args.background_min), float(args.background_max)),
            "noise_range": (float(args.noise_min), float(args.noise_max)),
            "background_order": int(args.background_order),
        }
        counts = {
            "test_minor": int(args.minor_count if args.minor_count is not None else args.count),
            "test_overlap": int(args.overlap_count if args.overlap_count is not None else args.count),
            "test_battery_relevant": int(args.battery_count if args.battery_count is not None else args.count),
        }
        summaries = [
            generate_test_minor(source, output_dir / "test_minor.npz", counts["test_minor"], args.seed + 101, mix_kwargs),
            generate_test_overlap(source, output_dir / "test_overlap.npz", counts["test_overlap"], args.seed + 202, mix_kwargs),
            generate_test_battery_relevant(
                source,
                output_dir / "test_battery_relevant.npz",
                counts["test_battery_relevant"],
                args.seed + 303,
                mix_kwargs,
            ),
        ]
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_name": args.dataset_name,
            "source_path": str(source_path),
            "source_split": args.source_split,
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "max_components": MAX_COMPONENTS,
            "hard_eval_policy": {
                "test_minor": {
                    "nominal_fraction_scenarios": ["95/5", "90/10", "85/15", "80/10/10", "85/10/5"],
                    "purpose": "minor phase sensitivity",
                },
                "test_overlap": {
                    "pair_selection": "cosine similarity among per-phase mean spectra, sampled from top 10 percent",
                    "overlap_bins": ["top_1pct", "top_5pct", "top_10pct"],
                    "purpose": "peak overlap sensitivity",
                },
                "test_battery_relevant": {
                    "category_scenarios": [
                        "cathode_reference",
                        "anode_reference",
                        "solid_electrolyte_reference",
                        "cathode_solid_electrolyte_reference",
                        "anode_solid_electrolyte_reference",
                    ],
                    "purpose": "battery-relevant mixture transfer",
                },
                "augmentation": {
                    "max_component_shift_bins": int(args.max_component_shift_bins),
                    "max_global_shift_bins": int(args.max_global_shift_bins),
                    "background_range": [float(args.background_min), float(args.background_max)],
                    "noise_range": [float(args.noise_min), float(args.noise_max)],
                    "background_order": int(args.background_order),
                },
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
        print(f"Saved hard-eval mixed-phase dataset to {output_dir}")
        for summary in summaries:
            print(f"  {summary['split']}: {summary['X_shape']} -> {summary['path']}")
    except Exception as exc:
        print(f"Hard-eval dataset generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
