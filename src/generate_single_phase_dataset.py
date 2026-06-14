from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed" / "single_phase"

RANDOM_SEED = 42
TWO_THETA_MIN = 10.0
TWO_THETA_MAX = 90.0
NUM_POINTS = 3501
WAVELENGTH = "CuKa"
TOP_K_PEAKS = 10
XRD_CALCULATOR = XRDCalculator(wavelength=WAVELENGTH)

USE_LATTICE_PERTURBATION = True
LATTICE_PERTURBATION_RANGE = 0.005


SPLIT_CONFIGS = {
    "train": {
        "count": 150,
        "shift_range": 0.03,
        "noise_range": (0.0, 0.03),
        "background_range": (0.0, 0.04),
        "scale_range": (0.80, 1.20),
        "sigma_range": (0.04, 0.10),
        "eta_range": (0.00, 0.35),
        "peak_intensity_jitter": 0.12,
        "background_order": 3,
    },
    "val": {
        "count": 20,
        "shift_range": 0.03,
        "noise_range": (0.0, 0.03),
        "background_range": (0.0, 0.04),
        "scale_range": (0.80, 1.20),
        "sigma_range": (0.04, 0.10),
        "eta_range": (0.00, 0.35),
        "peak_intensity_jitter": 0.12,
        "background_order": 3,
    },
    "test_normal": {
        "count": 20,
        "shift_range": 0.03,
        "noise_range": (0.0, 0.03),
        "background_range": (0.0, 0.04),
        "scale_range": (0.80, 1.20),
        "sigma_range": (0.04, 0.10),
        "eta_range": (0.00, 0.35),
        "peak_intensity_jitter": 0.12,
        "background_order": 3,
    },
    "test_hard": {
        "count": 20,
        "shift_range": 0.06,
        "noise_range": (0.03, 0.06),
        "background_range": (0.03, 0.08),
        "scale_range": (0.75, 1.25),
        "sigma_range": (0.06, 0.16),
        "eta_range": (0.15, 0.60),
        "peak_intensity_jitter": 0.22,
        "background_order": 4,
    },
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def two_theta_grid() -> np.ndarray:
    return np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, NUM_POINTS, dtype=np.float32)


def normalize_max(x: np.ndarray) -> np.ndarray:
    max_value = float(np.max(x))
    if max_value <= 0:
        return x.astype(np.float32)
    return (x / max_value).astype(np.float32)


def gaussian_profile(grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid - center) / sigma) ** 2)


def lorentzian_profile(grid: np.ndarray, center: float, gamma: float) -> np.ndarray:
    return 1.0 / (1.0 + ((grid - center) / gamma) ** 2)


def pseudo_voigt_profile(grid: np.ndarray, center: float, sigma: float, eta: float) -> np.ndarray:
    return (1.0 - eta) * gaussian_profile(grid, center, sigma) + eta * lorentzian_profile(grid, center, sigma)


def pattern_to_spectrum(pattern: Any, rng: np.random.Generator, cfg: dict[str, Any]) -> np.ndarray:
    grid = two_theta_grid()
    spectrum = np.zeros_like(grid, dtype=np.float32)
    for peak_pos, intensity in zip(pattern.x, pattern.y):
        sigma = float(rng.uniform(*cfg["sigma_range"]))
        eta = float(rng.uniform(*cfg["eta_range"]))
        peak_scale = float(rng.lognormal(mean=0.0, sigma=cfg["peak_intensity_jitter"]))
        spectrum += float(intensity) * peak_scale * pseudo_voigt_profile(grid, float(peak_pos), sigma, eta)
    return normalize_max(spectrum)


def random_polynomial_background(rng: np.random.Generator, max_level: float, order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, NUM_POINTS, dtype=np.float32)
    coeffs = rng.normal(0.0, 1.0, size=order + 1).astype(np.float32)
    coeffs[0] = abs(coeffs[0]) + 0.5
    background = np.polynomial.polynomial.polyval(x, coeffs)
    background = background - np.min(background)
    if float(np.max(background)) > 0:
        background = background / np.max(background)
    return (background * max_level).astype(np.float32)


def shift_spectrum(spectrum: np.ndarray, shift_deg: float) -> np.ndarray:
    grid = two_theta_grid()
    shifted_grid = grid - shift_deg
    return np.interp(grid, shifted_grid, spectrum, left=0.0, right=0.0).astype(np.float32)


def augment_spectrum(base_spectrum: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any]) -> np.ndarray:
    x = shift_spectrum(base_spectrum, rng.uniform(-cfg["shift_range"], cfg["shift_range"]))
    x = x * float(rng.uniform(*cfg["scale_range"]))
    x = x + random_polynomial_background(
        rng,
        max_level=float(rng.uniform(*cfg["background_range"])),
        order=int(cfg["background_order"]),
    )
    noise_level = float(rng.uniform(*cfg["noise_range"]))
    if noise_level > 0:
        x = x + rng.normal(0.0, noise_level, size=x.shape)
    return normalize_max(np.clip(x, 0.0, None))


def perturb_lattice(structure: Structure, rng: np.random.Generator) -> Structure:
    if not USE_LATTICE_PERTURBATION:
        return structure.copy()
    lattice = structure.lattice
    eps = rng.uniform(-LATTICE_PERTURBATION_RANGE, LATTICE_PERTURBATION_RANGE, size=3)
    new_lattice = Lattice.from_parameters(
        lattice.a * (1.0 + eps[0]),
        lattice.b * (1.0 + eps[1]),
        lattice.c * (1.0 + eps[2]),
        lattice.alpha,
        lattice.beta,
        lattice.gamma,
    )
    return Structure(new_lattice, structure.species, structure.frac_coords, site_properties=structure.site_properties)


def lattice_params(structure: Structure) -> np.ndarray:
    lattice = structure.lattice
    return np.asarray([lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma], dtype=np.float32)


def hkl_from_peak(peak_hkls: list[dict[str, Any]]) -> tuple[int, int, int]:
    if not peak_hkls:
        return (0, 0, 0)
    hkl = tuple(int(value) for value in peak_hkls[0].get("hkl", (0, 0, 0)))
    if len(hkl) == 4:
        return (hkl[0], hkl[1], hkl[3])
    if len(hkl) < 3:
        hkl = hkl + (0,) * (3 - len(hkl))
    return hkl[:3]


def top_k_peak_labels(pattern: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peaks = sorted(
        [(float(intensity), float(pos), hkl_from_peak(hkls)) for pos, intensity, hkls in zip(pattern.x, pattern.y, pattern.hkls)],
        key=lambda item: item[0],
        reverse=True,
    )[:TOP_K_PEAKS]
    peak_hkls = np.zeros((TOP_K_PEAKS, 3), dtype=np.float32)
    peak_2theta = np.zeros(TOP_K_PEAKS, dtype=np.float32)
    peak_mask = np.zeros(TOP_K_PEAKS, dtype=np.float32)
    for idx, (_, pos, hkl) in enumerate(peaks):
        peak_hkls[idx] = np.asarray(hkl, dtype=np.float32)
        peak_2theta[idx] = pos
        peak_mask[idx] = 1.0
    return peak_hkls, peak_2theta, peak_mask


def load_material_index(path: Path, limit_labels: int | None, limit_records: int | None) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/download_materials.py first.")

    df = pd.read_csv(path)
    df = df.sort_values(["phase_label_idx", "structure_rank", "material_id"]).reset_index(drop=True)
    labels = list(dict.fromkeys(df["phase_label"].astype(str).tolist()))
    if limit_labels is not None:
        labels = labels[:limit_labels]
        df = df[df["phase_label"].astype(str).isin(labels)].copy()
    if limit_records is not None:
        df = df.head(limit_records).copy()
        labels = list(dict.fromkeys(df["phase_label"].astype(str).tolist()))
    return df.reset_index(drop=True), labels


def load_structure(row: Any) -> Structure:
    cif_path = PROJECT_ROOT / str(row.cif_path)
    return Structure.from_file(cif_path)


def structure_is_ordered(structure: Structure) -> bool:
    ordered = getattr(structure, "is_ordered", False)
    return bool(ordered() if callable(ordered) else ordered)


def get_xrd_pattern(structure: Structure, row: Any, label: str) -> Any:
    try:
        return XRD_CALCULATOR.get_pattern(
            structure,
            two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX),
        )
    except Exception as exc:
        raise RuntimeError(
            "XRDCalculator failed for "
            f"label={label}, material_id={row.material_id}, "
            f"record_id={row.record_id}, cif_path={row.cif_path}, "
            f"is_ordered={structure_is_ordered(structure)}, "
            f"error={type(exc).__name__}: {exc}"
        ) from exc


def make_split(split_name: str, cfg: dict[str, Any], df: pd.DataFrame, labels: list[str], rng: np.random.Generator) -> dict[str, Any]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    xs, ys = [], []
    lattice_targets, peak_hkls_targets, peak_2theta_targets, peak_mask_targets = [], [], [], []
    sample_material_ids, sample_record_ids, sample_categories, sample_subclasses, sample_cif_paths = [], [], [], [], []
    sample_phase_labels, sample_crystal_systems, sample_spacegroup_symbols, sample_spacegroup_numbers = [], [], [], []

    rows_by_label = {
        label: list(df[df["phase_label"].astype(str) == label].itertuples(index=False))
        for label in labels
    }
    structure_cache: dict[str, Structure] = {}

    for label in tqdm(labels, desc=f"Generating {split_name}"):
        variants = rows_by_label[label]
        label_idx = label_to_idx[label]
        for _ in range(int(cfg["count"])):
            row = variants[int(rng.integers(0, len(variants)))]
            record_id = str(row.record_id)
            if record_id not in structure_cache:
                structure_cache[record_id] = load_structure(row)
            structure = structure_cache[record_id]
            sample_structure = perturb_lattice(structure, rng)
            pattern = get_xrd_pattern(sample_structure, row, label)
            if len(pattern.x) == 0:
                raise ValueError(f"No XRD peaks generated for {record_id} in {TWO_THETA_MIN}-{TWO_THETA_MAX} degrees.")
            base_spectrum = pattern_to_spectrum(pattern, rng, cfg)
            xs.append(augment_spectrum(base_spectrum, rng, cfg))
            ys.append(label_idx)
            lattice_targets.append(lattice_params(sample_structure))
            peak_hkls, peak_2theta, peak_mask = top_k_peak_labels(pattern)
            peak_hkls_targets.append(peak_hkls)
            peak_2theta_targets.append(peak_2theta)
            peak_mask_targets.append(peak_mask)
            sample_material_ids.append(str(row.material_id))
            sample_record_ids.append(str(row.record_id))
            sample_categories.append(str(row.category))
            sample_subclasses.append(str(row.subclass))
            sample_cif_paths.append(str(row.cif_path))
            sample_phase_labels.append(str(row.phase_label))
            sample_crystal_systems.append(str(row.crystal_system))
            sample_spacegroup_symbols.append(str(row.spacegroup_symbol))
            sample_spacegroup_numbers.append(-1 if pd.isna(row.spacegroup_number) else int(row.spacegroup_number))

    output_path = PROCESSED_DIR / f"{split_name}.npz"
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    crystal_system_labels = sorted(str(value) for value in df["crystal_system"].fillna("unknown").astype(str).unique())
    crystal_to_idx = {label: idx for idx, label in enumerate(crystal_system_labels)}
    crystal_system_y = np.asarray([crystal_to_idx[label] for label in sample_crystal_systems], dtype=np.int64)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        labels=np.asarray(labels),
        crystal_system_y=crystal_system_y,
        crystal_system_labels=np.asarray(crystal_system_labels),
        two_theta=two_theta_grid(),
        lattice_params=np.stack(lattice_targets).astype(np.float32),
        peak_hkls=np.stack(peak_hkls_targets).astype(np.float32),
        peak_2theta=np.stack(peak_2theta_targets).astype(np.float32),
        peak_mask=np.stack(peak_mask_targets).astype(np.float32),
        sample_material_ids=np.asarray(sample_material_ids),
        sample_record_ids=np.asarray(sample_record_ids),
        sample_categories=np.asarray(sample_categories),
        sample_subclasses=np.asarray(sample_subclasses),
        sample_cif_paths=np.asarray(sample_cif_paths),
        sample_phase_labels=np.asarray(sample_phase_labels),
        sample_crystal_systems=np.asarray(sample_crystal_systems),
        sample_spacegroup_symbols=np.asarray(sample_spacegroup_symbols),
        sample_spacegroup_numbers=np.asarray(sample_spacegroup_numbers, dtype=np.int64),
    )
    return {"split": split_name, "path": str(output_path), "X_shape": list(X.shape), "class_count": len(labels)}


def diagnose_xrd(df: pd.DataFrame, labels: list[str]) -> int:
    silicon = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    print("Diagnostic 1/2: testing XRDCalculator on a simple ordered Si structure...", flush=True)
    si_pattern = XRD_CALCULATOR.get_pattern(
        silicon,
        two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX),
    )
    print(f"Si OK: {len(si_pattern.x)} peaks; first peaks={si_pattern.x[:3]}", flush=True)

    print("Diagnostic 2/2: testing downloaded structures in metadata order...", flush=True)
    tested = 0
    for row in df.itertuples(index=False):
        label = str(row.phase_label)
        if label not in labels:
            continue
        print(
            f"Testing {tested + 1}: label={label}, material_id={row.material_id}, "
            f"record_id={row.record_id}, cif_path={row.cif_path}",
            flush=True,
        )
        structure = load_structure(row)
        print(f"  is_ordered={structure_is_ordered(structure)}, nsites={len(structure)}", flush=True)
        pattern = get_xrd_pattern(structure, row, label)
        print(f"  OK: {len(pattern.x)} peaks", flush=True)
        tested += 1
    print(f"All tested structures passed: {tested}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-phase synthetic XRD NPZ datasets.")
    parser.add_argument("--metadata", default=str(METADATA_DIR / "materials_index.csv"))
    parser.add_argument("--limit-labels", type=int, default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    df, labels = load_material_index(Path(args.metadata), args.limit_labels, args.limit_records)
    if args.diagnose_only:
        return diagnose_xrd(df, labels)

    if any(PROCESSED_DIR.glob("*.npz")) and not args.overwrite:
        print(f"{PROCESSED_DIR} already contains .npz files. Use --overwrite.", file=sys.stderr)
        return 1

    df.to_csv(PROCESSED_DIR / "selected_material_records.csv", index=False)
    pd.DataFrame({"label_idx": range(len(labels)), "phase_label": labels}).to_csv(PROCESSED_DIR / "label_index.csv", index=False)

    rng = np.random.default_rng(args.seed)
    summaries = []
    for split_name, cfg in SPLIT_CONFIGS.items():
        summaries.append(make_split(split_name, cfg, df, labels, rng))

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "metadata": str(args.metadata),
        "record_count": int(len(df)),
        "label_count": int(len(labels)),
        "two_theta": {"min": TWO_THETA_MIN, "max": TWO_THETA_MAX, "num_points": NUM_POINTS, "wavelength": WAVELENGTH},
        "split_configs": SPLIT_CONFIGS,
        "split_summaries": summaries,
    }
    (PROCESSED_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved single-phase dataset to {PROCESSED_DIR}")
    for summary in summaries:
        print(f"- {summary['split']}: {summary['X_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
