from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CIF_DIR = DATA_DIR / "raw" / "cif"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

# Local-only convenience for this course project.
# Paste your Materials Project key here if you do not want to manage environment
# variables in the current shell. Before any public submission, set this back to "".
MP_API_KEY = ""

PILOT_TARGET_TOTAL = 300
QUERY_TOP_K = 20
MAX_VARIANTS_PER_PHASE_LABEL = 3
STRICT_E_ABOVE_HULL = 0.05
RELAXED_E_ABOVE_HULL = 0.10
SLEEP_SECONDS = 0.6


# First-stage scope: battery materials + common decomposition products.
# Electrocatalysts are useful later, but are excluded from the first baseline to keep
# the project centered on battery-material phase/family identification.
# The first-stage label is formula_pretty + space group, not formula alone.
MATERIAL_SCOPE = [
    {
        "category": "cathode",
        "target_fraction": 0.40,
        "subclasses": {
            "layered_oxide": [
                "Li-Co-O",
                "Li-Ni-O",
                "Li-Mn-O",
                "Co-Li-Ni-O",
                "Co-Li-Mn-O",
                "Li-Mn-Ni-O",
                "Co-Li-Mn-Ni-O",
                "Na-Co-O",
                "Na-Mn-O",
                "Na-Ni-O",
            ],
            "spinel_oxide": [
                "Li-Mn-O",
                "Li-Mn-Ni-O",
                "Co-Li-Mn-O",
                "Li-Ti-Mn-O",
            ],
            "olivine_polyanion": [
                "Fe-Li-O-P",
                "Li-Mn-O-P",
                "Co-Li-O-P",
                "Li-Ni-O-P",
                "Li-O-P-V",
            ],
            "conversion_fluoride": [
                "Fe-F",
                "Co-F",
                "Ni-F",
                "Fe-Li-F",
                "Li-Mn-F-O",
            ],
        },
    },
    {
        "category": "anode",
        "target_fraction": 0.25,
        "subclasses": {
            "carbon_silicon_alloy": [
                "C",
                "Si",
                "Ge",
                "Li-C",
                "Li-Si",
                "Li-Sn",
                "Li-Sb",
                "Na-C",
                "Na-Si",
                "Na-Sn",
                "Mg-Si",
            ],
            "titanate_oxide": [
                "Li-O-Ti",
                "Li-Nb-O",
                "Na-O-Ti",
                "O-Ti",
                "Nb-O",
            ],
            "metal_reference": [
                "Li",
                "Na",
                "Mg",
                "Al",
                "Sn",
                "Sb",
                "Zn",
            ],
        },
    },
    {
        "category": "solid_electrolyte",
        "target_fraction": 0.25,
        "band_gap_min": 1.5,
        "subclasses": {
            "garnet_oxide": [
                "La-Li-O-Zr",
                "La-Li-O-Ta",
                "La-Li-Nb-O",
                "Al-La-Li-O-Zr",
            ],
            "nasicon_oxide": [
                "Li-O-P-Ti",
                "Li-O-P-Zr",
                "Ge-Li-O-P",
                "Li-O-P-Si",
                "Na-O-P-Zr",
            ],
            "sulfide": [
                "Ge-Li-P-S",
                "Li-P-S",
                "Cl-Li-P-S",
                "Br-Li-P-S",
                "Na-P-S",
            ],
            "halide": [
                "Cl-Li",
                "Br-Li",
                "Cl-Li-Y",
                "Cl-Li-Zr",
                "Cl-Na",
            ],
        },
    },
    {
        "category": "reference",
        "target_fraction": 0.10,
        "subclasses": {
            "common_decomposition_product": [
                "Li-O",
                "C-Li-O",
                "Li-F",
                "Li-P-O",
                "Na-O",
            ],
        },
    },
]


@dataclass
class MaterialRecord:
    record_id: str
    material_id: str
    phase_label: str
    phase_label_idx: int
    formula_pretty: str
    category: str
    subclass: str
    chemical_system: str
    elements: str
    spacegroup_symbol: str
    spacegroup_number: int | None
    crystal_system: str
    energy_above_hull: float | None
    formation_energy_per_atom: float | None
    band_gap: float | None
    density: float | None
    volume: float | None
    nsites: int | None
    is_stable: bool
    is_ordered: bool | None
    cif_path: str
    structure_json_path: str
    query_chemsys: str
    structure_rank: int
    selection_reason: str
    download_time_utc: str
    xrd_validation_status: str = "not_checked"
    xrd_peak_count: int | None = None
    xrd_validation_error: str = ""


@dataclass
class Candidate:
    doc: Any
    query: dict[str, Any]
    reason: str
    sort_key: tuple[int, float, str]


def ensure_dirs() -> None:
    for path in [RAW_CIF_DIR, METADATA_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown"


def canonical_chemsys(chemsys: str) -> str:
    return "-".join(sorted(part.strip() for part in chemsys.split("-") if part.strip()))


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def nested_attr(obj: Any, names: list[str], default: Any = None) -> Any:
    current = obj
    for name in names:
        current = get_attr(current, name, default)
        if current is default:
            return default
    return current


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def structure_elements(structure: Any) -> str:
    try:
        return "-".join(sorted(str(element) for element in structure.composition.elements))
    except Exception:
        return ""


def structure_is_valid(structure: Any) -> bool:
    try:
        lattice = structure.lattice
        return (
            len(structure) > 0
            and float(lattice.a) > 0
            and float(lattice.b) > 0
            and float(lattice.c) > 0
        )
    except Exception:
        return False


def structure_is_ordered(structure: Any) -> bool:
    try:
        ordered = getattr(structure, "is_ordered", False)
        return bool(ordered() if callable(ordered) else ordered)
    except Exception:
        return False


def spacegroup_symbol(doc: Any) -> str:
    return str(nested_attr(doc, ["symmetry", "symbol"], "") or "unknown")


def phase_label_from_doc(doc: Any) -> str:
    formula = str(get_attr(doc, "formula_pretty", "") or get_attr(doc, "material_id", "unknown"))
    return f"{formula}_{safe_name(spacegroup_symbol(doc))}"


def target_counts(total: int) -> dict[str, int]:
    counts = {
        item["category"]: max(1, round(total * float(item["target_fraction"])))
        for item in MATERIAL_SCOPE
    }
    delta = total - sum(counts.values())
    order = [item["category"] for item in MATERIAL_SCOPE]
    idx = 0
    while delta != 0:
        category = order[idx % len(order)]
        if delta > 0:
            counts[category] += 1
            delta -= 1
        elif counts[category] > 1:
            counts[category] -= 1
            delta += 1
        idx += 1
    return counts


def subclass_target_counts(category_targets: dict[str, int]) -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    for item in MATERIAL_SCOPE:
        category = item["category"]
        subclasses = list(item["subclasses"])
        base = category_targets[category] // len(subclasses)
        remainder = category_targets[category] % len(subclasses)
        for idx, subclass in enumerate(subclasses):
            targets[(category, subclass)] = base + (1 if idx < remainder else 0)
    return targets


def make_query_plan(targets: dict[str, int]) -> list[dict[str, Any]]:
    plan = []
    for item in MATERIAL_SCOPE:
        for subclass, chemsys_list in item["subclasses"].items():
            for chemsys in chemsys_list:
                plan.append(
                    {
                        "category": item["category"],
                        "subclass": subclass,
                        "chemsys": canonical_chemsys(chemsys),
                        "target": targets[item["category"]],
                        "band_gap_min": item.get("band_gap_min"),
                        "allow_metallic": bool(item.get("allow_metallic", False)),
                    }
                )
    return plan


def fetch_docs(mpr: Any, chemsys: str, stable_only: bool) -> list[Any]:
    fields = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "density",
        "volume",
        "nsites",
        "is_stable",
        "structure",
    ]
    kwargs = {"chemsys": chemsys, "fields": fields}
    if stable_only:
        kwargs["is_stable"] = True
    try:
        return list(mpr.materials.summary.search(**kwargs))
    except Exception:
        kwargs.pop("fields", None)
        return list(mpr.materials.summary.search(**kwargs))


def sort_key(doc: Any) -> tuple[int, float, str]:
    stable = bool(get_attr(doc, "is_stable", False))
    e_hull = optional_float(get_attr(doc, "energy_above_hull"))
    return (0 if stable else 1, float("inf") if e_hull is None else e_hull, str(get_attr(doc, "material_id", "")))


def passes_filters(doc: Any, query: dict[str, Any]) -> tuple[bool, str]:
    stable = bool(get_attr(doc, "is_stable", False))
    e_hull = optional_float(get_attr(doc, "energy_above_hull"))
    band_gap = optional_float(get_attr(doc, "band_gap"))

    threshold = STRICT_E_ABOVE_HULL if stable else RELAXED_E_ABOVE_HULL
    if e_hull is not None and e_hull > threshold:
        return False, f"e_hull>{threshold:g}"
    if e_hull is None and not stable:
        return False, "missing_stability"

    band_gap_min = query.get("band_gap_min")
    if band_gap_min is not None and not query["allow_metallic"]:
        if band_gap is not None and band_gap < float(band_gap_min):
            return False, f"band_gap<{band_gap_min:g}"

    return True, "stable" if stable else f"e_hull<={threshold:g}"


def diverse_order(docs: list[Any]) -> list[Any]:
    buckets: dict[str, list[Any]] = {}
    for doc in sorted(docs, key=sort_key):
        sg = str(nested_attr(doc, ["symmetry", "symbol"], "") or "unknown")
        buckets.setdefault(sg, []).append(doc)

    ordered = []
    while buckets:
        for sg in sorted(list(buckets)):
            bucket = buckets[sg]
            if bucket:
                ordered.append(bucket.pop(0))
            if not bucket:
                buckets.pop(sg, None)
    return ordered


def better_candidate(new: Candidate, old: Candidate | None) -> bool:
    if old is None:
        return True
    return new.sort_key < old.sort_key


def add_candidates(
    candidate_by_material_id: dict[str, Candidate],
    docs: list[Any],
    query: dict[str, Any],
    query_top_k: int,
) -> None:
    accepted = []
    reason_by_id = {}
    for doc in docs:
        ok, reason = passes_filters(doc, query)
        if ok:
            accepted.append(doc)
            reason_by_id[str(get_attr(doc, "material_id", ""))] = reason

    for doc in diverse_order(accepted)[:query_top_k]:
        material_id = str(get_attr(doc, "material_id", ""))
        if not material_id:
            continue
        candidate = Candidate(
            doc=doc,
            query=query,
            reason=reason_by_id.get(material_id, "selected"),
            sort_key=sort_key(doc),
        )
        if better_candidate(candidate, candidate_by_material_id.get(material_id)):
            candidate_by_material_id[material_id] = candidate


def diverse_candidates(candidates: list[Candidate]) -> list[Candidate]:
    buckets: dict[str, list[Candidate]] = {}
    for candidate in sorted(candidates, key=lambda item: item.sort_key):
        sg = spacegroup_symbol(candidate.doc)
        buckets.setdefault(sg, []).append(candidate)

    ordered: list[Candidate] = []
    while buckets:
        for sg in sorted(list(buckets)):
            bucket = buckets[sg]
            if bucket:
                ordered.append(bucket.pop(0))
            if not bucket:
                buckets.pop(sg, None)
    return ordered


def save_outputs(records: list[MaterialRecord]) -> None:
    if not records:
        return

    label_to_idx: dict[str, int] = {}
    for record in records:
        label_to_idx.setdefault(record.phase_label, len(label_to_idx))
        record.phase_label_idx = label_to_idx[record.phase_label]

    df = pd.DataFrame([asdict(record) for record in records])
    df.to_csv(METADATA_DIR / "materials_index.csv", index=False)
    df.to_json(METADATA_DIR / "materials_index.json", orient="records", indent=2, force_ascii=False)

    label_map = (
        df[["phase_label_idx", "phase_label"]]
        .drop_duplicates()
        .sort_values("phase_label_idx")
        .to_dict(orient="records")
    )
    pd.DataFrame(label_map).to_csv(METADATA_DIR / "label_map.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "record_count": int(len(df)),
        "phase_label_count": int(df["phase_label"].nunique()),
        "material_id_count": int(df["material_id"].nunique()),
        "category_counts": df["category"].value_counts().sort_index().to_dict(),
        "subclass_counts": df.groupby(["category", "subclass"]).size().astype(int).reset_index(name="count").to_dict(orient="records"),
        "ordered_counts": df["is_ordered"].value_counts(dropna=False).astype(int).to_dict() if "is_ordered" in df else {},
        "xrd_validation_counts": df["xrd_validation_status"].value_counts(dropna=False).astype(int).to_dict()
        if "xrd_validation_status" in df
        else {},
        "records_per_phase_label": df.groupby("phase_label").size().astype(int).describe().to_dict(),
        "phase_label_policy": "formula_pretty + spacegroup_symbol",
    }
    (METADATA_DIR / "materials_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_existing_records() -> list[MaterialRecord]:
    path = METADATA_DIR / "materials_index.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    records = []
    for row in df.to_dict(orient="records"):
        clean = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        clean.setdefault("is_ordered", None)
        clean.setdefault("xrd_validation_status", "not_checked")
        clean.setdefault("xrd_peak_count", None)
        clean.setdefault("xrd_validation_error", "")
        records.append(MaterialRecord(**clean))
    return records


def validate_xrd(structure: Any) -> tuple[str, int | None, str]:
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator

        pattern = XRDCalculator(wavelength="CuKa").get_pattern(
            structure,
            two_theta_range=(10, 90),
        )
        return "ok", int(len(pattern.x)), ""
    except Exception as exc:
        return "failed", None, f"{type(exc).__name__}: {exc}"


def build_record(
    doc: Any,
    query: dict[str, Any],
    rank: int,
    reason: str,
    require_ordered: bool,
    validate_xrd_pattern: bool,
) -> tuple[MaterialRecord, Any]:
    structure = get_attr(doc, "structure")
    if structure is None:
        raise ValueError("document has no structure")
    if not structure_is_valid(structure):
        raise ValueError("invalid pymatgen structure")
    is_ordered = structure_is_ordered(structure)
    if require_ordered and not is_ordered:
        raise ValueError("disordered pymatgen structure")

    xrd_status = "not_checked"
    xrd_peak_count = None
    xrd_error = ""
    if validate_xrd_pattern:
        xrd_status, xrd_peak_count, xrd_error = validate_xrd(structure)
        if xrd_status != "ok":
            raise ValueError(f"XRD validation failed: {xrd_error}")

    material_id = str(get_attr(doc, "material_id", "unknown"))
    formula = str(get_attr(doc, "formula_pretty", "") or material_id)
    phase_label = phase_label_from_doc(doc)
    category = query["category"]
    subclass = query["subclass"]
    cif_path = RAW_CIF_DIR / category / subclass / safe_name(formula) / f"{safe_name(material_id)}_{safe_name(formula)}.cif"
    structure_json_path = cif_path.with_suffix(".structure.json")

    record = MaterialRecord(
        record_id=f"{category}_{subclass}_{safe_name(material_id)}",
        material_id=material_id,
        phase_label=phase_label,
        phase_label_idx=-1,
        formula_pretty=formula,
        category=category,
        subclass=subclass,
        chemical_system=str(get_attr(doc, "chemsys", query["chemsys"]) or query["chemsys"]),
        elements=structure_elements(structure),
        spacegroup_symbol=spacegroup_symbol(doc),
        spacegroup_number=get_attr(nested_attr(doc, ["symmetry"], None), "number", None),
        crystal_system=str(nested_attr(doc, ["symmetry", "crystal_system"], "") or ""),
        energy_above_hull=optional_float(get_attr(doc, "energy_above_hull")),
        formation_energy_per_atom=optional_float(get_attr(doc, "formation_energy_per_atom")),
        band_gap=optional_float(get_attr(doc, "band_gap")),
        density=optional_float(get_attr(doc, "density")),
        volume=optional_float(get_attr(doc, "volume")),
        nsites=get_attr(doc, "nsites", None) or len(structure),
        is_stable=bool(get_attr(doc, "is_stable", False)),
        is_ordered=is_ordered,
        cif_path=str(cif_path.relative_to(PROJECT_ROOT)),
        structure_json_path=str(structure_json_path.relative_to(PROJECT_ROOT)),
        query_chemsys=query["chemsys"],
        structure_rank=rank,
        selection_reason=reason,
        download_time_utc=datetime.now(UTC).isoformat(),
        xrd_validation_status=xrd_status,
        xrd_peak_count=xrd_peak_count,
        xrd_validation_error=xrd_error,
    )
    return record, structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Materials Project structures for the XRD project.")
    parser.add_argument("--target-total", type=int, default=PILOT_TARGET_TOTAL)
    parser.add_argument("--query-top-k", type=int, default=QUERY_TOP_K)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--allow-disordered", action="store_true")
    parser.add_argument("--validate-xrd", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    targets = target_counts(args.target_total)
    subclass_targets = subclass_target_counts(targets)
    plan = make_query_plan(targets)

    if args.dry_run:
        print(f"Target total: {args.target_total}")
        print(f"Category targets: {targets}")
        print("Subclass targets:")
        for (category, subclass), count in subclass_targets.items():
            print(f"  {category}/{subclass}: {count}")
        print(f"Queries: {len(plan)}")
        for idx, item in enumerate(plan[:40]):
            print(f"{idx:03d} {item['category']}/{item['subclass']}: {item['chemsys']}")
        if len(plan) > 40:
            print(f"... {len(plan) - 40} more")
        return 0

    api_key = args.api_key or MP_API_KEY.strip() or os.getenv("MP_API_KEY")
    if not api_key:
        print("Missing Materials Project API key. Set MP_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    if (METADATA_DIR / "materials_index.csv").exists() and not args.resume and not args.overwrite:
        print("materials_index.csv already exists. Use --resume or --overwrite.", file=sys.stderr)
        return 1

    records = [] if args.overwrite else load_existing_records()
    seen_material_ids = {record.material_id for record in records}
    category_counts = pd.Series([record.category for record in records]).value_counts().to_dict()
    subclass_counts = (
        pd.Series([(record.category, record.subclass) for record in records]).value_counts().to_dict()
        if records
        else {}
    )
    label_counts = pd.Series([record.phase_label for record in records]).value_counts().to_dict()
    failures = []

    try:
        from mp_api.client import MPRester
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install requirements.txt first.", file=sys.stderr)
        return 1

    candidate_by_material_id: dict[str, Candidate] = {}

    with MPRester(api_key) as mpr:
        for query in tqdm(plan, desc="Collecting MP candidates"):
            try:
                docs = fetch_docs(mpr, query["chemsys"], stable_only=True)
                if not docs:
                    docs = fetch_docs(mpr, query["chemsys"], stable_only=False)
            except Exception as exc:
                failures.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
                time.sleep(max(1.0, SLEEP_SECONDS))
                continue

            add_candidates(candidate_by_material_id, docs, query, args.query_top_k)
            time.sleep(SLEEP_SECONDS)

    candidates = list(candidate_by_material_id.values())
    subclass_targets = subclass_target_counts(targets)
    checkpoint = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_total": args.target_total,
        "category_targets": targets,
        "subclass_targets": {f"{category}/{subclass}": count for (category, subclass), count in subclass_targets.items()},
        "candidate_count": len(candidates),
        "query_top_k": args.query_top_k,
        "require_ordered": not args.allow_disordered,
        "validate_xrd": args.validate_xrd,
    }
    (CACHE_DIR / "download_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def try_accept(candidate: Candidate) -> bool:
        material_id = str(get_attr(candidate.doc, "material_id", ""))
        if material_id in seen_material_ids:
            return False
        phase_label = phase_label_from_doc(candidate.doc)
        if label_counts.get(phase_label, 0) >= MAX_VARIANTS_PER_PHASE_LABEL:
            return False
        category = candidate.query["category"]
        if category_counts.get(category, 0) >= targets[category]:
            return False

        try:
            record, structure = build_record(
                candidate.doc,
                candidate.query,
                rank=int(label_counts.get(phase_label, 0)),
                reason=candidate.reason,
                require_ordered=not args.allow_disordered,
                validate_xrd_pattern=args.validate_xrd,
            )
            cif_path = PROJECT_ROOT / record.cif_path
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            CifWriter(structure).write_file(cif_path)
            structure_json_path = PROJECT_ROOT / record.structure_json_path
            structure_json_path.write_text(
                json.dumps(structure.as_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            failures.append(
                {
                    "query": candidate.query,
                    "material_id": material_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return False

        records.append(record)
        seen_material_ids.add(record.material_id)
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
        subclass_key = (record.category, record.subclass)
        subclass_counts[subclass_key] = subclass_counts.get(subclass_key, 0) + 1
        label_counts[record.phase_label] = label_counts.get(record.phase_label, 0) + 1
        return True

    candidates_by_subclass: dict[tuple[str, str], list[Candidate]] = {}
    candidates_by_category: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        category = candidate.query["category"]
        subclass = candidate.query["subclass"]
        candidates_by_subclass.setdefault((category, subclass), []).append(candidate)
        candidates_by_category.setdefault(category, []).append(candidate)

    for item in MATERIAL_SCOPE:
        category = item["category"]
        for subclass in item["subclasses"]:
            key = (category, subclass)
            needed = subclass_targets[key] - subclass_counts.get(key, 0)
            if needed <= 0:
                continue
            accepted_here = 0
            for candidate in diverse_candidates(candidates_by_subclass.get(key, [])):
                if accepted_here >= needed:
                    break
                if try_accept(candidate):
                    accepted_here += 1
            save_outputs(records)

    for item in MATERIAL_SCOPE:
        category = item["category"]
        needed = targets[category] - category_counts.get(category, 0)
        if needed <= 0:
            continue
        accepted_here = 0
        for candidate in diverse_candidates(candidates_by_category.get(category, [])):
            if accepted_here >= needed:
                break
            if try_accept(candidate):
                accepted_here += 1
        save_outputs(records)

    save_outputs(records)
    pd.DataFrame(failures).to_json(CACHE_DIR / "download_failures.json", orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(records)} records to {METADATA_DIR / 'materials_index.csv'}")
    print(f"Category counts: {category_counts}")
    if failures:
        print(f"Recorded {len(failures)} failures in data/cache/download_failures.json")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
