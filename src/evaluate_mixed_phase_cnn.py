from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from model_mixture import MixedPhaseXRDConvNet
from train_mixed_phase_cnn import (
    MixedPhaseDataset,
    calibrate_threshold,
    compute_metrics,
    evaluate_by_mixture_size,
    per_phase_metrics,
    predict_split,
    resolve_device,
    save_predictions_csv,
    save_threshold_curve,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v2_hard_eval"
DEFAULT_CALIBRATION_DATA = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v1" / "val.npz"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval"
THRESHOLD_MODES = ("fixed", "val_calibrated", "oracle", "all")


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())


def discover_splits(data_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    split_paths = sorted(path for path in data_dir.glob("*.npz") if path.name != "train.npz")
    return [path.stem for path in split_paths]


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[MixedPhaseXRDConvNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model_config = checkpoint.get("model_config")
    if model_config is None:
        labels = checkpoint.get("phase_labels")
        if labels is None:
            raise ValueError("Checkpoint has no model_config or phase_labels; cannot infer class count.")
        model_config = {
            "num_phase_classes": len(labels),
            "pooled_bins": 32,
            "dropout": 0.20,
            "hidden_dim": 512,
            "head_hidden_dim": None,
            "use_fraction_level_head": any(key.startswith("fraction_level_head.") for key in state_dict),
            "fraction_level_count": 3,
        }
    else:
        model_config = dict(model_config)
        has_fraction_weights = any(key.startswith("fraction_level_head.") for key in state_dict)
        model_config["use_fraction_level_head"] = bool(has_fraction_weights)
    model = MixedPhaseXRDConvNet(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    checkpoint_tag = safe_name(args.checkpoint.parent.name or args.checkpoint.stem)
    dataset_tag = safe_name(args.data_dir.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.results_root / f"{checkpoint_tag}_on_{dataset_tag}_{timestamp}"


def threshold_modes_for_arg(mode: str) -> list[str]:
    if mode == "all":
        return ["fixed", "val_calibrated", "oracle"]
    return [mode]


def make_phase_criterion(num_classes: int, device: torch.device) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=torch.ones(num_classes, device=device))


def checkpoint_labels(checkpoint: dict[str, Any]) -> list[str] | None:
    labels = checkpoint.get("phase_labels")
    if labels is None:
        run_config = checkpoint.get("run_config")
        if isinstance(run_config, dict):
            labels = run_config.get("phase_labels")
    if labels is None:
        return None
    return [str(label) for label in labels]


def metrics_for_mask(
    pred: dict[str, Any],
    mask: np.ndarray,
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> dict[str, float]:
    return compute_metrics(
        pred["y_true"][mask],
        pred["probabilities"][mask],
        pred["component_phase_indices"][mask],
        pred["n_phases"][mask],
        pred["major_phase_index"][mask],
        threshold,
        min_predictions,
        max_predictions,
    )


def group_metrics_from_array(
    pred: dict[str, Any],
    group_values: np.ndarray,
    group_name: str,
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = np.asarray(group_values).astype(str)
    for value in sorted(set(values.tolist())):
        if value in {"", "none", "nan"}:
            continue
        mask = values == value
        if not np.any(mask):
            continue
        metrics = metrics_for_mask(pred, mask, threshold, min_predictions, max_predictions)
        metrics[group_name] = value
        rows.append(metrics)
    return rows


def inferred_minor_bins(component_fractions: np.ndarray) -> np.ndarray:
    bins = np.full(component_fractions.shape[0], "none", dtype="<U16")
    for row_idx, fractions in enumerate(component_fractions):
        positive = fractions[fractions > 0]
        if len(positive) <= 1:
            continue
        minor = float(np.min(positive))
        if minor <= 0.075:
            bins[row_idx] = "minor_5pct"
        elif minor <= 0.125:
            bins[row_idx] = "minor_10pct"
        elif minor <= 0.175:
            bins[row_idx] = "minor_15pct"
        else:
            bins[row_idx] = "minor_gt15pct"
    return bins


def load_group_values(npz_path: Path, source_indices: np.ndarray) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    groups: dict[str, np.ndarray] = {}
    if "minor_fraction_bin" in data.files:
        groups["minor_fraction_bin"] = np.asarray(data["minor_fraction_bin"])[source_indices]
    if "overlap_bin" in data.files:
        groups["overlap_bin"] = np.asarray(data["overlap_bin"])[source_indices]
    if "battery_scenario" in data.files:
        groups["battery_scenario"] = np.asarray(data["battery_scenario"])[source_indices]
    if "hard_eval_type" in data.files:
        groups["hard_eval_type"] = np.asarray(data["hard_eval_type"])[source_indices]
    return groups


def evaluate_one_mode(
    split_name: str,
    dataset: MixedPhaseDataset,
    pred: dict[str, Any],
    output_dir: Path,
    mode: str,
    threshold: float,
    labels: list[str],
    min_predictions: int,
    max_predictions: int,
    group_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    mode_prefix = f"{split_name}_{mode}"
    metrics = compute_metrics(
        pred["y_true"],
        pred["probabilities"],
        pred["component_phase_indices"],
        pred["n_phases"],
        pred["major_phase_index"],
        threshold,
        min_predictions,
        max_predictions,
    )
    metrics["loss"] = float(pred["loss"])
    metrics["phase_loss"] = float(pred["phase_loss"])
    metrics["fraction_level_loss"] = float(pred["fraction_level_loss"])
    metrics["fraction_level_accuracy"] = float(pred["fraction_level_accuracy"])
    metrics["threshold_mode"] = mode

    write_csv(
        output_dir / f"{mode_prefix}_per_phase_metrics.csv",
        per_phase_metrics(pred["y_true"], pred["probabilities"], labels, threshold, min_predictions, max_predictions),
    )
    write_csv(
        output_dir / f"{mode_prefix}_mixture_size_metrics.csv",
        evaluate_by_mixture_size(
            pred["y_true"],
            pred["probabilities"],
            pred["component_phase_indices"],
            pred["n_phases"],
            pred["major_phase_index"],
            threshold,
            min_predictions,
            max_predictions,
        ),
    )
    save_predictions_csv(
        output_dir / f"{mode_prefix}_predictions.csv",
        split_name,
        pred,
        labels,
        threshold,
        min_predictions,
        max_predictions,
    )

    group_summary: dict[str, list[dict[str, Any]]] = {}
    minor_bins = group_values.get("minor_fraction_bin")
    if minor_bins is None:
        minor_bins = inferred_minor_bins(pred["component_fractions"])
    groups_to_write = dict(group_values)
    groups_to_write["minor_fraction_bin"] = minor_bins
    for group_name, values in groups_to_write.items():
        rows = group_metrics_from_array(pred, values, group_name, threshold, min_predictions, max_predictions)
        if rows:
            group_summary[group_name] = rows
            write_csv(output_dir / f"{mode_prefix}_{group_name}_metrics.csv", rows)

    return {"metrics": metrics, "groups": group_summary}


def calibrate_on_dataset(
    model: nn.Module,
    dataset: MixedPhaseDataset,
    batch_size: int,
    num_workers: int,
    phase_criterion: nn.Module,
    fraction_level_criterion: nn.Module,
    lambda_fraction_level: float,
    device: torch.device,
    threshold_values: np.ndarray,
    min_predictions: int,
    max_predictions: int,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    pred = predict_split(
        model,
        dataset,
        batch_size,
        num_workers,
        phase_criterion,
        fraction_level_criterion,
        lambda_fraction_level,
        device,
    )
    return calibrate_threshold(
        pred["y_true"],
        pred["probabilities"],
        pred["component_phase_indices"],
        pred["n_phases"],
        pred["major_phase_index"],
        threshold_values,
        min_predictions,
        max_predictions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a mixed-phase CNN checkpoint on mixed-phase hard benchmarks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_CALIBRATION_DATA)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", action="append", default=None, help="NPZ stem to evaluate. Defaults to all npz files.")
    parser.add_argument("--threshold-mode", choices=THRESHOLD_MODES, default="all")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=37)
    parser.add_argument("--min-predictions", type=int, default=1)
    parser.add_argument("--max-predictions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = make_output_dir(args)
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing evaluation directory. "
            "Choose --output-dir or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = resolve_device(args.device)
        model, checkpoint = load_checkpoint_model(args.checkpoint, device)
        labels_from_checkpoint = checkpoint_labels(checkpoint)
        model_config = checkpoint.get("model_config", {})
        num_classes = int(model_config.get("num_phase_classes", len(labels_from_checkpoint or [])))
        if num_classes <= 0:
            raise ValueError("Could not infer num_phase_classes from checkpoint.")
        phase_criterion = make_phase_criterion(num_classes, device)
        fraction_level_criterion = nn.CrossEntropyLoss()
        run_config = checkpoint.get("run_config", {})
        lambda_fraction_level = float(run_config.get("loss", {}).get("lambda_fraction_level", 0.0)) if isinstance(run_config, dict) else 0.0
        if not bool(model_config.get("use_fraction_level_head", False)):
            lambda_fraction_level = 0.0

        threshold_values = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float32)
        modes = threshold_modes_for_arg(args.threshold_mode)
        fixed_threshold = args.fixed_threshold
        if fixed_threshold is None:
            fixed_threshold = float(checkpoint.get("best_threshold", 0.5))

        val_calibrated_threshold = None
        val_calibrated_metrics = None
        if "val_calibrated" in modes:
            calibration_dataset = MixedPhaseDataset(args.calibration_data)
            if labels_from_checkpoint is not None and calibration_dataset.labels != labels_from_checkpoint:
                raise ValueError("Calibration dataset labels do not match checkpoint labels.")
            val_calibrated_threshold, val_rows, val_calibrated_metrics = calibrate_on_dataset(
                model,
                calibration_dataset,
                args.batch_size,
                args.num_workers,
                phase_criterion,
                fraction_level_criterion,
                lambda_fraction_level,
                device,
                threshold_values,
                args.min_predictions,
                args.max_predictions,
            )
            write_csv(output_dir / "val_calibrated_threshold_calibration.csv", val_rows)
            save_threshold_curve(val_rows, output_dir / "val_calibrated_threshold_calibration.png")

        split_names = discover_splits(args.data_dir, args.split)
        if not split_names:
            raise ValueError(f"No .npz split files found in {args.data_dir}.")

        summary: dict[str, Any] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "calibration_data": str(args.calibration_data),
            "output_dir": str(output_dir),
            "device": str(device),
            "threshold_mode": args.threshold_mode,
            "fixed_threshold": float(fixed_threshold),
            "val_calibrated_threshold": float(val_calibrated_threshold) if val_calibrated_threshold is not None else None,
            "val_calibrated_metrics": val_calibrated_metrics,
            "min_predictions": int(args.min_predictions),
            "max_predictions": int(args.max_predictions),
            "splits": {},
        }
        write_json(output_dir / "run_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

        print(f"Device: {device}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Data dir: {args.data_dir}")
        print(f"Output dir: {output_dir}")

        for split_name in split_names:
            split_path = args.data_dir / f"{split_name}.npz"
            dataset = MixedPhaseDataset(split_path)
            if labels_from_checkpoint is not None and dataset.labels != labels_from_checkpoint:
                raise ValueError(f"{split_name} labels do not match checkpoint labels.")
            pred = predict_split(
                model,
                dataset,
                args.batch_size,
                args.num_workers,
                phase_criterion,
                fraction_level_criterion,
                lambda_fraction_level,
                device,
            )
            group_values = load_group_values(split_path, pred["source_index"].astype(np.int64))
            split_summary: dict[str, Any] = {}
            for mode in modes:
                if mode == "fixed":
                    threshold = float(fixed_threshold)
                elif mode == "val_calibrated":
                    if val_calibrated_threshold is None:
                        raise RuntimeError("val_calibrated threshold was not computed.")
                    threshold = float(val_calibrated_threshold)
                elif mode == "oracle":
                    threshold, oracle_rows, oracle_metrics = calibrate_threshold(
                        pred["y_true"],
                        pred["probabilities"],
                        pred["component_phase_indices"],
                        pred["n_phases"],
                        pred["major_phase_index"],
                        threshold_values,
                        args.min_predictions,
                        args.max_predictions,
                    )
                    write_csv(output_dir / f"{split_name}_oracle_threshold_calibration.csv", oracle_rows)
                    split_summary["oracle_calibration_best"] = oracle_metrics
                else:
                    raise ValueError(f"Unknown threshold mode: {mode}")
                split_summary[mode] = evaluate_one_mode(
                    split_name,
                    dataset,
                    pred,
                    output_dir,
                    mode,
                    threshold,
                    dataset.labels,
                    args.min_predictions,
                    args.max_predictions,
                    group_values,
                )
                print(
                    f"{split_name} [{mode}] "
                    f"thr={threshold:.3f} "
                    f"micro_f1={split_summary[mode]['metrics']['micro_f1']:.4f} "
                    f"minor_recall={split_summary[mode]['metrics']['minor_phase_recall']:.4f}"
                )
            summary["splits"][split_name] = split_summary

        write_json(output_dir / "metrics_summary.json", summary)
        print(f"Saved metrics: {output_dir / 'metrics_summary.json'}")
    except Exception as exc:
        print(f"Mixed-phase evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
