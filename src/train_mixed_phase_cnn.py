from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model_mixture import MixedPhaseXRDConvNet, load_single_phase_backbone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v1"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models" / "mixed_phase"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "mixed_phase"
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "logs" / "mixed_phase"
DEFAULT_RUN_PREFIX = "mixed_phase_cnn"
RANDOM_SEED = 42
REQUIRED_FIELDS = [
    "X",
    "y_multi",
    "y_fraction_level",
    "labels",
    "component_phase_indices",
    "component_fractions",
    "n_phases",
    "major_phase_index",
]


class MixedPhaseDataset(Dataset):
    def __init__(self, npz_path: Path, max_samples: int | None = None, seed: int = RANDOM_SEED) -> None:
        data = np.load(npz_path, allow_pickle=True)
        missing = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"{npz_path} is missing fields: {missing}")

        n_samples = int(data["X"].shape[0])
        if max_samples is not None and max_samples > 0 and max_samples < n_samples:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
        else:
            indices = np.arange(n_samples)

        self.X = torch.from_numpy(data["X"][indices].astype(np.float32)).unsqueeze(1)
        self.y_multi = torch.from_numpy(data["y_multi"][indices].astype(np.float32))
        self.y_fraction_level = torch.from_numpy(data["y_fraction_level"][indices].astype(np.int64))
        self.component_phase_indices = torch.from_numpy(data["component_phase_indices"][indices].astype(np.int64))
        self.component_fractions = torch.from_numpy(data["component_fractions"][indices].astype(np.float32))
        self.n_phases = torch.from_numpy(data["n_phases"][indices].astype(np.int64))
        self.major_phase_index = torch.from_numpy(data["major_phase_index"][indices].astype(np.int64))
        self.labels = [str(label) for label in data["labels"]]
        self.source_indices = indices.astype(np.int64)
        if "component_source_indices" in data.files:
            self.component_source_indices = torch.from_numpy(data["component_source_indices"][indices].astype(np.int64))
        else:
            self.component_source_indices = torch.full_like(self.component_phase_indices, -1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self.X[idx],
            "y_multi": self.y_multi[idx],
            "y_fraction_level": self.y_fraction_level[idx],
            "component_phase_indices": self.component_phase_indices[idx],
            "component_source_indices": self.component_source_indices[idx],
            "component_fractions": self.component_fractions[idx],
            "n_phases": self.n_phases[idx],
            "major_phase_index": self.major_phase_index[idx],
            "source_index": torch.tensor(int(self.source_indices[idx]), dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def make_run_name(run_name: str | None) -> str:
    if run_name:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip())
        return safe or f"{DEFAULT_RUN_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"{DEFAULT_RUN_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_pos_weight(dataset: MixedPhaseDataset, max_pos_weight: float) -> torch.Tensor:
    positives = dataset.y_multi.sum(dim=0)
    negatives = float(len(dataset)) - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    return torch.clamp(pos_weight, min=1.0, max=max_pos_weight).float()


def compute_sample_weights(
    dataset: MixedPhaseDataset,
    ternary_weight: float,
    minor_weight: float,
    minor_threshold: float,
) -> torch.Tensor:
    weights = torch.ones(len(dataset), dtype=torch.float64)
    if ternary_weight > 1.0:
        weights = torch.where(dataset.n_phases >= 3, weights * ternary_weight, weights)
    if minor_weight > 1.0:
        fractions = dataset.component_fractions.clone()
        fractions = torch.where(fractions > 0, fractions, torch.ones_like(fractions))
        min_fraction = fractions.min(dim=1).values
        has_minor = (dataset.n_phases > 1) & (min_fraction <= minor_threshold)
        weights = torch.where(has_minor, weights * minor_weight, weights)
    return weights


def fraction_level_loss_and_accuracy(
    outputs: dict[str, torch.Tensor],
    y_fraction_level: torch.Tensor,
    y_multi: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, float]:
    logits = outputs.get("fraction_level_logits")
    if logits is None:
        return torch.tensor(0.0, device=y_multi.device), 0.0

    present_mask = y_multi > 0.5
    if not bool(present_mask.any()):
        return torch.tensor(0.0, device=y_multi.device), 0.0

    targets = (y_fraction_level[present_mask] - 1).long()
    present_logits = logits[present_mask]
    loss = criterion(present_logits, targets)
    accuracy = float((present_logits.argmax(dim=1) == targets).float().mean().detach().item())
    return loss, accuracy


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> np.ndarray:
    pred = probabilities >= threshold
    if min_predictions > 0:
        for row_idx in np.where(pred.sum(axis=1) < min_predictions)[0]:
            top = np.argsort(-probabilities[row_idx])[:min_predictions]
            pred[row_idx, top] = True
    if max_predictions > 0:
        too_many = np.where(pred.sum(axis=1) > max_predictions)[0]
        for row_idx in too_many:
            keep = np.argsort(-probabilities[row_idx])[:max_predictions]
            pred[row_idx, :] = False
            pred[row_idx, keep] = True
    return pred


def precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    component_phase_indices: np.ndarray,
    n_phases: np.ndarray,
    major_phase_index: np.ndarray,
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> dict[str, float]:
    true = y_true >= 0.5
    pred = apply_threshold(probabilities, threshold, min_predictions, max_predictions)

    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(tp, fp, fn)

    class_tp = np.logical_and(pred, true).sum(axis=0).astype(np.float64)
    class_fp = np.logical_and(pred, ~true).sum(axis=0).astype(np.float64)
    class_fn = np.logical_and(~pred, true).sum(axis=0).astype(np.float64)
    support = true.sum(axis=0).astype(np.float64)
    valid = support > 0
    class_precision = class_tp / np.maximum(class_tp + class_fp, 1.0)
    class_recall = class_tp / np.maximum(class_tp + class_fn, 1.0)
    class_f1 = 2.0 * class_precision * class_recall / np.maximum(class_precision + class_recall, 1e-12)

    subset_accuracy = float(np.mean(np.all(pred == true, axis=1)))
    avg_false_positives = float(np.mean(np.logical_and(pred, ~true).sum(axis=1)))
    predicted_phase_count_mae = float(np.mean(np.abs(pred.sum(axis=1) - true.sum(axis=1))))
    major_top1 = probabilities.argmax(axis=1)
    major_phase_accuracy = float(np.mean(major_top1 == major_phase_index))

    minor_hits = 0.0
    minor_total = 0.0
    for row_idx in range(component_phase_indices.shape[0]):
        for component_idx in range(1, int(n_phases[row_idx])):
            phase_idx = int(component_phase_indices[row_idx, component_idx])
            if phase_idx >= 0:
                minor_total += 1.0
                minor_hits += float(pred[row_idx, phase_idx])
    minor_phase_recall = minor_hits / max(minor_total, 1.0)

    metrics = {
        "threshold": float(threshold),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean(class_precision[valid])) if np.any(valid) else 0.0,
        "macro_recall": float(np.mean(class_recall[valid])) if np.any(valid) else 0.0,
        "macro_f1": float(np.mean(class_f1[valid])) if np.any(valid) else 0.0,
        "subset_accuracy": subset_accuracy,
        "major_phase_accuracy": major_phase_accuracy,
        "minor_phase_recall": float(minor_phase_recall),
        "avg_false_positives_per_sample": avg_false_positives,
        "predicted_phase_count_mae": predicted_phase_count_mae,
        "sample_count": int(y_true.shape[0]),
    }

    for k in (2, 3, 5):
        top_k = np.argsort(-probabilities, axis=1)[:, : min(k, probabilities.shape[1])]
        hit_fraction = []
        all_hit = []
        for row_idx, top_indices in enumerate(top_k):
            true_indices = set(np.where(true[row_idx])[0].tolist())
            top_set = set(int(idx) for idx in top_indices)
            hits = len(true_indices & top_set)
            hit_fraction.append(hits / max(len(true_indices), 1))
            all_hit.append(float(true_indices.issubset(top_set)))
        metrics[f"top{k}_recall"] = float(np.mean(hit_fraction))
        metrics[f"top{k}_all_phases_hit"] = float(np.mean(all_hit))

    return metrics


def per_phase_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> list[dict[str, Any]]:
    true = y_true >= 0.5
    pred = apply_threshold(probabilities, threshold, min_predictions, max_predictions)
    rows: list[dict[str, Any]] = []
    for phase_idx, label in enumerate(labels):
        tp = float(np.logical_and(pred[:, phase_idx], true[:, phase_idx]).sum())
        fp = float(np.logical_and(pred[:, phase_idx], ~true[:, phase_idx]).sum())
        fn = float(np.logical_and(~pred[:, phase_idx], true[:, phase_idx]).sum())
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        rows.append(
            {
                "phase_idx": phase_idx,
                "phase_label": label,
                "support": int(true[:, phase_idx].sum()),
                "predicted": int(pred[:, phase_idx].sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def evaluate_by_mixture_size(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    component_phase_indices: np.ndarray,
    n_phases: np.ndarray,
    major_phase_index: np.ndarray,
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in sorted(int(value) for value in np.unique(n_phases)):
        mask = n_phases == size
        metrics = compute_metrics(
            y_true[mask],
            probabilities[mask],
            component_phase_indices[mask],
            n_phases[mask],
            major_phase_index[mask],
            threshold,
            min_predictions,
            max_predictions,
        )
        metrics["n_phases"] = size
        rows.append(metrics)
    return rows


@torch.no_grad()
def predict_split(
    model: nn.Module,
    dataset: MixedPhaseDataset,
    batch_size: int,
    num_workers: int,
    phase_criterion: nn.Module,
    fraction_level_criterion: nn.Module,
    lambda_fraction_level: float,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval()
    probabilities: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    component_phase_indices: list[np.ndarray] = []
    component_source_indices: list[np.ndarray] = []
    component_fractions: list[np.ndarray] = []
    n_phases: list[np.ndarray] = []
    major_phase_index: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    total_loss = 0.0
    total_phase_loss = 0.0
    total_fraction_level_loss = 0.0
    total_fraction_level_acc = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["X"])
        logits = outputs["phase_logits"]
        phase_loss = phase_criterion(logits, batch["y_multi"])
        fraction_level_loss, fraction_level_acc = fraction_level_loss_and_accuracy(
            outputs,
            batch["y_fraction_level"],
            batch["y_multi"],
            fraction_level_criterion,
        )
        loss = phase_loss + lambda_fraction_level * fraction_level_loss
        probs = torch.sigmoid(logits)
        batch_size_actual = int(batch["X"].shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_phase_loss += float(phase_loss.item()) * batch_size_actual
        total_fraction_level_loss += float(fraction_level_loss.item()) * batch_size_actual
        total_fraction_level_acc += float(fraction_level_acc) * batch_size_actual
        total_count += batch_size_actual

        probabilities.append(probs.cpu().numpy())
        y_true.append(batch["y_multi"].cpu().numpy())
        component_phase_indices.append(batch["component_phase_indices"].cpu().numpy())
        component_source_indices.append(batch["component_source_indices"].cpu().numpy())
        component_fractions.append(batch["component_fractions"].cpu().numpy())
        n_phases.append(batch["n_phases"].cpu().numpy())
        major_phase_index.append(batch["major_phase_index"].cpu().numpy())
        source_index.append(batch["source_index"].cpu().numpy())

    return {
        "loss": total_loss / max(total_count, 1),
        "phase_loss": total_phase_loss / max(total_count, 1),
        "fraction_level_loss": total_fraction_level_loss / max(total_count, 1),
        "fraction_level_accuracy": total_fraction_level_acc / max(total_count, 1),
        "probabilities": np.concatenate(probabilities, axis=0),
        "y_true": np.concatenate(y_true, axis=0),
        "component_phase_indices": np.concatenate(component_phase_indices, axis=0),
        "component_source_indices": np.concatenate(component_source_indices, axis=0),
        "component_fractions": np.concatenate(component_fractions, axis=0),
        "n_phases": np.concatenate(n_phases, axis=0),
        "major_phase_index": np.concatenate(major_phase_index, axis=0),
        "source_index": np.concatenate(source_index, axis=0),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    phase_criterion: nn.Module,
    fraction_level_criterion: nn.Module,
    lambda_fraction_level: float,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_phase_loss = 0.0
    total_fraction_level_loss = 0.0
    total_fraction_level_acc = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["X"])
        logits = outputs["phase_logits"]
        phase_loss = phase_criterion(logits, batch["y_multi"])
        fraction_level_loss, fraction_level_acc = fraction_level_loss_and_accuracy(
            outputs,
            batch["y_fraction_level"],
            batch["y_multi"],
            fraction_level_criterion,
        )
        loss = phase_loss + lambda_fraction_level * fraction_level_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(batch["X"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_phase_loss += float(phase_loss.detach().item()) * batch_size
        total_fraction_level_loss += float(fraction_level_loss.detach().item()) * batch_size
        total_fraction_level_acc += float(fraction_level_acc) * batch_size
        total_count += batch_size
    return {
        "loss": total_loss / max(total_count, 1),
        "phase_loss": total_phase_loss / max(total_count, 1),
        "fraction_level_loss": total_fraction_level_loss / max(total_count, 1),
        "fraction_level_accuracy": total_fraction_level_acc / max(total_count, 1),
        "sample_count": total_count,
    }


def calibrate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    component_phase_indices: np.ndarray,
    n_phases: np.ndarray,
    major_phase_index: np.ndarray,
    threshold_values: np.ndarray,
    min_predictions: int,
    max_predictions: int,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    best_threshold = float(threshold_values[0])
    best_metrics: dict[str, float] | None = None
    best_f1 = -1.0
    for threshold in threshold_values:
        metrics = compute_metrics(
            y_true,
            probabilities,
            component_phase_indices,
            n_phases,
            major_phase_index,
            float(threshold),
            min_predictions,
            max_predictions,
        )
        rows.append(metrics)
        if metrics["micro_f1"] > best_f1:
            best_f1 = float(metrics["micro_f1"])
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Threshold calibration produced no metrics.")
    return best_threshold, rows, best_metrics


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def join_indices(indices: np.ndarray) -> str:
    return ";".join(str(int(idx)) for idx in indices if int(idx) >= 0)


def join_labels(indices: np.ndarray, labels: list[str]) -> str:
    return ";".join(labels[int(idx)] for idx in indices if int(idx) >= 0)


def join_floats(values: np.ndarray, count: int | None = None) -> str:
    if count is None:
        filtered = [float(value) for value in values if float(value) > 0]
    else:
        filtered = [float(value) for value in values[:count]]
    return ";".join(f"{value:.6f}" for value in filtered)


def save_predictions_csv(
    output_path: Path,
    split_name: str,
    pred: dict[str, Any],
    labels: list[str],
    threshold: float,
    min_predictions: int,
    max_predictions: int,
) -> None:
    probabilities = pred["probabilities"]
    binary = apply_threshold(probabilities, threshold, min_predictions, max_predictions)
    top5 = np.argsort(-probabilities, axis=1)[:, : min(5, probabilities.shape[1])]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_index",
                "n_phases",
                "true_phase_indices",
                "true_phase_labels",
                "true_fractions",
                "component_source_indices",
                "major_phase_index",
                "major_phase_label",
                "pred_phase_indices",
                "pred_phase_labels",
                "pred_probabilities",
                "top5_phase_indices",
                "top5_phase_labels",
                "top5_probabilities",
            ]
        )
        for row_idx in range(probabilities.shape[0]):
            true_indices = pred["component_phase_indices"][row_idx]
            pred_indices = np.where(binary[row_idx])[0]
            top_indices = top5[row_idx]
            n_phase = int(pred["n_phases"][row_idx])
            major_idx = int(pred["major_phase_index"][row_idx])
            writer.writerow(
                [
                    split_name,
                    int(pred["source_index"][row_idx]),
                    n_phase,
                    join_indices(true_indices),
                    join_labels(true_indices, labels),
                    join_floats(pred["component_fractions"][row_idx], n_phase),
                    join_indices(pred["component_source_indices"][row_idx]),
                    major_idx,
                    labels[major_idx],
                    join_indices(pred_indices),
                    join_labels(pred_indices, labels),
                    join_floats(probabilities[row_idx, pred_indices], len(pred_indices)),
                    join_indices(top_indices),
                    join_labels(top_indices, labels),
                    join_floats(probabilities[row_idx, top_indices], len(top_indices)),
                ]
            )


def save_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    width, height = 1250, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((36, 20), "Mixed-phase CNN training curves", fill="black", font=font)

    epochs = [int(row["epoch"]) for row in history]
    panels = [
        ((45, 70, 600, 330), "BCE loss", [("train", [row["train"]["loss"] for row in history], (30, 95, 180)), ("val", [row["val"]["loss"] for row in history], (210, 85, 45))]),
        ((675, 70, 1230, 330), "Validation F1", [("micro", [row["val"]["micro_f1"] for row in history], (30, 95, 180)), ("macro", [row["val"]["macro_f1"] for row in history], (130, 80, 170))]),
        ((45, 410, 600, 690), "Validation precision/recall", [("precision", [row["val"]["micro_precision"] for row in history], (40, 145, 90)), ("recall", [row["val"]["micro_recall"] for row in history], (210, 85, 45))]),
        (
            (675, 410, 1230, 690),
            "Mixture behavior",
            [
                ("major acc", [row["val"]["major_phase_accuracy"] for row in history], (30, 95, 180)),
                ("minor recall", [row["val"]["minor_phase_recall"] for row in history], (210, 85, 45)),
                ("fraction level", [row["val"].get("fraction_level_accuracy", 0.0) for row in history], (130, 80, 170)),
            ],
        ),
    ]
    for box, title, series in panels:
        draw_line_panel(draw, box, title, epochs, series, font)
    canvas.save(output_path)


def save_threshold_curve(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    width, height = 900, 520
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    thresholds = [row["threshold"] for row in rows]
    series = [
        ("micro F1", [row["micro_f1"] for row in rows], (30, 95, 180)),
        ("precision", [row["micro_precision"] for row in rows], (40, 145, 90)),
        ("recall", [row["micro_recall"] for row in rows], (210, 85, 45)),
    ]
    draw.text((32, 18), "Validation threshold calibration", fill="black", font=font)
    draw_line_panel(draw, (45, 65, 855, 470), "metric vs threshold", thresholds, series, font)
    canvas.save(output_path)


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_values: list[float] | list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    plot_left = x0 + 50
    plot_top = y0 + 36
    plot_right = x1 - 20
    plot_bottom = y1 - 44
    draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)
    draw.text((x0 + 12, y0 + 10), title, fill="black", font=font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(80, 80, 80), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(80, 80, 80), width=1)

    values = [float(value) for _, seq, _ in series for value in seq]
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    if abs(x_max - x_min) < 1e-12:
        x_min -= 1.0
        x_max += 1.0

    def map_point(x_value: float, y_value: float) -> tuple[int, int]:
        x_coord = plot_left + int((float(x_value) - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y_coord = plot_bottom - int((float(y_value) - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        return x_coord, y_coord

    legend_x = plot_left + 8
    legend_y = plot_top + 8
    for name, seq, color in series:
        points = [map_point(x_value, value) for x_value, value in zip(x_values, seq)]
        if len(points) == 1:
            x_coord, y_coord = points[0]
            draw.ellipse((x_coord - 3, y_coord - 3, x_coord + 3, y_coord + 3), fill=color)
        else:
            draw.line(points, fill=color, width=2)
        draw.rectangle((legend_x, legend_y + 3, legend_x + 12, legend_y + 11), fill=color)
        draw.text((legend_x + 18, legend_y), name, fill="black", font=font)
        legend_y += 16
    draw.text((plot_left, plot_bottom + 16), f"{x_min:.3g}", fill="black", font=font)
    draw.text((plot_right - 45, plot_bottom + 16), f"{x_max:.3g}", fill="black", font=font)
    draw.text((plot_left, plot_top - 18), f"{y_max:.4g}", fill="black", font=font)
    draw.text((plot_left, plot_bottom - 12), f"{y_min:.4g}", fill="black", font=font)


def build_checkpoint_payload(
    model: nn.Module,
    epoch: int,
    threshold: float,
    val_metrics: dict[str, float],
    config: dict[str, Any],
    labels: list[str],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_epoch": int(epoch),
        "best_threshold": float(threshold),
        "best_val_micro_f1": float(val_metrics["micro_f1"]),
        "best_val_metrics": val_metrics,
        "phase_labels": labels,
        "model_config": config["model_config"],
        "run_config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mixed-phase multi-label 1D-CNN for XRD phase recognition.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--include-pretrained-phase-head", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=0)
    parser.add_argument("--lambda-fraction-level", type=float, default=0.25)
    parser.add_argument("--disable-fraction-level-head", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    parser.add_argument("--ternary-sample-weight", type=float, default=1.50)
    parser.add_argument("--minor-sample-weight", type=float, default=1.50)
    parser.add_argument("--minor-sample-threshold", type=float, default=0.15)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=37)
    parser.add_argument("--min-predictions", type=int, default=1)
    parser.add_argument("--max-predictions", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_name = make_run_name(args.run_name)
    if args.model_dir is None:
        args.model_dir = args.models_root / run_name
    if args.results_dir is None:
        args.results_dir = args.results_root / run_name
    if args.logs_dir is None:
        args.logs_dir = args.logs_root / run_name
    if not args.overwrite_run:
        occupied = [path for path in [args.model_dir, args.results_dir, args.logs_dir] if path_has_files(path)]
        if occupied:
            print(
                "Refusing to overwrite an existing mixed-phase run directory. "
                "Choose a new --run-name or pass --overwrite-run.\n"
                + "\n".join(str(path) for path in occupied),
                file=sys.stderr,
            )
            return 1

    ensure_dirs(args.model_dir, args.results_dir, args.logs_dir)
    set_seed(args.seed)

    try:
        device = resolve_device(args.device)
        train_dataset = MixedPhaseDataset(args.data_dir / "train.npz", args.max_train_samples, args.seed)
        val_dataset = MixedPhaseDataset(args.data_dir / "val.npz", args.max_val_samples, args.seed + 1)
        if train_dataset.labels != val_dataset.labels:
            raise ValueError("Train and val phase labels do not match.")

        sampler = None
        shuffle_train = True
        sample_weight_summary = None
        if not args.disable_weighted_sampling:
            sample_weights = compute_sample_weights(
                train_dataset,
                ternary_weight=float(args.ternary_sample_weight),
                minor_weight=float(args.minor_sample_weight),
                minor_threshold=float(args.minor_sample_threshold),
            )
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
            shuffle_train = False
            sample_weight_summary = {
                "min": float(sample_weights.min().item()),
                "max": float(sample_weights.max().item()),
                "mean": float(sample_weights.mean().item()),
                "ternary_sample_weight": float(args.ternary_sample_weight),
                "minor_sample_weight": float(args.minor_sample_weight),
                "minor_sample_threshold": float(args.minor_sample_threshold),
            }
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        pos_weight = compute_pos_weight(train_dataset, args.max_pos_weight).to(device)
        phase_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        fraction_level_criterion = nn.CrossEntropyLoss()
        use_fraction_level_head = not args.disable_fraction_level_head and args.lambda_fraction_level > 0.0

        model_config = {
            "num_phase_classes": len(train_dataset.labels),
            "pooled_bins": 32,
            "dropout": float(args.dropout),
            "hidden_dim": int(args.hidden_dim),
            "head_hidden_dim": int(args.head_hidden_dim) if args.head_hidden_dim > 0 else None,
            "use_fraction_level_head": bool(use_fraction_level_head),
            "fraction_level_count": 3,
        }
        model = MixedPhaseXRDConvNet(**model_config).to(device)
        pretrained_info = None
        if args.pretrained_checkpoint is not None:
            pretrained_info = load_single_phase_backbone(
                model,
                args.pretrained_checkpoint,
                device,
                include_phase_head=bool(args.include_pretrained_phase_head),
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        threshold_values = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float32)

        run_config = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_name": run_name,
            "data_dir": str(args.data_dir),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_phase_classes": len(train_dataset.labels),
            "train_sample_count": len(train_dataset),
            "val_sample_count": len(val_dataset),
            "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "model_config": model_config,
            "pos_weight": {
                "min": float(pos_weight.min().detach().cpu()),
                "max": float(pos_weight.max().detach().cpu()),
                "mean": float(pos_weight.mean().detach().cpu()),
                "max_pos_weight": float(args.max_pos_weight),
            },
            "sample_weights": sample_weight_summary,
            "loss": {
                "phase_loss": "BCEWithLogitsLoss",
                "fraction_level_loss": "present-phase CrossEntropyLoss over low/medium/high",
                "lambda_fraction_level": float(args.lambda_fraction_level),
            },
            "pretrained": pretrained_info,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        write_json(args.results_dir / "run_config.json", run_config)

        best_path = args.model_dir / "best_mixed_phase_cnn.pt"
        last_path = args.model_dir / "last_mixed_phase_cnn.pt"
        log_path = args.logs_dir / "training_log.jsonl"
        history: list[dict[str, Any]] = []
        best_val_f1 = -1.0
        best_epoch = 0
        best_threshold = 0.5

        print(f"Device: {device} | Run: {run_name}")
        print(f"Data: {args.data_dir}")
        print(f"Model dir: {args.model_dir}")
        if pretrained_info is not None:
            print(f"Loaded pretrained keys: {pretrained_info['loaded_key_count']}")

        with log_path.open("w", encoding="utf-8") as log_handle:
            for epoch in range(1, args.epochs + 1):
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    phase_criterion,
                    fraction_level_criterion,
                    args.lambda_fraction_level,
                    device,
                    args.grad_clip,
                )
                val_pred = predict_split(
                    model,
                    val_dataset,
                    args.batch_size,
                    args.num_workers,
                    phase_criterion,
                    fraction_level_criterion,
                    args.lambda_fraction_level,
                    device,
                )
                val_threshold, _, val_metrics = calibrate_threshold(
                    val_pred["y_true"],
                    val_pred["probabilities"],
                    val_pred["component_phase_indices"],
                    val_pred["n_phases"],
                    val_pred["major_phase_index"],
                    threshold_values,
                    args.min_predictions,
                    args.max_predictions,
                )
                val_metrics["loss"] = float(val_pred["loss"])
                val_metrics["phase_loss"] = float(val_pred["phase_loss"])
                val_metrics["fraction_level_loss"] = float(val_pred["fraction_level_loss"])
                val_metrics["fraction_level_accuracy"] = float(val_pred["fraction_level_accuracy"])
                scheduler.step(float(val_pred["loss"]))

                row = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "threshold": val_threshold,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                history.append(row)
                log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_handle.flush()

                print(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                    f"val_major_acc={val_metrics['major_phase_accuracy']:.4f} "
                    f"val_minor_recall={val_metrics['minor_phase_recall']:.4f} "
                    f"val_frac_lvl_acc={val_metrics['fraction_level_accuracy']:.4f} "
                    f"thr={val_threshold:.3f}"
                )

                if val_metrics["micro_f1"] > best_val_f1:
                    best_val_f1 = float(val_metrics["micro_f1"])
                    best_epoch = epoch
                    best_threshold = float(val_threshold)
                    torch.save(
                        build_checkpoint_payload(
                            model,
                            epoch,
                            best_threshold,
                            val_metrics,
                            run_config,
                            train_dataset.labels,
                            optimizer,
                        ),
                        best_path,
                    )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": int(args.epochs),
                "best_epoch": int(best_epoch),
                "best_threshold": float(best_threshold),
                "best_val_micro_f1": float(best_val_f1),
                "phase_labels": train_dataset.labels,
                "model_config": model_config,
                "run_config": run_config,
            },
            last_path,
        )

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_threshold = float(checkpoint["best_threshold"])

        metrics_summary: dict[str, Any] = {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_threshold": best_threshold,
            "best_val_micro_f1": float(checkpoint["best_val_micro_f1"]),
            "checkpoint": str(best_path),
        }

        split_max_samples = {
            "val": args.max_val_samples,
            "test_normal": args.max_test_samples,
            "test_hard": args.max_test_samples,
        }
        threshold_rows_for_plot: list[dict[str, float]] | None = None
        for split_name in ("val", "test_normal", "test_hard"):
            dataset = MixedPhaseDataset(args.data_dir / f"{split_name}.npz", split_max_samples[split_name], args.seed + 11)
            if dataset.labels != train_dataset.labels:
                raise ValueError(f"{split_name} phase labels do not match training labels.")
            pred = predict_split(
                model,
                dataset,
                args.batch_size,
                args.num_workers,
                phase_criterion,
                fraction_level_criterion,
                args.lambda_fraction_level,
                device,
            )

            if split_name == "val":
                _, threshold_rows, _ = calibrate_threshold(
                    pred["y_true"],
                    pred["probabilities"],
                    pred["component_phase_indices"],
                    pred["n_phases"],
                    pred["major_phase_index"],
                    threshold_values,
                    args.min_predictions,
                    args.max_predictions,
                )
                threshold_rows_for_plot = threshold_rows
                write_csv(args.results_dir / "threshold_calibration.csv", threshold_rows)

            metrics = compute_metrics(
                pred["y_true"],
                pred["probabilities"],
                pred["component_phase_indices"],
                pred["n_phases"],
                pred["major_phase_index"],
                best_threshold,
                args.min_predictions,
                args.max_predictions,
            )
            metrics["loss"] = float(pred["loss"])
            metrics["phase_loss"] = float(pred["phase_loss"])
            metrics["fraction_level_loss"] = float(pred["fraction_level_loss"])
            metrics["fraction_level_accuracy"] = float(pred["fraction_level_accuracy"])
            metrics_summary[split_name] = metrics

            per_phase = per_phase_metrics(
                pred["y_true"],
                pred["probabilities"],
                dataset.labels,
                best_threshold,
                args.min_predictions,
                args.max_predictions,
            )
            write_csv(args.results_dir / f"{split_name}_per_phase_metrics.csv", per_phase)

            mixture_size_rows = evaluate_by_mixture_size(
                pred["y_true"],
                pred["probabilities"],
                pred["component_phase_indices"],
                pred["n_phases"],
                pred["major_phase_index"],
                best_threshold,
                args.min_predictions,
                args.max_predictions,
            )
            write_csv(args.results_dir / f"{split_name}_mixture_size_metrics.csv", mixture_size_rows)
            save_predictions_csv(
                args.results_dir / f"{split_name}_predictions.csv",
                split_name,
                pred,
                dataset.labels,
                best_threshold,
                args.min_predictions,
                args.max_predictions,
            )

        write_json(args.results_dir / "metrics_summary.json", metrics_summary)
        write_json(args.results_dir / "training_history.json", history)
        save_training_curves(history, args.results_dir / "training_curves.png")
        if threshold_rows_for_plot is not None:
            save_threshold_curve(threshold_rows_for_plot, args.results_dir / "threshold_calibration.png")

        print(f"Best epoch: {best_epoch} | best threshold: {best_threshold:.3f} | val micro-F1: {best_val_f1:.4f}")
        print(f"Saved checkpoint: {best_path}")
        print(f"Saved metrics: {args.results_dir / 'metrics_summary.json'}")
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
