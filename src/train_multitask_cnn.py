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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model_multitask import MultiTaskXRDConvNet
from physics import masked_peak_mae, masked_peak_mse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "logs"
DEFAULT_RUN_PREFIX = "multitask_cnn"
DEFAULT_MODEL_DIR = DEFAULT_MODELS_ROOT / DEFAULT_RUN_PREFIX
DEFAULT_RESULTS_DIR = DEFAULT_RESULTS_ROOT / DEFAULT_RUN_PREFIX
DEFAULT_LOG_DIR = DEFAULT_LOGS_ROOT / DEFAULT_RUN_PREFIX

RANDOM_SEED = 42
LATTICE_DIM = 3
LATTICE_NAMES = ["a", "b", "c"]
REQUIRED_FIELDS = [
    "X",
    "y",
    "labels",
    "lattice_params",
    "peak_hkls",
    "peak_2theta",
    "peak_mask",
    "crystal_system_y",
    "crystal_system_labels",
]


class XRDMultiTaskDataset(Dataset):
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
        self.phase_y = torch.from_numpy(data["y"][indices].astype(np.int64))
        self.crystal_y = torch.from_numpy(data["crystal_system_y"][indices].astype(np.int64))
        self.lattice_params = torch.from_numpy(data["lattice_params"][indices].astype(np.float32))
        self.peak_hkls = torch.from_numpy(data["peak_hkls"][indices].astype(np.float32))
        self.peak_2theta = torch.from_numpy(data["peak_2theta"][indices].astype(np.float32))
        self.peak_mask = torch.from_numpy(data["peak_mask"][indices].astype(np.float32))
        self.labels = [str(label) for label in data["labels"]]
        self.crystal_system_labels = [str(label) for label in data["crystal_system_labels"]]
        self.source_indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.phase_y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self.X[idx],
            "phase_y": self.phase_y[idx],
            "crystal_y": self.crystal_y[idx],
            "lattice_params": self.lattice_params[idx],
            "peak_hkls": self.peak_hkls[idx],
            "peak_2theta": self.peak_2theta[idx],
            "peak_mask": self.peak_mask[idx],
            "source_index": torch.tensor(self.source_indices[idx], dtype=torch.long),
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


def configure_fine_tuning(model: MultiTaskXRDConvNet, mode: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if mode == "none":
        return ["all"]

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_modules: list[str] = []
    if mode == "lattice_head":
        for parameter in model.lattice_head.parameters():
            parameter.requires_grad = True
        trainable_modules.append("lattice_head")
    elif mode == "lattice_late":
        for parameter in model.backbone[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.shared_projection.parameters():
            parameter.requires_grad = True
        for parameter in model.lattice_head.parameters():
            parameter.requires_grad = True
        trainable_modules.extend(["backbone_last_block", "shared_projection", "lattice_head"])
    else:
        raise ValueError(f"Unknown fine-tune mode: {mode}")

    return trainable_modules


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def denormalize_lattice(lattice_norm: torch.Tensor, lattice_mean: torch.Tensor, lattice_std: torch.Tensor) -> torch.Tensor:
    return lattice_norm * lattice_std + lattice_mean


def compute_phase_mean_abc(dataset: XRDMultiTaskDataset, num_phase_classes: int) -> torch.Tensor:
    sums = torch.zeros((num_phase_classes, LATTICE_DIM), dtype=torch.float32)
    counts = torch.zeros(num_phase_classes, dtype=torch.float32)
    abc = dataset.lattice_params[:, :LATTICE_DIM]
    for class_idx in range(num_phase_classes):
        mask = dataset.phase_y == class_idx
        if bool(mask.any()):
            sums[class_idx] = abc[mask].mean(dim=0)
            counts[class_idx] = float(mask.sum().item())
    global_mean = abc.mean(dim=0)
    missing = counts == 0
    if bool(missing.any()):
        sums[missing] = global_mean
    return sums


def phase_mean_for_batch(
    phase_logits: torch.Tensor,
    phase_y: torch.Tensor,
    phase_mean_abc: torch.Tensor,
    source: str,
) -> torch.Tensor:
    if source == "true":
        return phase_mean_abc[phase_y]
    if source == "pred":
        phase_prob = torch.softmax(phase_logits.detach(), dim=1)
        return phase_prob @ phase_mean_abc
    raise ValueError(f"Unknown phase mean source: {source}")


def lattice_prediction_abc(
    outputs: dict[str, torch.Tensor],
    phase_y: torch.Tensor,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    phase_mean_abc: torch.Tensor | None,
    lattice_mode: str,
    residual_scale: float,
    residual_mean_source: str,
) -> torch.Tensor:
    if lattice_mode == "absolute":
        return denormalize_lattice(outputs["lattice_norm"], lattice_mean, lattice_std)
    if lattice_mode == "phase_residual":
        if phase_mean_abc is None:
            raise ValueError("phase_mean_abc is required for phase_residual lattice mode.")
        base_abc = phase_mean_for_batch(outputs["phase_logits"], phase_y, phase_mean_abc, residual_mean_source)
        relative_delta = residual_scale * torch.tanh(outputs["lattice_raw"])
        return base_abc * (1.0 + relative_delta)
    raise ValueError(f"Unknown lattice mode: {lattice_mode}")


def lattice_training_loss(
    outputs: dict[str, torch.Tensor],
    lattice_target: torch.Tensor,
    phase_y: torch.Tensor,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    phase_mean_abc: torch.Tensor | None,
    lattice_mode: str,
    residual_scale: float,
    residual_mean_source: str,
    lattice_criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lattice_mode == "absolute":
        lattice_target_norm = (lattice_target - lattice_mean) / lattice_std
        loss = lattice_criterion(outputs["lattice_norm"], lattice_target_norm)
        pred_abc = denormalize_lattice(outputs["lattice_norm"], lattice_mean, lattice_std)
        return loss, pred_abc

    pred_abc = lattice_prediction_abc(
        outputs,
        phase_y,
        lattice_mean,
        lattice_std,
        phase_mean_abc,
        lattice_mode,
        residual_scale,
        residual_mean_source,
    )
    relative_error = (pred_abc - lattice_target) / torch.clamp(lattice_target, min=1e-6)
    loss = F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), beta=0.002)
    return loss, pred_abc


def full_lattice_from_pred(
    lattice_pred_abc: torch.Tensor,
    lattice_params_true: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([lattice_pred_abc, lattice_params_true[:, LATTICE_DIM:]], dim=1)


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    pred = logits.topk(k, dim=1).indices
    return int(pred.eq(target.unsqueeze(1)).any(dim=1).sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    phase_criterion: nn.Module,
    crystal_criterion: nn.Module,
    lattice_criterion: nn.Module,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    lambda_sys: float,
    lambda_lat: float,
    lambda_phys: float,
    lattice_mode: str,
    phase_mean_abc: torch.Tensor | None,
    residual_scale: float,
    residual_mean_source: str,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "phase_loss": 0.0,
        "crystal_loss": 0.0,
        "lattice_loss": 0.0,
        "physics_loss": 0.0,
        "phase_correct": 0.0,
        "phase_top3_correct": 0.0,
        "phase_top5_correct": 0.0,
        "crystal_correct": 0.0,
        "crystal_top3_correct": 0.0,
        "count": 0.0,
    }
    lattice_abs_sum = torch.zeros(LATTICE_DIM, dtype=torch.float64)

    for batch in loader:
        batch = move_batch(batch, device)
        X = batch["X"]
        phase_y = batch["phase_y"]
        crystal_y = batch["crystal_y"]
        lattice_params = batch["lattice_params"]
        lattice_target = lattice_params[:, :LATTICE_DIM]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(X)
            phase_loss = phase_criterion(outputs["phase_logits"], phase_y)
            crystal_loss = crystal_criterion(outputs["crystal_logits"], crystal_y)
            lattice_loss, lattice_pred_abc = lattice_training_loss(
                outputs,
                lattice_target,
                phase_y,
                lattice_mean,
                lattice_std,
                phase_mean_abc,
                lattice_mode,
                residual_scale,
                residual_mean_source,
                lattice_criterion,
            )

            if lambda_phys > 0:
                lattice_pred_full = full_lattice_from_pred(
                    lattice_pred_abc,
                    lattice_params,
                )
                physics_loss = masked_peak_mse(
                    lattice_pred_full,
                    batch["peak_hkls"],
                    batch["peak_2theta"],
                    batch["peak_mask"],
                )
            else:
                with torch.no_grad():
                    lattice_pred_full = full_lattice_from_pred(
                        lattice_pred_abc.detach(),
                        lattice_params,
                    )
                    physics_loss = masked_peak_mse(
                        lattice_pred_full,
                        batch["peak_hkls"],
                        batch["peak_2theta"],
                        batch["peak_mask"],
                    )

            loss = phase_loss + lambda_sys * crystal_loss + lambda_lat * lattice_loss + lambda_phys * physics_loss

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(phase_y.shape[0])
        lattice_abs_sum += torch.abs(lattice_pred_abc.detach().cpu() - lattice_target.detach().cpu()).sum(dim=0).double()

        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["phase_loss"] += float(phase_loss.detach().item()) * batch_size
        totals["crystal_loss"] += float(crystal_loss.detach().item()) * batch_size
        totals["lattice_loss"] += float(lattice_loss.detach().item()) * batch_size
        totals["physics_loss"] += float(physics_loss.detach().item()) * batch_size
        totals["phase_correct"] += float((outputs["phase_logits"].argmax(dim=1) == phase_y).sum().item())
        totals["phase_top3_correct"] += float(topk_correct(outputs["phase_logits"].detach(), phase_y, 3))
        totals["phase_top5_correct"] += float(topk_correct(outputs["phase_logits"].detach(), phase_y, 5))
        totals["crystal_correct"] += float((outputs["crystal_logits"].argmax(dim=1) == crystal_y).sum().item())
        totals["crystal_top3_correct"] += float(topk_correct(outputs["crystal_logits"].detach(), crystal_y, 3))
        totals["count"] += batch_size

    count = max(totals["count"], 1.0)
    lattice_mae = lattice_abs_sum.numpy() / count
    return {
        "loss": totals["loss"] / count,
        "phase_loss": totals["phase_loss"] / count,
        "crystal_loss": totals["crystal_loss"] / count,
        "lattice_loss": totals["lattice_loss"] / count,
        "physics_loss": totals["physics_loss"] / count,
        "phase_accuracy": totals["phase_correct"] / count,
        "phase_top3_accuracy": totals["phase_top3_correct"] / count,
        "phase_top5_accuracy": totals["phase_top5_correct"] / count,
        "crystal_accuracy": totals["crystal_correct"] / count,
        "crystal_top3_accuracy": totals["crystal_top3_correct"] / count,
        "lattice_mae_mean": float(np.mean(lattice_mae)),
        "lattice_mae_a": float(lattice_mae[0]),
        "lattice_mae_b": float(lattice_mae[1]),
        "lattice_mae_c": float(lattice_mae[2]),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    phase_mean_abc: torch.Tensor | None,
    lattice_mode: str,
    residual_scale: float,
    residual_mean_source: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {
        "source_index": [],
        "phase_true": [],
        "phase_pred": [],
        "phase_confidence": [],
        "phase_top3_hit": [],
        "phase_top5_hit": [],
        "crystal_true": [],
        "crystal_pred": [],
        "crystal_confidence": [],
        "crystal_top3_hit": [],
        "lattice_true": [],
        "lattice_pred": [],
        "peak_hkls": [],
        "peak_2theta": [],
        "peak_mask": [],
    }

    for batch in loader:
        batch = move_batch(batch, device)
        model_outputs = model(batch["X"])
        phase_probs = torch.softmax(model_outputs["phase_logits"], dim=1)
        crystal_probs = torch.softmax(model_outputs["crystal_logits"], dim=1)
        lattice_pred_abc = lattice_prediction_abc(
            model_outputs,
            batch["phase_y"],
            lattice_mean,
            lattice_std,
            phase_mean_abc,
            lattice_mode,
            residual_scale,
            residual_mean_source,
        )
        lattice_pred_full = torch.cat([lattice_pred_abc, batch["lattice_params"][:, LATTICE_DIM:]], dim=1)

        outputs["source_index"].append(batch["source_index"].cpu().numpy())
        outputs["phase_true"].append(batch["phase_y"].cpu().numpy())
        outputs["phase_pred"].append(phase_probs.argmax(dim=1).cpu().numpy())
        outputs["phase_confidence"].append(phase_probs.max(dim=1).values.cpu().numpy())
        outputs["phase_top3_hit"].append(
            phase_probs.topk(min(3, phase_probs.shape[1]), dim=1).indices.eq(batch["phase_y"].unsqueeze(1)).any(dim=1).cpu().numpy()
        )
        outputs["phase_top5_hit"].append(
            phase_probs.topk(min(5, phase_probs.shape[1]), dim=1).indices.eq(batch["phase_y"].unsqueeze(1)).any(dim=1).cpu().numpy()
        )
        outputs["crystal_true"].append(batch["crystal_y"].cpu().numpy())
        outputs["crystal_pred"].append(crystal_probs.argmax(dim=1).cpu().numpy())
        outputs["crystal_confidence"].append(crystal_probs.max(dim=1).values.cpu().numpy())
        outputs["crystal_top3_hit"].append(
            crystal_probs.topk(min(3, crystal_probs.shape[1]), dim=1).indices.eq(batch["crystal_y"].unsqueeze(1)).any(dim=1).cpu().numpy()
        )
        outputs["lattice_true"].append(batch["lattice_params"].cpu().numpy())
        outputs["lattice_pred"].append(lattice_pred_full.cpu().numpy())
        outputs["peak_hkls"].append(batch["peak_hkls"].cpu().numpy())
        outputs["peak_2theta"].append(batch["peak_2theta"].cpu().numpy())
        outputs["peak_mask"].append(batch["peak_mask"].cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_prefix: Path,
    title: str,
    show_labels: bool,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    np.savetxt(output_prefix.with_suffix(".csv"), cm, fmt="%d", delimiter=",")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(np.float32) / np.maximum(row_sums, 1)
    save_heatmap_png(cm_norm, labels, output_prefix.with_suffix(".png"), title, show_labels)
    return cm


def blue_heatmap_rgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)[..., None]
    white = np.array([255, 255, 255], dtype=np.float32)
    blue = np.array([30, 95, 180], dtype=np.float32)
    return (white * (1.0 - values) + blue * values).astype(np.uint8)


def save_heatmap_png(values: np.ndarray, labels: list[str], output_path: Path, title: str, show_labels: bool) -> None:
    font = ImageFont.load_default()
    n_rows, n_cols = values.shape
    if show_labels and n_rows <= 20:
        cell = 42
        left = 150
        top = 52
        bottom = 90
        right = 24
        heatmap = Image.fromarray(blue_heatmap_rgb(values), mode="RGB").resize((n_cols * cell, n_rows * cell), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (left + heatmap.width + right, top + heatmap.height + bottom), "white")
        canvas.paste(heatmap, (left, top))
        draw = ImageDraw.Draw(canvas)
        draw.text((left, 16), title, fill="black", font=font)
        for idx, label in enumerate(labels):
            y = top + idx * cell + cell // 3
            x = left + idx * cell + 4
            draw.text((8, y), label[:22], fill="black", font=font)
            draw.text((x, top + heatmap.height + 8), str(idx), fill="black", font=font)
        draw.text((left + heatmap.width // 2 - 35, top + heatmap.height + 38), "Predicted", fill="black", font=font)
        draw.text((8, top - 22), "True label", fill="black", font=font)
    else:
        size = 1200
        heatmap = Image.fromarray(blue_heatmap_rgb(values), mode="RGB").resize((size, size), Image.Resampling.NEAREST)
        top = 52
        margin = 34
        canvas = Image.new("RGB", (size + 2 * margin, size + top + margin), "white")
        canvas.paste(heatmap, (margin, top))
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 16), title, fill="black", font=font)
        draw.text((margin, top + size + 10), "Predicted label index", fill="black", font=font)
        draw.text((margin, top - 18), "True label index", fill="black", font=font)
    canvas.save(output_path)


def save_per_class_accuracy(cm: np.ndarray, labels: list[str], output_path: Path, label_column: str) -> None:
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    accuracy = correct.astype(np.float32) / np.maximum(support, 1)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_idx", label_column, "support", "correct", "accuracy"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label, int(support[idx]), int(correct[idx]), float(accuracy[idx])])


def save_top_confusions(cm: np.ndarray, labels: list[str], output_path: Path, top_k: int = 50) -> None:
    rows = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx or cm[true_idx, pred_idx] == 0:
                continue
            rows.append((int(cm[true_idx, pred_idx]), true_idx, pred_idx))
    rows.sort(reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "true_idx", "true_label", "pred_idx", "pred_label"])
        for count, true_idx, pred_idx in rows[:top_k]:
            writer.writerow([count, true_idx, labels[true_idx], pred_idx, labels[pred_idx]])


def save_predictions_csv(
    split_name: str,
    predictions: dict[str, np.ndarray],
    phase_labels: list[str],
    crystal_labels: list[str],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_index",
                "true_phase_idx",
                "true_phase_label",
                "pred_phase_idx",
                "pred_phase_label",
                "phase_confidence",
                "phase_top3_hit",
                "phase_top5_hit",
                "true_crystal_idx",
                "true_crystal_system",
                "pred_crystal_idx",
                "pred_crystal_system",
                "crystal_confidence",
                "crystal_top3_hit",
                "a_true",
                "a_pred",
                "b_true",
                "b_pred",
                "c_true",
                "c_pred",
            ]
        )
        for idx in range(len(predictions["phase_true"])):
            phase_true = int(predictions["phase_true"][idx])
            phase_pred = int(predictions["phase_pred"][idx])
            crystal_true = int(predictions["crystal_true"][idx])
            crystal_pred = int(predictions["crystal_pred"][idx])
            lattice_true = predictions["lattice_true"][idx]
            lattice_pred = predictions["lattice_pred"][idx]
            writer.writerow(
                [
                    split_name,
                    int(predictions["source_index"][idx]),
                    phase_true,
                    phase_labels[phase_true],
                    phase_pred,
                    phase_labels[phase_pred],
                    float(predictions["phase_confidence"][idx]),
                    bool(predictions["phase_top3_hit"][idx]),
                    bool(predictions["phase_top5_hit"][idx]),
                    crystal_true,
                    crystal_labels[crystal_true],
                    crystal_pred,
                    crystal_labels[crystal_pred],
                    float(predictions["crystal_confidence"][idx]),
                    bool(predictions["crystal_top3_hit"][idx]),
                    float(lattice_true[0]),
                    float(lattice_pred[0]),
                    float(lattice_true[1]),
                    float(lattice_pred[1]),
                    float(lattice_true[2]),
                    float(lattice_pred[2]),
                ]
            )


def evaluate_split(
    model: nn.Module,
    split_name: str,
    dataset: XRDMultiTaskDataset,
    batch_size: int,
    num_workers: int,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    phase_mean_abc: torch.Tensor | None,
    lattice_mode: str,
    residual_scale: float,
    residual_mean_source: str,
    device: torch.device,
    results_dir: Path,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    pred = predict(
        model,
        loader,
        lattice_mean,
        lattice_std,
        phase_mean_abc,
        lattice_mode,
        residual_scale,
        residual_mean_source,
        device,
    )

    phase_acc = float(accuracy_score(pred["phase_true"], pred["phase_pred"]))
    phase_top3_acc = float(np.mean(pred["phase_top3_hit"]))
    phase_top5_acc = float(np.mean(pred["phase_top5_hit"]))
    crystal_acc = float(accuracy_score(pred["crystal_true"], pred["crystal_pred"]))
    crystal_top3_acc = float(np.mean(pred["crystal_top3_hit"]))
    lattice_mae = np.mean(np.abs(pred["lattice_pred"][:, :LATTICE_DIM] - pred["lattice_true"][:, :LATTICE_DIM]), axis=0)
    bragg_mae = float(
        masked_peak_mae(
            torch.from_numpy(pred["lattice_pred"].astype(np.float32)),
            torch.from_numpy(pred["peak_hkls"].astype(np.float32)),
            torch.from_numpy(pred["peak_2theta"].astype(np.float32)),
            torch.from_numpy(pred["peak_mask"].astype(np.float32)),
        ).item()
    )

    phase_prefix = results_dir / f"{split_name}_phase_confusion_matrix"
    crystal_prefix = results_dir / f"{split_name}_crystal_confusion_matrix"
    phase_cm = save_confusion_outputs(
        pred["phase_true"],
        pred["phase_pred"],
        dataset.labels,
        phase_prefix,
        title=f"{split_name} phase confusion matrix",
        show_labels=False,
    )
    crystal_cm = save_confusion_outputs(
        pred["crystal_true"],
        pred["crystal_pred"],
        dataset.crystal_system_labels,
        crystal_prefix,
        title=f"{split_name} crystal-system confusion matrix",
        show_labels=True,
    )
    save_per_class_accuracy(
        phase_cm,
        dataset.labels,
        results_dir / f"{split_name}_phase_per_class_accuracy.csv",
        "phase_label",
    )
    save_per_class_accuracy(
        crystal_cm,
        dataset.crystal_system_labels,
        results_dir / f"{split_name}_crystal_per_class_accuracy.csv",
        "crystal_system",
    )
    save_top_confusions(phase_cm, dataset.labels, results_dir / f"{split_name}_phase_top_confusions.csv")
    save_predictions_csv(
        split_name,
        pred,
        dataset.labels,
        dataset.crystal_system_labels,
        results_dir / f"{split_name}_predictions.csv",
    )

    metrics = {
        "phase_accuracy": phase_acc,
        "phase_top3_accuracy": phase_top3_acc,
        "phase_top5_accuracy": phase_top5_acc,
        "crystal_accuracy": crystal_acc,
        "crystal_top3_accuracy": crystal_top3_acc,
        "lattice_mae": {name: float(value) for name, value in zip(LATTICE_NAMES, lattice_mae)},
        "lattice_mae_mean": float(np.mean(lattice_mae)),
        "bragg_peak_mae_deg": bragg_mae,
        "sample_count": int(len(dataset)),
    }
    if lattice_mode == "phase_residual" and residual_mean_source != "true":
        oracle_pred = predict(
            model,
            loader,
            lattice_mean,
            lattice_std,
            phase_mean_abc,
            lattice_mode,
            residual_scale,
            "true",
            device,
        )
        oracle_lattice_mae = np.mean(
            np.abs(oracle_pred["lattice_pred"][:, :LATTICE_DIM] - oracle_pred["lattice_true"][:, :LATTICE_DIM]),
            axis=0,
        )
        oracle_bragg_mae = float(
            masked_peak_mae(
                torch.from_numpy(oracle_pred["lattice_pred"].astype(np.float32)),
                torch.from_numpy(oracle_pred["peak_hkls"].astype(np.float32)),
                torch.from_numpy(oracle_pred["peak_2theta"].astype(np.float32)),
                torch.from_numpy(oracle_pred["peak_mask"].astype(np.float32)),
            ).item()
        )
        metrics["oracle_true_phase_lattice_mae"] = {
            name: float(value) for name, value in zip(LATTICE_NAMES, oracle_lattice_mae)
        }
        metrics["oracle_true_phase_lattice_mae_mean"] = float(np.mean(oracle_lattice_mae))
        metrics["oracle_true_phase_bragg_peak_mae_deg"] = oracle_bragg_mae
    return metrics


def save_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    font = ImageFont.load_default()
    width, height = 1280, 860
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [
        (
            (55, 55, 610, 390),
            "Total loss",
            [
                ("train", [row["train"]["loss"] for row in history], (30, 95, 180)),
                ("val", [row["val"]["loss"] for row in history], (210, 85, 45)),
            ],
        ),
        (
            (680, 55, 1235, 390),
            "Classification accuracy",
            [
                ("train phase", [row["train"]["phase_accuracy"] for row in history], (30, 95, 180)),
                ("val phase", [row["val"]["phase_accuracy"] for row in history], (210, 85, 45)),
                ("train crystal", [row["train"]["crystal_accuracy"] for row in history], (40, 145, 90)),
                ("val crystal", [row["val"]["crystal_accuracy"] for row in history], (130, 80, 170)),
            ],
        ),
        (
            (55, 465, 610, 800),
            "Mean lattice MAE (angstrom)",
            [
                ("train", [row["train"]["lattice_mae_mean"] for row in history], (30, 95, 180)),
                ("val", [row["val"]["lattice_mae_mean"] for row in history], (210, 85, 45)),
            ],
        ),
        (
            (680, 465, 1235, 800),
            "Bragg consistency MSE",
            [
                ("train", [row["train"]["physics_loss"] for row in history], (30, 95, 180)),
                ("val", [row["val"]["physics_loss"] for row in history], (210, 85, 45)),
            ],
        ),
    ]
    epochs = [int(row["epoch"]) for row in history]
    for box, title, series in panels:
        draw_line_panel(draw, box, title, epochs, series, font)
    canvas.save(output_path)


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    epochs: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    plot_left = x0 + 48
    plot_top = y0 + 36
    plot_right = x1 - 18
    plot_bottom = y1 - 48
    draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)
    draw.text((x0 + 12, y0 + 10), title, fill="black", font=font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(80, 80, 80), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(80, 80, 80), width=1)

    values = [value for _, seq, _ in series for value in seq]
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5
    x_min = min(epochs)
    x_max = max(epochs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def map_point(epoch: int, value: float) -> tuple[int, int]:
        x = plot_left + int((epoch - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y = plot_bottom - int((value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        return x, y

    legend_x = plot_left + 8
    legend_y = plot_top + 8
    for name, seq, color in series:
        points = [map_point(epoch, float(value)) for epoch, value in zip(epochs, seq)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=2)
        draw.rectangle((legend_x, legend_y + 3, legend_x + 12, legend_y + 11), fill=color)
        draw.text((legend_x + 18, legend_y), name, fill="black", font=font)
        legend_y += 16
    draw.text((plot_left, plot_bottom + 16), f"epoch {min(epochs)}", fill="black", font=font)
    draw.text((plot_right - 58, plot_bottom + 16), f"epoch {max(epochs)}", fill="black", font=font)
    draw.text((plot_left, plot_top - 18), f"{y_max:.4g}", fill="black", font=font)
    draw.text((plot_left, plot_bottom - 12), f"{y_min:.4g}", fill="black", font=font)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_is_better(current: float, best: float, mode: str) -> bool:
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"Unknown metric mode: {mode}")


def composite_lattice_score(phase_accuracy: float, lattice_mae_mean: float, min_phase_accuracy: float) -> float:
    if phase_accuracy < min_phase_accuracy:
        return -1.0 - (min_phase_accuracy - phase_accuracy)
    return -lattice_mae_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-task 1D-CNN for single-phase XRD analysis.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--fine-tune-mode", choices=["none", "lattice_head", "lattice_late"], default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--strong-lattice-head", action="store_true")
    parser.add_argument("--lattice-mode", choices=["absolute", "phase_residual"], default="absolute")
    parser.add_argument("--residual-scale", type=float, default=0.02)
    parser.add_argument("--residual-mean-source", choices=["pred", "true"], default="pred")
    parser.add_argument("--phase-embedding-dim", type=int, default=64)
    parser.add_argument("--allow-phase-grad-to-lattice", action="store_true")
    parser.add_argument("--lambda-sys", type=float, default=0.30)
    parser.add_argument("--lambda-lat", type=float, default=0.01)
    parser.add_argument("--lambda-phys", type=float, default=0.0)
    parser.add_argument(
        "--primary-metric",
        choices=["auto", "phase_accuracy", "lattice_mae_mean", "composite"],
        default="auto",
        help=(
            "Metric used for best_multitask_cnn.pt. "
            "Regardless of this choice, phase/lattice/composite checkpoints are also saved."
        ),
    )
    parser.add_argument(
        "--composite-min-phase-acc",
        type=float,
        default=0.995,
        help="Minimum validation phase accuracy required before composite selection prioritizes lattice MAE.",
    )
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
                "Refusing to overwrite an existing run directory. "
                "Choose a new --run-name or pass --overwrite-run.\n"
                + "\n".join(str(path) for path in occupied),
                file=sys.stderr,
            )
            return 1
    ensure_dirs(args.model_dir, args.results_dir, args.logs_dir)
    set_seed(args.seed)

    try:
        device = resolve_device(args.device)
        train_dataset = XRDMultiTaskDataset(args.data_dir / "train.npz", args.max_train_samples, args.seed)
        val_dataset = XRDMultiTaskDataset(args.data_dir / "val.npz", args.max_val_samples, args.seed + 1)
    except Exception as exc:
        print(f"Setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if train_dataset.labels != val_dataset.labels:
        print("Setup failed: train and val phase labels do not match.", file=sys.stderr)
        return 1
    if train_dataset.crystal_system_labels != val_dataset.crystal_system_labels:
        print("Setup failed: train and val crystal-system labels do not match.", file=sys.stderr)
        return 1

    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if train_dataset.labels != [str(label) for label in resume_checkpoint.get("phase_labels", [])]:
            print("Setup failed: checkpoint phase labels do not match the training dataset.", file=sys.stderr)
            return 1
        if train_dataset.crystal_system_labels != [str(label) for label in resume_checkpoint.get("crystal_system_labels", [])]:
            print("Setup failed: checkpoint crystal-system labels do not match the training dataset.", file=sys.stderr)
            return 1
        lattice_mean = resume_checkpoint["lattice_mean"].to(device)
        lattice_std = resume_checkpoint["lattice_std"].to(device)
    else:
        lattice_mean = train_dataset.lattice_params[:, :LATTICE_DIM].mean(dim=0).to(device)
        lattice_std = train_dataset.lattice_params[:, :LATTICE_DIM].std(dim=0).clamp_min(1e-6).to(device)
    phase_mean_abc = None
    if args.lattice_mode == "phase_residual":
        phase_mean_abc = compute_phase_mean_abc(train_dataset, len(train_dataset.labels)).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if resume_checkpoint is not None:
        model_config = dict(resume_checkpoint["model_config"])
        model_config.setdefault("lattice_hidden_dims", [256])
        model_config.setdefault("phase_conditioned_lattice", False)
        model_config.setdefault("phase_embedding_dim", args.phase_embedding_dim)
        model_config.setdefault("detach_phase_for_lattice", not args.allow_phase_grad_to_lattice)
        if args.lattice_mode == "phase_residual" and not model_config["phase_conditioned_lattice"]:
            print(
                "Setup failed: phase_residual mode requires a checkpoint trained with "
                "phase_conditioned_lattice=True. Train this mode from scratch first.",
                file=sys.stderr,
            )
            return 1
        model = MultiTaskXRDConvNet(**model_config).to(device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    else:
        lattice_hidden_dims = [256, 128] if args.strong_lattice_head else [256]
        model_config = {
            "num_phase_classes": len(train_dataset.labels),
            "num_crystal_classes": len(train_dataset.crystal_system_labels),
            "lattice_dim": LATTICE_DIM,
            "pooled_bins": 32,
            "dropout": args.dropout,
            "lattice_hidden_dims": lattice_hidden_dims,
            "phase_conditioned_lattice": args.lattice_mode == "phase_residual",
            "phase_embedding_dim": args.phase_embedding_dim,
            "detach_phase_for_lattice": not args.allow_phase_grad_to_lattice,
        }
        model = MultiTaskXRDConvNet(**model_config).to(device)

    trainable_module_names = configure_fine_tuning(model, args.fine_tune_mode)
    parameters = trainable_parameters(model)
    if not parameters:
        print("Setup failed: no trainable parameters selected.", file=sys.stderr)
        return 1

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_mode = "max" if args.fine_tune_mode == "none" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=4)
    phase_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    crystal_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    lattice_criterion = nn.MSELoss()

    run_config = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "fine_tune_mode": args.fine_tune_mode,
        "trainable_modules": trainable_module_names,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "num_phase_classes": len(train_dataset.labels),
        "num_crystal_classes": len(train_dataset.crystal_system_labels),
        "lattice_mean": lattice_mean.detach().cpu().tolist(),
        "lattice_std": lattice_std.detach().cpu().tolist(),
        "phase_mean_abc": phase_mean_abc.detach().cpu().tolist() if phase_mean_abc is not None else None,
        "model": {
            "backbone_channels": [32, 64, 128, 128],
            "kernel_sizes": [11, 9, 7, 5],
            "pooled_bins": model.pooled_bins,
            "feature_dim_after_pool": 128 * model.pooled_bins,
            "shared_projection_dim": 512,
            "lattice_dim": LATTICE_DIM,
            "lattice_hidden_dims": model.lattice_hidden_dims,
            "lattice_mode": args.lattice_mode,
            "residual_scale": args.residual_scale,
            "residual_mean_source": args.residual_mean_source,
            "phase_conditioned_lattice": model.phase_conditioned_lattice,
            "phase_embedding_dim": model.phase_embedding_dim,
            "detach_phase_for_lattice": model.detach_phase_for_lattice,
        },
    }
    write_json(args.results_dir / "run_config.json", run_config)

    if args.primary_metric == "auto":
        primary_metric_name = "phase_accuracy" if args.fine_tune_mode == "none" else "lattice_mae_mean"
    else:
        primary_metric_name = args.primary_metric

    checkpoint_specs = {
        "primary": {
            "path": args.model_dir / "best_multitask_cnn.pt",
            "metric_name": primary_metric_name,
            "mode": "min" if primary_metric_name == "lattice_mae_mean" else "max",
            "best_value": float("inf") if primary_metric_name == "lattice_mae_mean" else -float("inf"),
            "best_epoch": 0,
        },
        "phase": {
            "path": args.model_dir / "best_phase_model.pt",
            "metric_name": "phase_accuracy",
            "mode": "max",
            "best_value": -float("inf"),
            "best_epoch": 0,
        },
        "lattice": {
            "path": args.model_dir / "best_lattice_model.pt",
            "metric_name": "lattice_mae_mean",
            "mode": "min",
            "best_value": float("inf"),
            "best_epoch": 0,
        },
        "composite": {
            "path": args.model_dir / "best_composite_model.pt",
            "metric_name": "composite_lattice_score",
            "mode": "max",
            "best_value": -float("inf"),
            "best_epoch": 0,
        },
    }
    best_path = args.model_dir / "best_multitask_cnn.pt"
    history: list[dict[str, Any]] = []
    log_path = args.logs_dir / "training_log.jsonl"

    def build_checkpoint_payload(epoch: int, metric_name: str, metric_mode: str, metric_value: float, val_metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_phase_classes": len(train_dataset.labels),
                "num_crystal_classes": len(train_dataset.crystal_system_labels),
                "lattice_dim": LATTICE_DIM,
                "pooled_bins": model.pooled_bins,
                "dropout": model_config.get("dropout", args.dropout),
                "lattice_hidden_dims": model.lattice_hidden_dims,
                "phase_conditioned_lattice": model.phase_conditioned_lattice,
                "phase_embedding_dim": model.phase_embedding_dim,
                "detach_phase_for_lattice": model.detach_phase_for_lattice,
            },
            "phase_labels": train_dataset.labels,
            "crystal_system_labels": train_dataset.crystal_system_labels,
            "lattice_mean": lattice_mean.detach().cpu(),
            "lattice_std": lattice_std.detach().cpu(),
            "phase_mean_abc": phase_mean_abc.detach().cpu() if phase_mean_abc is not None else None,
            "lattice_mode": args.lattice_mode,
            "residual_scale": args.residual_scale,
            "residual_mean_source": args.residual_mean_source,
            "best_epoch": epoch,
            "best_val_phase_accuracy": val_metrics["phase_accuracy"],
            "best_val_crystal_accuracy": val_metrics["crystal_accuracy"],
            "best_val_lattice_mae_mean": val_metrics["lattice_mae_mean"],
            "best_metric_name": metric_name,
            "best_metric_mode": metric_mode,
            "best_metric_value": float(metric_value),
            "composite_min_phase_accuracy": args.composite_min_phase_acc,
            "run_config": run_config,
        }

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Run name: {run_name}")
    print(f"Fine-tune mode: {args.fine_tune_mode} | trainable modules: {', '.join(trainable_module_names)}")
    print(f"Saving primary checkpoint by val {primary_metric_name}: {best_path}")
    print("Also saving best_phase_model.pt, best_lattice_model.pt, and best_composite_model.pt")

    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                phase_criterion,
                crystal_criterion,
                lattice_criterion,
                lattice_mean,
                lattice_std,
                args.lambda_sys,
                args.lambda_lat,
                args.lambda_phys,
                args.lattice_mode,
                phase_mean_abc,
                args.residual_scale,
                args.residual_mean_source,
                device,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                phase_criterion,
                crystal_criterion,
                lattice_criterion,
                lattice_mean,
                lattice_std,
                args.lambda_sys,
                args.lambda_lat,
                args.lambda_phys,
                args.lattice_mode,
                phase_mean_abc,
                args.residual_scale,
                args.residual_mean_source,
                device,
            )
            scheduler.step(val_metrics["phase_accuracy"] if args.fine_tune_mode == "none" else val_metrics["lattice_mae_mean"])
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_handle.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_phase_acc={train_metrics['phase_accuracy']:.4f} "
                f"train_crystal_acc={train_metrics['crystal_accuracy']:.4f} "
                f"train_lat_mae={train_metrics['lattice_mae_mean']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_phase_acc={val_metrics['phase_accuracy']:.4f} "
                f"val_crystal_acc={val_metrics['crystal_accuracy']:.4f} "
                f"val_lat_mae={val_metrics['lattice_mae_mean']:.4f}"
            )

            current_values = {
                "phase_accuracy": val_metrics["phase_accuracy"],
                "lattice_mae_mean": val_metrics["lattice_mae_mean"],
                "composite_lattice_score": composite_lattice_score(
                    val_metrics["phase_accuracy"],
                    val_metrics["lattice_mae_mean"],
                    args.composite_min_phase_acc,
                ),
            }
            for spec in checkpoint_specs.values():
                metric_name = str(spec["metric_name"])
                current_metric = current_values[metric_name]
                best_value = float(spec["best_value"])
                mode = str(spec["mode"])
                if metric_is_better(current_metric, best_value, mode):
                    spec["best_value"] = current_metric
                    spec["best_epoch"] = epoch
                    torch.save(
                        build_checkpoint_payload(epoch, metric_name, mode, current_metric, val_metrics),
                        spec["path"],
                    )

    write_json(args.results_dir / "training_history.json", history)
    save_training_curves(history, args.results_dir / "training_curves.png")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    lattice_mean = checkpoint["lattice_mean"].to(device)
    lattice_std = checkpoint["lattice_std"].to(device)
    if checkpoint.get("phase_mean_abc") is not None:
        phase_mean_abc = checkpoint["phase_mean_abc"].to(device)

    split_metrics: dict[str, Any] = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_phase_accuracy": float(checkpoint["best_val_phase_accuracy"]),
        "best_val_crystal_accuracy": float(checkpoint.get("best_val_crystal_accuracy", float("nan"))),
        "best_val_lattice_mae_mean": float(checkpoint.get("best_val_lattice_mae_mean", float("nan"))),
        "best_metric_name": str(checkpoint.get("best_metric_name", "phase_accuracy")),
        "best_metric_value": float(checkpoint.get("best_metric_value", checkpoint["best_val_phase_accuracy"])),
        "checkpoint_selection": {
            name: {
                "path": str(spec["path"]),
                "metric_name": str(spec["metric_name"]),
                "metric_mode": str(spec["mode"]),
                "best_epoch": int(spec["best_epoch"]),
                "best_metric_value": float(spec["best_value"]),
            }
            for name, spec in checkpoint_specs.items()
        },
        "val": evaluate_split(
            model,
            "val",
            val_dataset,
            args.batch_size,
            args.num_workers,
            lattice_mean,
            lattice_std,
            phase_mean_abc,
            args.lattice_mode,
            args.residual_scale,
            args.residual_mean_source,
            device,
            args.results_dir,
        ),
    }

    for split_name in ["test_normal", "test_hard"]:
        dataset = XRDMultiTaskDataset(args.data_dir / f"{split_name}.npz", args.max_test_samples, args.seed + 2)
        split_metrics[split_name] = evaluate_split(
            model,
            split_name,
            dataset,
            args.batch_size,
            args.num_workers,
            lattice_mean,
            lattice_std,
            phase_mean_abc,
            args.lattice_mode,
            args.residual_scale,
            args.residual_mean_source,
            device,
            args.results_dir,
        )

    write_json(args.results_dir / "metrics_summary.json", split_metrics)
    primary_spec = checkpoint_specs["primary"]
    print(
        f"Best primary epoch: {int(primary_spec['best_epoch'])} | "
        f"best val {primary_spec['metric_name']}: {float(primary_spec['best_value']):.4f}"
    )
    print(
        "Saved additional checkpoints: "
        f"phase={checkpoint_specs['phase']['path']}, "
        f"lattice={checkpoint_specs['lattice']['path']}, "
        f"composite={checkpoint_specs['composite']['path']}"
    )
    print(f"Saved metrics summary: {args.results_dir / 'metrics_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
