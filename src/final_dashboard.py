from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_mixture import MixedPhaseXRDConvNet
from model_multitask import MultiTaskXRDConvNet
from physics import masked_peak_mae


SINGLE_CHECKPOINT = PROJECT_ROOT / "models" / "residual_strong_lat003_ls005_v2_best_lattice" / "best_multitask_cnn.pt"
MIXED_CHECKPOINT = PROJECT_ROOT / "models" / "mixed_phase" / "mixed_phase_cnn_v3_hard_aug_presence_50ep" / "best_mixed_phase_cnn.pt"

SINGLE_METRICS = PROJECT_ROOT / "results" / "residual_strong_lat003_ls005_v2_best_lattice" / "metrics_summary.json"
MIXED_METRICS = PROJECT_ROOT / "results" / "mixed_phase" / "mixed_phase_cnn_v3_hard_aug_presence_50ep" / "metrics_summary.json"
MIXED_HARD_METRICS = PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "v3_hard_aug_presence_on_mixed_v2_hard_eval" / "metrics_summary.json"
ACTIVE_SUMMARY = PROJECT_ROOT / "results" / "active_learning" / "battery_round2" / "active_learning_summary.json"

ACTIVE_HARD_EVALS = {
    "active_round2": PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "mixed_v5_battery_active_round2_on_mixed_v2_hard_eval" / "metrics_summary.json",
    "random_seed123": PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "mixed_v5_battery_random_round2_seed123_on_mixed_v2_hard_eval" / "metrics_summary.json",
    "random_seed456": PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "mixed_v5_battery_random_round2_seed456_on_mixed_v2_hard_eval" / "metrics_summary.json",
    "random_seed789": PROJECT_ROOT / "results" / "mixed_phase" / "hard_eval" / "mixed_v5_battery_random_round2_seed789_on_mixed_v2_hard_eval" / "metrics_summary.json",
}

SINGLE_SPLITS = {
    "single/val": PROJECT_ROOT / "data" / "processed" / "single_phase" / "val.npz",
    "single/test_normal": PROJECT_ROOT / "data" / "processed" / "single_phase" / "test_normal.npz",
    "single/test_hard": PROJECT_ROOT / "data" / "processed" / "single_phase" / "test_hard.npz",
}

MIXED_SPLITS = {
    "mixed_v3/test_normal": PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v3_hard_augmented" / "test_normal.npz",
    "mixed_v3/test_hard": PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v3_hard_augmented" / "test_hard.npz",
    "hard_eval/test_minor": PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v2_hard_eval" / "test_minor.npz",
    "hard_eval/test_overlap": PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v2_hard_eval" / "test_overlap.npz",
    "hard_eval/test_battery_relevant": PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v2_hard_eval" / "test_battery_relevant.npz",
}

LATTICE_NAMES = ["a", "b", "c"]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(data: dict, dotted_path: str, default: float | str | None = None):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def compact_metrics() -> dict:
    single = read_json(SINGLE_METRICS)
    mixed = read_json(MIXED_METRICS)
    mixed_hard = read_json(MIXED_HARD_METRICS)
    active = read_json(ACTIVE_SUMMARY)

    active_rows = []
    baseline_metrics = metric_value(mixed_hard, "splits.test_battery_relevant.fixed.metrics", {})
    if baseline_metrics:
        active_rows.append(
            {
                "name": "v3_baseline",
                "battery_micro_precision": baseline_metrics.get("micro_precision"),
                "battery_micro_recall": baseline_metrics.get("micro_recall"),
                "battery_micro_f1": baseline_metrics.get("micro_f1"),
                "battery_minor_recall": baseline_metrics.get("minor_phase_recall"),
                "battery_top3_all_hit": baseline_metrics.get("top3_all_phases_hit"),
                "battery_fp_per_sample": baseline_metrics.get("avg_false_positives_per_sample"),
                "role": "baseline",
            }
        )
    for name, path in ACTIVE_HARD_EVALS.items():
        data = read_json(path)
        if not data:
            continue
        row = {"name": name}
        fixed = metric_value(data, "splits.test_battery_relevant.fixed.metrics", {})
        row["battery_micro_precision"] = fixed.get("micro_precision")
        row["battery_micro_recall"] = fixed.get("micro_recall")
        row["battery_micro_f1"] = fixed.get("micro_f1")
        row["battery_minor_recall"] = fixed.get("minor_phase_recall")
        row["battery_top3_all_hit"] = fixed.get("top3_all_phases_hit")
        row["battery_fp_per_sample"] = fixed.get("avg_false_positives_per_sample")
        row["role"] = "active" if name == "active_round2" else "random"
        active_rows.append(row)

    active_row = next((row for row in active_rows if row.get("name") == "active_round2"), {})
    random_rows = [row for row in active_rows if row.get("role") == "random"]

    def average_random(key: str):
        values = [row.get(key) for row in random_rows if row.get(key) is not None]
        if not values:
            return None
        return float(sum(float(value) for value in values) / len(values))

    active_gain = {
        "baseline_f1": baseline_metrics.get("micro_f1"),
        "active_f1": active_row.get("battery_micro_f1"),
        "delta_f1": None,
        "baseline_minor_recall": baseline_metrics.get("minor_phase_recall"),
        "active_minor_recall": active_row.get("battery_minor_recall"),
        "delta_minor_recall": None,
        "baseline_top3_all_hit": baseline_metrics.get("top3_all_phases_hit"),
        "active_top3_all_hit": active_row.get("battery_top3_all_hit"),
        "delta_top3_all_hit": None,
        "baseline_fp_per_sample": baseline_metrics.get("avg_false_positives_per_sample"),
        "active_fp_per_sample": active_row.get("battery_fp_per_sample"),
        "delta_fp_per_sample": None,
        "random_avg_f1": average_random("battery_micro_f1"),
        "random_avg_minor_recall": average_random("battery_minor_recall"),
    }
    if active_gain["baseline_f1"] is not None and active_gain["active_f1"] is not None:
        active_gain["delta_f1"] = float(active_gain["active_f1"]) - float(active_gain["baseline_f1"])
    if active_gain["baseline_minor_recall"] is not None and active_gain["active_minor_recall"] is not None:
        active_gain["delta_minor_recall"] = float(active_gain["active_minor_recall"]) - float(active_gain["baseline_minor_recall"])
    if active_gain["baseline_top3_all_hit"] is not None and active_gain["active_top3_all_hit"] is not None:
        active_gain["delta_top3_all_hit"] = float(active_gain["active_top3_all_hit"]) - float(active_gain["baseline_top3_all_hit"])
    if active_gain["baseline_fp_per_sample"] is not None and active_gain["active_fp_per_sample"] is not None:
        active_gain["delta_fp_per_sample"] = float(active_gain["active_fp_per_sample"]) - float(active_gain["baseline_fp_per_sample"])

    return {
        "cards": [
            {
                "label": "single-phase accuracy",
                "value": pct(metric_value(single, "test_normal.phase_accuracy")),
                "detail": "test_normal",
            },
            {
                "label": "single-phase lattice MAE",
                "value": f"{num(metric_value(single, 'test_normal.lattice_mae_mean'), 3)} A",
                "detail": "test_normal mean",
            },
            {
                "label": "mixed-phase micro-F1",
                "value": pct(metric_value(mixed, "test_normal.micro_f1")),
                "detail": "mixed_v3/test_normal",
            },
            {
                "label": "mixed-phase minor recall",
                "value": pct(metric_value(mixed, "test_hard.minor_phase_recall")),
                "detail": "mixed_v3/test_hard",
            },
            {
                "label": "Battery hard micro-F1",
                "value": pct(metric_value(mixed_hard, "splits.test_battery_relevant.fixed.metrics.micro_f1")),
                "detail": "v3 on mixed_v2 hard eval",
            },
            {
                "label": "Active selected",
                "value": str(active.get("selected_count", "-")),
                "detail": f"pool {active.get('pool_sample_count', '-')}",
            },
        ],
        "single": {
            split: {
                "phase_accuracy": metric_value(single, f"{split}.phase_accuracy"),
                "phase_top3_accuracy": metric_value(single, f"{split}.phase_top3_accuracy"),
                "crystal_accuracy": metric_value(single, f"{split}.crystal_accuracy"),
                "lattice_mae_mean": metric_value(single, f"{split}.lattice_mae_mean"),
                "bragg_peak_mae_deg": metric_value(single, f"{split}.bragg_peak_mae_deg"),
            }
            for split in ["val", "test_normal", "test_hard"]
        },
        "mixed": {
            split: {
                "micro_precision": metric_value(mixed, f"{split}.micro_precision"),
                "micro_recall": metric_value(mixed, f"{split}.micro_recall"),
                "micro_f1": metric_value(mixed, f"{split}.micro_f1"),
                "minor_phase_recall": metric_value(mixed, f"{split}.minor_phase_recall"),
                "top3_all_phases_hit": metric_value(mixed, f"{split}.top3_all_phases_hit"),
                "avg_false_positives_per_sample": metric_value(mixed, f"{split}.avg_false_positives_per_sample"),
            }
            for split in ["val", "test_normal", "test_hard"]
        },
        "mixed_hard": {
            split: {
                "micro_f1": metric_value(mixed_hard, f"splits.{split}.fixed.metrics.micro_f1"),
                "minor_phase_recall": metric_value(mixed_hard, f"splits.{split}.fixed.metrics.minor_phase_recall"),
                "top3_all_phases_hit": metric_value(mixed_hard, f"splits.{split}.fixed.metrics.top3_all_phases_hit"),
                "avg_false_positives_per_sample": metric_value(mixed_hard, f"splits.{split}.fixed.metrics.avg_false_positives_per_sample"),
            }
            for split in ["test_minor", "test_overlap", "test_battery_relevant"]
        },
        "active": {
            "summary": active,
            "hard_eval_rows": active_rows,
            "gain": active_gain,
        },
    }


def apply_threshold(probabilities: np.ndarray, threshold: float, min_predictions: int, max_predictions: int) -> np.ndarray:
    pred = probabilities >= threshold
    if min_predictions > 0 and pred.sum() < min_predictions:
        top = np.argsort(-probabilities)[:min_predictions]
        pred[top] = True
    if max_predictions > 0 and pred.sum() > max_predictions:
        keep = np.argsort(-probabilities)[:max_predictions]
        pred[:] = False
        pred[keep] = True
    return pred


def safe_label_list(values: np.ndarray, count: int) -> list[str]:
    labels = []
    for value in values[:count]:
        label = str(value)
        if label:
            labels.append(label)
    return labels


class DashboardState:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.single_model, self.single_checkpoint = self.load_single_model()
        self.mixed_model, self.mixed_checkpoint = self.load_mixed_model()
        self.single_datasets = self.load_single_datasets()
        self.mixed_datasets = self.load_mixed_datasets()
        self.metrics = compact_metrics()

    def load_single_model(self):
        if not SINGLE_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing single-phase checkpoint: {SINGLE_CHECKPOINT}")
        checkpoint = torch.load(SINGLE_CHECKPOINT, map_location=self.device, weights_only=False)
        model = MultiTaskXRDConvNet(**checkpoint["model_config"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    def load_mixed_model(self):
        if not MIXED_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing mixed-phase checkpoint: {MIXED_CHECKPOINT}")
        checkpoint = torch.load(MIXED_CHECKPOINT, map_location=self.device, weights_only=False)
        model = MixedPhaseXRDConvNet(**checkpoint["model_config"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    def load_single_datasets(self) -> dict[str, dict]:
        datasets = {}
        required = ["X", "y", "crystal_system_y", "labels", "crystal_system_labels", "two_theta", "lattice_params", "peak_hkls", "peak_2theta", "peak_mask"]
        for name, path in SINGLE_SPLITS.items():
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            datasets[name] = {
                "X": data["X"].astype(np.float32),
                "y": data["y"].astype(np.int64),
                "crystal_y": data["crystal_system_y"].astype(np.int64),
                "labels": [str(v) for v in data["labels"]],
                "crystal_labels": [str(v) for v in data["crystal_system_labels"]],
                "two_theta": data["two_theta"].astype(np.float32),
                "lattice_params": data["lattice_params"].astype(np.float32),
                "peak_hkls": data["peak_hkls"].astype(np.float32),
                "peak_2theta": data["peak_2theta"].astype(np.float32),
                "peak_mask": data["peak_mask"].astype(np.float32),
                "sample_categories": data["sample_categories"].astype(str) if "sample_categories" in data else None,
                "sample_subclasses": data["sample_subclasses"].astype(str) if "sample_subclasses" in data else None,
                "sample_material_ids": data["sample_material_ids"].astype(str) if "sample_material_ids" in data else None,
            }
        if not datasets:
            raise FileNotFoundError("No single-phase datasets were found.")
        return datasets

    def load_mixed_datasets(self) -> dict[str, dict]:
        datasets = {}
        required = ["X", "y_multi", "labels", "two_theta", "component_phase_labels", "component_fractions", "n_phases", "major_phase_index"]
        for name, path in MIXED_SPLITS.items():
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            datasets[name] = {
                "X": data["X"].astype(np.float32),
                "y_multi": data["y_multi"].astype(np.float32),
                "labels": [str(v) for v in data["labels"]],
                "two_theta": data["two_theta"].astype(np.float32),
                "component_phase_labels": data["component_phase_labels"].astype(str),
                "component_fractions": data["component_fractions"].astype(np.float32),
                "n_phases": data["n_phases"].astype(np.int64),
                "major_phase_index": data["major_phase_index"].astype(np.int64),
                "minor_fraction_bin": data["minor_fraction_bin"].astype(str) if "minor_fraction_bin" in data else None,
                "hard_eval_type": data["hard_eval_type"].astype(str) if "hard_eval_type" in data else None,
                "battery_scenario": data["battery_scenario"].astype(str) if "battery_scenario" in data else None,
                "overlap_bin": data["overlap_bin"].astype(str) if "overlap_bin" in data else None,
            }
        if not datasets:
            raise FileNotFoundError("No mixed-phase datasets were found.")
        return datasets

    def overview(self) -> dict:
        return {
            "metrics": self.metrics,
            "single_splits": [{"name": name, "count": int(data["X"].shape[0])} for name, data in self.single_datasets.items()],
            "mixed_splits": [{"name": name, "count": int(data["X"].shape[0])} for name, data in self.mixed_datasets.items()],
            "mixed_default_threshold": float(self.mixed_checkpoint.get("best_threshold", 0.5)),
        }

    def mixed_split_metrics(self, split: str) -> dict:
        if split.startswith("mixed_v3/"):
            key = split.split("/", 1)[1]
            return self.metrics.get("mixed", {}).get(key, {})
        if split.startswith("hard_eval/"):
            key = split.split("/", 1)[1]
            return self.metrics.get("mixed_hard", {}).get(key, {})
        return {}

    def checked_single(self, split: str, index: int) -> tuple[dict, int]:
        if split not in self.single_datasets:
            raise ValueError(f"Unknown single split: {split}")
        data = self.single_datasets[split]
        index = int(index)
        if index < 0 or index >= data["X"].shape[0]:
            raise ValueError(f"Index out of range: {index}")
        return data, index

    def checked_mixed(self, split: str, index: int) -> tuple[dict, int]:
        if split not in self.mixed_datasets:
            raise ValueError(f"Unknown mixed split: {split}")
        data = self.mixed_datasets[split]
        index = int(index)
        if index < 0 or index >= data["X"].shape[0]:
            raise ValueError(f"Index out of range: {index}")
        return data, index

    def single_sample(self, split: str, index: int) -> dict:
        data, index = self.checked_single(split, index)
        y = int(data["y"][index])
        crystal = int(data["crystal_y"][index])
        return {
            "task": "single",
            "split": split,
            "index": index,
            "count": int(data["X"].shape[0]),
            "two_theta": data["two_theta"].tolist(),
            "X": data["X"][index].tolist(),
            "true_label": data["labels"][y],
            "true_crystal": data["crystal_labels"][crystal],
            "category": str(data["sample_categories"][index]) if data["sample_categories"] is not None else "",
            "subclass": str(data["sample_subclasses"][index]) if data["sample_subclasses"] is not None else "",
            "material_id": str(data["sample_material_ids"][index]) if data["sample_material_ids"] is not None else "",
        }

    def mixed_sample(self, split: str, index: int) -> dict:
        data, index = self.checked_mixed(split, index)
        n_phases = int(data["n_phases"][index])
        phases = safe_label_list(data["component_phase_labels"][index], n_phases)
        fractions = [float(v) for v in data["component_fractions"][index][:n_phases]]
        extra = {}
        for key in ["minor_fraction_bin", "hard_eval_type", "battery_scenario", "overlap_bin"]:
            values = data.get(key)
            if values is not None:
                extra[key] = str(values[index])
        return {
            "task": "mixed",
            "split": split,
            "index": index,
            "count": int(data["X"].shape[0]),
            "two_theta": data["two_theta"].tolist(),
            "X": data["X"][index].tolist(),
            "true_phases": [{"label": label, "fraction": fraction} for label, fraction in zip(phases, fractions)],
            "extra": extra,
        }

    @torch.no_grad()
    def predict_single(self, split: str, index: int) -> dict:
        data, index = self.checked_single(split, index)
        x = torch.from_numpy(data["X"][index]).view(1, 1, -1).to(self.device)
        y_true = int(data["y"][index])
        crystal_true = int(data["crystal_y"][index])

        start = time.perf_counter()
        outputs = self.single_model(x)
        phase_probs_t = torch.softmax(outputs["phase_logits"], dim=1)
        crystal_probs_t = torch.softmax(outputs["crystal_logits"], dim=1)
        lattice_pred = self.single_lattice_prediction(outputs, torch.tensor([y_true], device=self.device), phase_probs_t)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        phase_probs = phase_probs_t.cpu().numpy()[0]
        crystal_probs = crystal_probs_t.cpu().numpy()[0]
        lattice_pred_np = lattice_pred.cpu().numpy()[0]
        lattice_true = data["lattice_params"][index]
        lattice_pred_full = lattice_true.copy()
        lattice_pred_full[:3] = lattice_pred_np
        peak_error = masked_peak_mae(
            torch.from_numpy(lattice_pred_full.reshape(1, 6).astype(np.float32)),
            torch.from_numpy(data["peak_hkls"][index : index + 1]),
            torch.from_numpy(data["peak_2theta"][index : index + 1]),
            torch.from_numpy(data["peak_mask"][index : index + 1]),
        ).item()

        phase_pred = int(np.argmax(phase_probs))
        crystal_pred = int(np.argmax(crystal_probs))
        top_phase = np.argsort(-phase_probs)[:5]
        top_crystal = np.argsort(-crystal_probs)[:3]
        return {
            **self.single_sample(split, index),
            "pred_label": data["labels"][phase_pred],
            "pred_crystal": data["crystal_labels"][crystal_pred],
            "phase_correct": bool(phase_pred == y_true),
            "crystal_correct": bool(crystal_pred == crystal_true),
            "phase_confidence": float(phase_probs[phase_pred]),
            "crystal_confidence": float(crystal_probs[crystal_pred]),
            "inference_time_ms": float(elapsed_ms),
            "lattice_true": [float(v) for v in lattice_true[:3]],
            "lattice_pred": [float(v) for v in lattice_pred_np],
            "lattice_abs_error": [float(v) for v in np.abs(lattice_pred_np - lattice_true[:3])],
            "peak_error_deg": float(peak_error),
            "top5": [{"label": data["labels"][int(i)], "probability": float(phase_probs[int(i)])} for i in top_phase],
            "top_crystal": [{"label": data["crystal_labels"][int(i)], "probability": float(crystal_probs[int(i)])} for i in top_crystal],
        }

    def single_lattice_prediction(self, outputs: dict[str, torch.Tensor], phase_y: torch.Tensor, phase_probs: torch.Tensor) -> torch.Tensor:
        mode = str(self.single_checkpoint.get("lattice_mode", "absolute"))
        if mode == "absolute":
            mean = self.single_checkpoint["lattice_mean"].to(self.device)
            std = self.single_checkpoint["lattice_std"].to(self.device)
            return outputs["lattice_norm"] * std + mean
        if mode == "phase_residual":
            phase_mean = self.single_checkpoint["phase_mean_abc"].to(self.device)
            source = str(self.single_checkpoint.get("residual_mean_source", "pred"))
            if source == "true":
                base = phase_mean[phase_y]
            else:
                base = phase_probs.detach() @ phase_mean
            scale = float(self.single_checkpoint.get("residual_scale", 0.02))
            return base * (1.0 + scale * torch.tanh(outputs["lattice_raw"]))
        raise ValueError(f"Unsupported lattice mode: {mode}")

    @torch.no_grad()
    def predict_mixed(self, split: str, index: int, threshold: float, max_predictions: int) -> dict:
        data, index = self.checked_mixed(split, index)
        x = torch.from_numpy(data["X"][index]).view(1, 1, -1).to(self.device)
        threshold = float(threshold)
        max_predictions = int(max_predictions)

        start = time.perf_counter()
        outputs = self.mixed_model(x)
        probabilities = torch.sigmoid(outputs["phase_logits"]).cpu().numpy()[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pred_mask = apply_threshold(probabilities, threshold, min_predictions=1, max_predictions=max_predictions)
        pred_indices = np.where(pred_mask)[0]
        true_indices = np.where(data["y_multi"][index] >= 0.5)[0]
        true_set = {int(i) for i in true_indices}
        pred_set = {int(i) for i in pred_indices}
        missed = sorted(true_set - pred_set)
        false_positive = sorted(pred_set - true_set)
        major_idx = int(data["major_phase_index"][index])
        minor_indices = [int(i) for i in true_indices if int(i) != major_idx]
        minor_hit = bool(minor_indices and all(idx in pred_set for idx in minor_indices))
        top_indices = np.argsort(-probabilities)[:8]
        hit_count = len(true_set & pred_set)
        true_count = max(1, len(true_set))

        return {
            **self.mixed_sample(split, index),
            "threshold": threshold,
            "max_predictions": max_predictions,
            "inference_time_ms": float(elapsed_ms),
            "overall_metrics": self.mixed_split_metrics(split),
            "true_phase_hit_count": int(hit_count),
            "true_phase_count": int(len(true_set)),
            "true_phase_coverage": float(hit_count / true_count),
            "false_positive_count": int(len(false_positive)),
            "predicted_phases": [
                {"label": data["labels"][int(i)], "probability": float(probabilities[int(i)]), "is_true": bool(int(i) in true_set)}
                for i in pred_indices
            ],
            "missed_phases": [{"label": data["labels"][int(i)], "probability": float(probabilities[int(i)])} for i in missed],
            "false_positives": [{"label": data["labels"][int(i)], "probability": float(probabilities[int(i)])} for i in false_positive],
            "major_hit": bool(major_idx in pred_set),
            "minor_hit": minor_hit,
            "all_hit": bool(true_set.issubset(pred_set)),
            "top8": [{"label": data["labels"][int(i)], "probability": float(probabilities[int(i)]), "is_true": bool(int(i) in true_set)} for i in top_indices],
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>XRD Autonomous Characterization Dashboard</title>
  <style>
    :root {
      --page: #f5f7fa; --panel: #ffffff; --ink: #18242e; --muted: #667482;
      --line: #d9e2ea; --blue: #245a8d; --green: #28785a; --red: #b64747;
      --amber: #a66b24; --teal: #13747c; --violet: #6f5b9a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: var(--ink); background: var(--page); }
    header { padding: 18px 26px; color: white; background: #18384f; }
    h1 { margin: 0; font-size: 24px; line-height: 1.25; letter-spacing: 0; }
    header p { margin: 7px 0 0; max-width: 1040px; color: #e6eef4; line-height: 1.5; }
    main { max-width: 1400px; margin: 0 auto; padding: 18px; }
    nav { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    nav button { width: auto; min-width: 120px; height: 38px; padding: 0 14px; color: var(--ink); background: white; border: 1px solid var(--line); border-radius: 6px; }
    nav button.active { color: white; background: var(--blue); border-color: var(--blue); }
    section.view { display: none; }
    section.view.active { display: block; }
    .cards { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin-bottom: 14px; }
    .card, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 7px 18px rgba(30, 45, 60, 0.05); }
    .card { flex: 0 1 210px; padding: 12px; min-height: 88px; border-left: 4px solid var(--blue); }
    .card small { display: block; color: var(--muted); font-weight: 700; margin-bottom: 7px; }
    .card strong { display: block; font-size: 22px; line-height: 1.15; }
    .card span { display: block; margin-top: 6px; color: var(--muted); font-size: 12px; }
    .panel { padding: 14px; margin-bottom: 14px; }
    .panel h2 { margin: 0 0 12px; font-size: 17px; letter-spacing: 0; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .grid-main { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(360px, 1fr); gap: 14px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin-bottom: 12px; }
    .toolbar > div { flex: 0 1 150px; }
    .toolbar > div:nth-child(2) { flex-basis: 300px; }
    .toolbar button { flex: 0 0 118px; width: auto; }
    label { display: block; color: var(--muted); font-size: 13px; font-weight: 700; margin-bottom: 5px; }
    select, input, button { width: 100%; height: 38px; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--ink); font-size: 14px; }
    input { padding: 0 9px; }
    button { border: 0; color: white; background: var(--blue); font-weight: 700; cursor: pointer; }
    button.secondary { background: var(--teal); }
    button:disabled { opacity: .6; cursor: wait; }
    canvas { display: block; width: 100%; height: 360px; border: 1px solid var(--line); border-radius: 6px; background: #fbfdff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 7px; border-bottom: 1px solid #edf2f6; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; font-weight: 700; color: var(--muted); }
    .result-list { list-style: none; padding: 0; margin: 0; }
    .result-list li { display: grid; grid-template-columns: minmax(120px, 1fr) minmax(120px, 1.3fr) 64px; gap: 8px; align-items: center; padding: 7px 0; border-bottom: 1px solid #edf2f6; font-size: 13px; }
    .bar { height: 9px; background: #e7edf2; border-radius: 5px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: var(--amber); }
    .ok { color: var(--green); } .bad { color: var(--red); } .muted { color: var(--muted); }
    .chips { display: flex; gap: 6px; flex-wrap: wrap; }
    .chip { display: inline-block; border: 1px solid var(--line); border-radius: 6px; padding: 5px 7px; background: #f7fafc; font-size: 12px; }
    .chip.true { border-color: #9ed0b5; background: #effaf4; }
    .chip.miss { border-color: #e2b4b4; background: #fff5f5; }
    .note { color: var(--muted); font-size: 13px; line-height: 1.5; margin-top: 8px; }
    @media (max-width: 1100px) {
      .grid2, .grid-main { grid-template-columns: 1fr; }
      .toolbar > div, .toolbar > div:nth-child(2), .toolbar button { flex: 1 1 180px; }
    }
    @media (max-width: 700px) { .card { flex-basis: 100%; } canvas { height: 280px; } }
  </style>
</head>
<body>
  <header>
    <h1>XRD Autonomous Characterization Dashboard</h1>
    <p>Integrated view of single-phase identification, mixed-phase recognition, and the battery-focused active-learning extension. Models are loaded once locally and run single-sample inference on demand.</p>
  </header>
  <main>
    <nav>
      <button class="tab active" data-view="overview">Overview</button>
      <button class="tab" data-view="single">Single Phase</button>
      <button class="tab" data-view="mixed">Mixed Phase</button>
      <button class="tab" data-view="active">Active Learning</button>
      <button class="tab" data-view="simulate">Simulation</button>
    </nav>

    <section id="overview" class="view active">
      <div class="cards" id="metricCards"></div>
      <div class="grid2">
        <div class="panel">
          <h2>High-Throughput Workflow</h2>
          <table>
            <tbody>
              <tr><td>1</td><td>Batch sample preparation and sample ID tracking</td></tr>
              <tr><td>2</td><td>Automated XRD acquisition and pattern storage</td></tr>
              <tr><td>3</td><td>Single-phase or mixed-phase model inference</td></tr>
              <tr><td>4</td><td>Confidence, error, and Bragg-law reliability outputs</td></tr>
              <tr><td>5</td><td>Low-confidence samples routed to expert review or active learning</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>System Scope</h2>
          <p class="note">The closed-loop core is single-phase modeling, mixed-phase modeling, hard benchmarks, and this dashboard. Active learning is included as a battery-focused workflow extension for selecting high-value samples.</p>
          <div class="chips">
            <span class="chip true">Materials Project</span>
            <span class="chip true">single-phase CNN</span>
            <span class="chip true">mixed-phase CNN</span>
            <span class="chip true">Bragg consistency</span>
            <span class="chip">active learning extension</span>
          </div>
        </div>
      </div>
    </section>

    <section id="single" class="view">
      <div class="panel">
        <h2>Metric Notes</h2>
        <table>
          <tbody>
            <tr><td>phase accuracy</td><td>Top-1 phase-label accuracy, the main metric for single-phase identification.</td></tr>
            <tr><td>top-3 accuracy</td><td>Whether the true phase appears in the top three candidates; useful for high-throughput screening.</td></tr>
            <tr><td>crystal accuracy</td><td>Crystal-system classification accuracy, indicating whether the model learned structural information.</td></tr>
            <tr><td>lattice MAE</td><td>Mean absolute error of predicted lattice lengths a/b/c, in A.</td></tr>
            <tr><td>Bragg MAE</td><td>Peak-position error computed from predicted lattice parameters using Bragg-law consistency.</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel"><h2>Final Single-Phase Performance</h2><table id="singleTable"></table></div>
    </section>

    <section id="mixed" class="view">
      <div class="panel">
        <h2>Metric Notes</h2>
        <table>
          <tbody>
            <tr><td>micro precision</td><td>Across all phase labels, how many predicted phases are actually present.</td></tr>
            <tr><td>micro recall</td><td>Across all true phases, how many were recovered by the model.</td></tr>
            <tr><td>micro-F1</td><td>Balanced precision-recall score for multi-label mixed-phase recognition.</td></tr>
            <tr><td>minor phase recall</td><td>Recall for secondary, impurity, or low-fraction phases; the hardest mixed-phase metric.</td></tr>
            <tr><td>top3 all-hit</td><td>Whether all true phases appear within the top three candidates.</td></tr>
            <tr><td>FP/sample</td><td>Average number of extra phases reported per sample; lower is better.</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel"><h2>Final Mixed-Phase Performance</h2><table id="mixedTable"></table></div>
      <div class="panel"><h2>Independent Hard Benchmark</h2><table id="mixedHardTable"></table></div>
    </section>

    <section id="active" class="view">
      <div class="grid2">
        <div class="panel">
          <h2>Battery-Focused Active Round Summary</h2>
          <p class="note">The active-learning module focuses only on battery-relevant mixtures. It selects high-uncertainty, high-value samples from the battery candidate pool; other hard splits are intentionally not compared here.</p>
          <table id="activeSummary"></table>
        </div>
        <div class="panel">
          <h2>Battery-Relevant Active vs Random</h2>
          <p class="note">`v3_baseline` is the pre-active-learning mixed-phase model. `active_round2` adds battery-relevant samples selected by the active strategy. Random seeds are equal-budget random-sampling baselines.</p>
          <h2>Baseline -> Active Improvement</h2>
          <table id="activeGainTable"></table>
          <table id="activeHardTable"></table>
        </div>
      </div>
    </section>

    <section id="simulate" class="view">
      <div class="panel">
        <h2>Recognition Simulation</h2>
        <p class="note">For live demos, start with “Mixed phase / normal test”. Challenge sets intentionally contain low-fraction minor phases, peak-overlap pairs, and battery-relevant mixtures; a miss on one sample is not the split-level accuracy.</p>
        <div class="toolbar">
          <div><label for="taskSelect">Task</label><select id="taskSelect"><option value="single">Single phase</option><option value="mixed">Mixed phase</option></select></div>
          <div><label for="splitSelect">Dataset</label><select id="splitSelect"></select></div>
          <div><label for="sampleIndex">Sample ID</label><input id="sampleIndex" type="number" min="0" value="0" /></div>
          <div><label for="thresholdInput">threshold</label><input id="thresholdInput" type="number" min="0.05" max="0.95" step="0.025" value="0.525" /></div>
          <div><label for="maxPredInput">max pred</label><input id="maxPredInput" type="number" min="1" max="8" step="1" value="3" /></div>
          <button id="randomBtn" class="secondary">Random</button>
          <button id="predictBtn">Run</button>
        </div>
      </div>
      <div class="grid-main">
        <div class="panel">
          <h2>XRD Pattern</h2>
          <canvas id="plot" width="980" height="440"></canvas>
          <div id="sampleNote" class="note">Waiting for a sample.</div>
        </div>
        <div class="panel">
          <h2>Prediction</h2>
          <div id="predictionPanel" class="note">Select a sample and click Run.</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    let app = { overview: null, task: "single", sample: null };
    const $ = (id) => document.getElementById(id);
    const fmtPct = (v) => v === null || v === undefined ? "-" : `${(Number(v) * 100).toFixed(2)}%`;
    const fmtNum = (v, d=3) => v === null || v === undefined ? "-" : Number(v).toFixed(d);
    const fmtMs = (v) => `${Number(v).toFixed(2)} ms`;
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    const splitLabels = {
      "single/val": "Single phase / validation",
      "single/test_normal": "Single phase / normal test",
      "single/test_hard": "Single phase / hard test",
      "mixed_v3/test_normal": "Mixed phase / normal test",
      "mixed_v3/test_hard": "Mixed phase / hard test",
      "hard_eval/test_minor": "Challenge / low-fraction minor phase",
      "hard_eval/test_overlap": "Challenge / peak-overlap pairs",
      "hard_eval/test_battery_relevant": "Challenge / battery-relevant mixtures",
    };
    const splitOptionLabel = (name, count) => `${splitLabels[name] || name} (${count})`;

    async function api(path, options) {
      const res = await fetch(path, options);
      const text = await res.text();
      if (!res.ok) throw new Error(text || res.statusText);
      return JSON.parse(text);
    }
    function setBusy(busy) { document.querySelectorAll("button").forEach(btn => btn.disabled = busy); }
    function table(headers, rows) {
      return `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }
    function showView(name) {
      document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === name));
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.view === name));
    }
    function drawSpectrum(twoTheta, x) {
      const canvas = $("plot"), ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height, pad = {left: 52, right: 18, top: 18, bottom: 42};
      ctx.clearRect(0,0,w,h); ctx.fillStyle="#fbfdff"; ctx.fillRect(0,0,w,h);
      ctx.strokeStyle="#d9e2ea"; ctx.lineWidth=1; ctx.beginPath();
      for (let i=0;i<=4;i++){ const y=pad.top+i*(h-pad.top-pad.bottom)/4; ctx.moveTo(pad.left,y); ctx.lineTo(w-pad.right,y); }
      ctx.stroke(); ctx.fillStyle="#667482"; ctx.font="13px Arial"; ctx.fillText("Intensity",10,22); ctx.fillText("2θ (degree)",w/2-34,h-10);
      const xMin=twoTheta[0], xMax=twoTheta[twoTheta.length-1], yMax=Math.max(...x,1e-6);
      const px = (v) => pad.left+(v-xMin)/(xMax-xMin)*(w-pad.left-pad.right);
      const py = (v) => h-pad.bottom-v/yMax*(h-pad.top-pad.bottom);
      ctx.strokeStyle="#245a8d"; ctx.lineWidth=1.7; ctx.beginPath();
      const step=Math.max(1, Math.floor(x.length/1500));
      for (let i=0;i<x.length;i+=step){ const xx=px(twoTheta[i]), yy=py(x[i]); if(i===0)ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy); }
      ctx.stroke();
    }
    function renderOverview() {
      const m = app.overview.metrics;
      $("metricCards").innerHTML = m.cards.map((card, i) => `<div class="card" style="border-left-color:${["#245a8d","#28785a","#a66b24","#13747c","#6f5b9a","#b64747"][i%6]}"><small>${esc(card.label)}</small><strong>${esc(card.value)}</strong><span>${esc(card.detail)}</span></div>`).join("");
      $("singleTable").innerHTML = table(["split","phase acc","top3 acc","crystal acc","lattice MAE","Bragg MAE"], Object.entries(m.single).map(([name, r]) => [name, fmtPct(r.phase_accuracy), fmtPct(r.phase_top3_accuracy), fmtPct(r.crystal_accuracy), `${fmtNum(r.lattice_mae_mean,3)} A`, `${fmtNum(r.bragg_peak_mae_deg,3)} deg`]));
      $("mixedTable").innerHTML = table(["split","micro P","micro R","micro F1","minor recall","top3 all-hit","FP/sample"], Object.entries(m.mixed).map(([name, r]) => [name, fmtPct(r.micro_precision), fmtPct(r.micro_recall), fmtPct(r.micro_f1), fmtPct(r.minor_phase_recall), fmtPct(r.top3_all_phases_hit), fmtNum(r.avg_false_positives_per_sample,3)]));
      $("mixedHardTable").innerHTML = table(["split","micro F1","minor recall","top3 all-hit","FP/sample"], Object.entries(m.mixed_hard).map(([name, r]) => [name, fmtPct(r.micro_f1), fmtPct(r.minor_phase_recall), fmtPct(r.top3_all_phases_hit), fmtNum(r.avg_false_positives_per_sample,3)]));
      const summary = m.active.summary || {};
      $("activeSummary").innerHTML = table(["field","value"], [["candidate pool", summary.pool_sample_count ?? "-"], ["shortlist", summary.shortlist_count ?? "-"], ["selected", summary.selected_count ?? "-"], ["mean uncertainty", fmtNum(summary.mean_selected_uncertainty,4)], ["similarity threshold", summary.similarity_threshold ?? "-"], ["note", esc(summary.scoring_note ?? "ongoing extension")]]);
      const gain = m.active.gain || {};
      $("activeGainTable").innerHTML = table(["metric","v3 baseline","active round2","change"], [
        ["battery F1", fmtPct(gain.baseline_f1), fmtPct(gain.active_f1), fmtPct(gain.delta_f1)],
        ["battery minor recall", fmtPct(gain.baseline_minor_recall), fmtPct(gain.active_minor_recall), fmtPct(gain.delta_minor_recall)],
        ["top3 all-hit", fmtPct(gain.baseline_top3_all_hit), fmtPct(gain.active_top3_all_hit), fmtPct(gain.delta_top3_all_hit)],
        ["FP/sample", fmtNum(gain.baseline_fp_per_sample,3), fmtNum(gain.active_fp_per_sample,3), fmtNum(gain.delta_fp_per_sample,3)],
        ["active vs random avg F1", fmtPct(gain.random_avg_f1), fmtPct(gain.active_f1), fmtPct((gain.active_f1 ?? 0) - (gain.random_avg_f1 ?? 0))],
        ["active vs random avg minor recall", fmtPct(gain.random_avg_minor_recall), fmtPct(gain.active_minor_recall), fmtPct((gain.active_minor_recall ?? 0) - (gain.random_avg_minor_recall ?? 0))]
      ]);
      $("activeHardTable").innerHTML = table(["run","battery precision","battery recall","battery F1","battery minor recall","top3 all-hit","FP/sample"], (m.active.hard_eval_rows || []).map(r => [r.name, fmtPct(r.battery_micro_precision), fmtPct(r.battery_micro_recall), fmtPct(r.battery_micro_f1), fmtPct(r.battery_minor_recall), fmtPct(r.battery_top3_all_hit), fmtNum(r.battery_fp_per_sample,3)]));
    }
    function refreshSplitOptions() {
      const task = $("taskSelect").value;
      app.task = task;
      const splits = task === "single" ? app.overview.single_splits : app.overview.mixed_splits;
      $("splitSelect").innerHTML = splits.map(s => `<option value="${esc(s.name)}">${esc(splitOptionLabel(s.name, s.count))}</option>`).join("");
      $("thresholdInput").disabled = task === "single";
      $("maxPredInput").disabled = task === "single";
      updateIndexMax();
      loadSample(0);
    }
    function updateIndexMax() {
      const task = $("taskSelect").value, split = $("splitSelect").value;
      const list = task === "single" ? app.overview.single_splits : app.overview.mixed_splits;
      const item = list.find(s => s.name === split);
      $("sampleIndex").max = item ? item.count - 1 : 0;
    }
    async function loadSample(index) {
      const task = $("taskSelect").value, split = $("splitSelect").value;
      updateIndexMax();
      const max = Number($("sampleIndex").max || 0);
      const safe = Math.max(0, Math.min(Number(index), max));
      $("sampleIndex").value = safe;
      const data = await api(`/api/sample?task=${task}&split=${encodeURIComponent(split)}&index=${safe}`);
      app.sample = data;
      drawSpectrum(data.two_theta, data.X);
      if (task === "single") {
        $("sampleNote").innerHTML = `Single-phase sample ${data.index} / true phase: <b>${esc(data.true_label)}</b> / crystal: ${esc(data.true_crystal)} / ${esc(data.category)} ${esc(data.subclass)} ${esc(data.material_id)}`;
      } else {
        const phases = data.true_phases.map(p => `${esc(p.label)} (${fmtPct(p.fraction)})`).join(" / ");
        const extras = Object.entries(data.extra || {}).map(([k,v]) => `${k}: ${esc(v)}`).join(" / ");
        $("sampleNote").innerHTML = `Mixed-phase sample ${data.index} / true phases: <b>${phases}</b><br>${extras}`;
      }
      $("predictionPanel").innerHTML = "Select a sample and click Run.";
    }
    function renderTopList(items) {
      return `<ul class="result-list">${items.map(item => `<li><span>${esc(item.label)}${item.is_true ? " <b class='ok'>true</b>" : ""}</span><div class="bar"><span style="width:${Math.max(2, Number(item.probability)*100)}%"></span></div><span>${fmtPct(item.probability)}</span></li>`).join("")}</ul>`;
    }
    async function predict() {
      setBusy(true);
      try {
        const task = $("taskSelect").value;
        const body = {split: $("splitSelect").value, index: Number($("sampleIndex").value), threshold: Number($("thresholdInput").value), max_predictions: Number($("maxPredInput").value)};
        const result = await api(task === "single" ? "/api/predict_single" : "/api/predict_mixed", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
        if (task === "single") renderSinglePrediction(result); else renderMixedPrediction(result);
      } finally { setBusy(false); }
    }
    function renderSinglePrediction(r) {
      const latticeRows = ["a","b","c"].map((name, i) => `<tr><td>${name}</td><td>${fmtNum(r.lattice_true[i],4)}</td><td>${fmtNum(r.lattice_pred[i],4)}</td><td>${fmtNum(r.lattice_abs_error[i],4)}</td></tr>`).join("");
      $("predictionPanel").innerHTML = `
        <div class="cards">
          <div class="card"><small>phase</small><strong class="${r.phase_correct ? "ok" : "bad"}">${esc(r.pred_label)}</strong><span>true: ${esc(r.true_label)}</span></div>
          <div class="card"><small>crystal</small><strong class="${r.crystal_correct ? "ok" : "bad"}">${esc(r.pred_crystal)}</strong><span>true: ${esc(r.true_crystal)}</span></div>
          <div class="card"><small>phase confidence</small><strong>${fmtPct(r.phase_confidence)}</strong><span>top-1 probability</span></div>
          <div class="card"><small>inference time</small><strong>${fmtMs(r.inference_time_ms)}</strong><span>CPU single sample</span></div>
        </div>
        <h2>Lattice Parameters</h2><table><thead><tr><th>param</th><th>true</th><th>pred</th><th>abs error</th></tr></thead><tbody>${latticeRows}</tbody></table>
        <p class="note">Bragg peak mean error: <b>${fmtNum(r.peak_error_deg,4)} deg</b></p>
        <h2>Top-5 phase confidence</h2>${renderTopList(r.top5)}
        <h2>Top crystal confidence</h2>${renderTopList(r.top_crystal)}
      `;
    }
    function renderMixedPrediction(r) {
      const trueChips = r.true_phases.map(p => `<span class="chip true">${esc(p.label)} ${fmtPct(p.fraction)}</span>`).join("");
      const predChips = r.predicted_phases.map(p => `<span class="chip ${p.is_true ? "true" : "miss"}">${esc(p.label)} ${fmtPct(p.probability)}</span>`).join("");
      const missed = r.missed_phases.length ? r.missed_phases.map(p => `<span class="chip miss">${esc(p.label)} ${fmtPct(p.probability)}</span>`).join("") : "<span class='chip true'>none</span>";
      const fp = r.false_positives.length ? r.false_positives.map(p => `<span class="chip miss">${esc(p.label)} ${fmtPct(p.probability)}</span>`).join("") : "<span class='chip true'>none</span>";
      const metrics = r.overall_metrics || {};
      const metricRows = Object.keys(metrics).length ? `
        <h2>Split-Level Metrics</h2>
        <table><tbody>
          <tr><td>micro-F1</td><td>${fmtPct(metrics.micro_f1)}</td></tr>
          <tr><td>minor recall</td><td>${fmtPct(metrics.minor_phase_recall)}</td></tr>
          <tr><td>top3 all-hit</td><td>${fmtPct(metrics.top3_all_phases_hit)}</td></tr>
          <tr><td>FP/sample</td><td>${fmtNum(metrics.avg_false_positives_per_sample,3)}</td></tr>
        </tbody></table>
        <p class="note">These values summarize the whole split; the yes/no cards above describe only this single sample.</p>
      ` : "";
      $("predictionPanel").innerHTML = `
        <div class="cards">
          <div class="card"><small>all true phases hit</small><strong class="${r.all_hit ? "ok" : "bad"}">${r.all_hit ? "yes" : "no"}</strong><span>single-sample result</span></div>
          <div class="card"><small>true phase coverage</small><strong class="${r.true_phase_coverage >= 1 ? "ok" : "bad"}">${fmtPct(r.true_phase_coverage)}</strong><span>${r.true_phase_hit_count}/${r.true_phase_count} true phases</span></div>
          <div class="card"><small>minor phases hit</small><strong class="${r.minor_hit ? "ok" : "bad"}">${r.minor_hit ? "yes" : "no"}</strong><span>major hit: ${r.major_hit ? "yes" : "no"}</span></div>
          <div class="card"><small>predicted count</small><strong>${r.predicted_phases.length}</strong><span>max ${r.max_predictions}, FP ${r.false_positive_count}</span></div>
          <div class="card"><small>inference time</small><strong>${fmtMs(r.inference_time_ms)}</strong><span>CPU single sample</span></div>
          <div class="card"><small>threshold</small><strong>${fmtNum(r.threshold,3)}</strong><span>default evaluation threshold</span></div>
        </div>
        ${metricRows}
        <h2>True phases</h2><div class="chips">${trueChips}</div>
        <h2>Predicted phases</h2><div class="chips">${predChips || "<span class='chip miss'>none</span>"}</div>
        <h2>Missed phases</h2><div class="chips">${missed}</div>
        <h2>False positives</h2><div class="chips">${fp}</div>
        <h2>Top probabilities</h2>${renderTopList(r.top8)}
      `;
    }
    async function init() {
      app.overview = await api("/api/overview");
      $("thresholdInput").value = Number(app.overview.mixed_default_threshold || 0.525).toFixed(3);
      renderOverview();
      refreshSplitOptions();
    }
    document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));
    $("taskSelect").addEventListener("change", refreshSplitOptions);
    $("splitSelect").addEventListener("change", () => loadSample(0));
    $("sampleIndex").addEventListener("change", () => loadSample($("sampleIndex").value));
    $("randomBtn").addEventListener("click", () => loadSample(Math.floor(Math.random() * (Number($("sampleIndex").max || 0) + 1))));
    $("predictBtn").addEventListener("click", predict);
    init().catch(err => { document.body.innerHTML = `<main><div class="panel"><h1>Dashboard failed to start</h1><p>${esc(err.message)}</p></div></main>`; });
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState | None = None

    def log_message(self, format, *args):
        return

    def send_json(self, payload, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8"):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def parse_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_text(HTML, content_type="text/html; charset=utf-8")
            elif parsed.path == "/api/overview":
                self.send_json(self.state.overview())
            elif parsed.path == "/api/sample":
                query = parse_qs(parsed.query)
                task = query.get("task", ["single"])[0]
                split = query.get("split", [""])[0]
                index = int(query.get("index", ["0"])[0])
                if task == "single":
                    self.send_json(self.state.single_sample(split, index))
                elif task == "mixed":
                    self.send_json(self.state.mixed_sample(split, index))
                else:
                    raise ValueError(f"Unknown task: {task}")
            else:
                self.send_text("Not found", status=404)
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self.parse_body()
            if parsed.path == "/api/predict_single":
                self.send_json(self.state.predict_single(body.get("split", ""), int(body.get("index", 0))))
            elif parsed.path == "/api/predict_mixed":
                self.send_json(
                    self.state.predict_mixed(
                        body.get("split", ""),
                        int(body.get("index", 0)),
                        float(body.get("threshold", self.state.mixed_checkpoint.get("best_threshold", 0.5))),
                        int(body.get("max_predictions", 3)),
                    )
                )
            else:
                self.send_text("Not found", status=404)
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final local XRD characterization dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DashboardHandler.state = DashboardState()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
