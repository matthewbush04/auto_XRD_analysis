from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model_mixture import MixedPhaseXRDConvNet
from train_mixed_phase_cnn import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "active_learning"
RANDOM_SEED = 42


class CandidatePoolDataset(Dataset):
    def __init__(self, npz_path: Path, max_samples: int | None = None, seed: int = RANDOM_SEED) -> None:
        data = np.load(npz_path, allow_pickle=True)
        if "X" not in data.files:
            raise ValueError(f"{npz_path} is missing X.")
        n_samples = int(data["X"].shape[0])
        if max_samples is not None and max_samples > 0 and max_samples < n_samples:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False)).astype(np.int64)
        else:
            self.indices = np.arange(n_samples, dtype=np.int64)
        self.X = torch.from_numpy(data["X"][self.indices].astype(np.float32)).unsqueeze(1)
        self.labels = [str(label) for label in data["labels"]] if "labels" in data.files else []
        self.metadata = self._load_metadata(data)

    def _load_metadata(self, data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
        metadata: dict[str, np.ndarray] = {}
        for key in (
            "pool_sample_index",
            "hard_eval_type",
            "minor_fraction_bin",
            "overlap_bin",
            "battery_scenario",
            "n_phases",
        ):
            if key in data.files:
                values = np.asarray(data[key])
                if values.ndim > 0 and values.shape[0] >= len(self.indices):
                    metadata[key] = values[self.indices]
        return metadata

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self.X[idx],
            "sample_index": torch.tensor(int(self.indices[idx]), dtype=torch.long),
        }


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[MixedPhaseXRDConvNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model_config = checkpoint.get("model_config")
    if model_config is None:
        labels = checkpoint.get("phase_labels")
        if labels is None:
            raise ValueError("Checkpoint has no model_config or phase_labels; cannot infer model shape.")
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
        model_config["use_fraction_level_head"] = any(key.startswith("fraction_level_head.") for key in state_dict)
    model = MixedPhaseXRDConvNet(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def resolve_threshold(args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[float, str]:
    if args.threshold is not None:
        return float(args.threshold), "argument"
    if args.metrics_summary is not None:
        metrics = load_json(args.metrics_summary)
        if "best_threshold" in metrics:
            return float(metrics["best_threshold"]), f"metrics_summary:{args.metrics_summary}"
    if "best_threshold" in checkpoint:
        return float(checkpoint["best_threshold"]), "checkpoint"
    return 0.5, "default"


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)


def top_indices(probabilities: np.ndarray, top_m: int) -> np.ndarray:
    top_m = max(1, min(int(top_m), probabilities.shape[1]))
    return np.argpartition(-probabilities, kth=top_m - 1, axis=1)[:, :top_m]


def row_take(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices, axis=1)


def compute_uncertainty_scores(
    probabilities: np.ndarray,
    threshold: float,
    margin: float,
    top_m: int,
) -> dict[str, np.ndarray]:
    indices = top_indices(probabilities, top_m)
    top_probs = row_take(probabilities, indices)
    entropy = row_take(binary_entropy(probabilities), indices).mean(axis=1)
    proximity_values = 1.0 - np.minimum(np.abs(top_probs - threshold) / max(margin, 1e-6), 1.0)
    threshold_proximity = proximity_values.mean(axis=1)
    near_threshold_count = (np.abs(probabilities - threshold) <= margin).sum(axis=1).astype(np.float32)
    near_threshold_score = np.minimum(near_threshold_count / max(float(top_m), 1.0), 1.0)
    uncertainty = 0.50 * entropy + 0.35 * threshold_proximity + 0.15 * near_threshold_score
    return {
        "uncertainty_score": uncertainty.astype(np.float32),
        "topm_entropy": entropy.astype(np.float32),
        "threshold_proximity": threshold_proximity.astype(np.float32),
        "near_threshold_count": near_threshold_count.astype(np.int64),
        "near_threshold_score": near_threshold_score.astype(np.float32),
    }


@torch.no_grad()
def predict_pool(
    model: nn.Module,
    dataset: CandidatePoolDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = model.shared_projection.register_forward_hook(hook)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    try:
        for batch in loader:
            captured.clear()
            x = batch["X"].to(device, non_blocking=True)
            outputs = model(x)
            prob = torch.sigmoid(outputs["phase_logits"]).detach().cpu().numpy()
            if not captured:
                raise RuntimeError("Feature hook did not capture shared_projection output.")
            probabilities.append(prob.astype(np.float32))
            features.append(captured[-1].cpu().numpy().astype(np.float32))
            sample_indices.append(batch["sample_index"].numpy().astype(np.int64))
    finally:
        handle.remove()
    return np.concatenate(probabilities), np.concatenate(features), np.concatenate(sample_indices)


def normalized_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def select_diverse(
    order: np.ndarray,
    features: np.ndarray,
    select_count: int,
    similarity_threshold: float,
) -> tuple[list[int], dict[int, float], int]:
    selected: list[int] = []
    diversity_scores: dict[int, float] = {}
    skipped = 0
    normalized = normalized_features(features)
    selected_matrix = np.empty((select_count, normalized.shape[1]), dtype=np.float32)
    for idx in order:
        idx = int(idx)
        candidate = normalized[idx].astype(np.float32, copy=False)
        if not selected:
            selected.append(idx)
            diversity_scores[idx] = 1.0
            selected_matrix[0] = candidate
        else:
            current = selected_matrix[: len(selected)]
            similarities = np.sum(current * candidate[None, :], axis=1, dtype=np.float32)
            max_similarity = float(np.max(similarities))
            if max_similarity > similarity_threshold:
                skipped += 1
                continue
            selected.append(idx)
            diversity_scores[idx] = float(1.0 - max_similarity)
            selected_matrix[len(selected) - 1] = candidate
        if len(selected) >= select_count:
            break
    return selected, diversity_scores, skipped


def label_for(labels: list[str], phase_idx: int) -> str:
    if 0 <= phase_idx < len(labels):
        return labels[phase_idx]
    return str(phase_idx)


def selection_reason(row: dict[str, Any], threshold: float) -> str:
    reasons: list[str] = []
    if int(row["near_threshold_count"]) > 0:
        reasons.append("near-threshold phase probabilities")
    if float(row["topm_entropy"]) >= 0.50:
        reasons.append("high multi-label entropy")
    if abs(float(row["top2_prob"]) - threshold) <= abs(float(row["top1_prob"]) - threshold):
        reasons.append("ambiguous secondary phase")
    if not reasons:
        reasons.append("high combined uncertainty")
    return "; ".join(reasons)


def build_rows(
    sample_indices: np.ndarray,
    probabilities: np.ndarray,
    scores: dict[str, np.ndarray],
    labels: list[str],
    metadata: dict[str, np.ndarray],
    order: np.ndarray,
    selected_positions: set[int] | None = None,
    diversity_scores: dict[int, float] | None = None,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    selected_positions = selected_positions or set()
    diversity_scores = diversity_scores or {}
    for rank, pos in enumerate(order, start=1):
        pos = int(pos)
        top = top5[pos]
        row: dict[str, Any] = {
            "rank": rank,
            "selected": int(pos in selected_positions),
            "sample_index": int(sample_indices[pos]),
            "pool_position": pos,
            "uncertainty_score": float(scores["uncertainty_score"][pos]),
            "diversity_score": float(diversity_scores.get(pos, np.nan)),
            "final_score": float(scores["uncertainty_score"][pos]),
            "topm_entropy": float(scores["topm_entropy"][pos]),
            "threshold_proximity": float(scores["threshold_proximity"][pos]),
            "near_threshold_count": int(scores["near_threshold_count"][pos]),
            "predicted_phase_count": int(np.sum(probabilities[pos] >= threshold)),
        }
        for top_rank in range(5):
            phase_idx = int(top[top_rank])
            row[f"top{top_rank + 1}_phase_idx"] = phase_idx
            row[f"top{top_rank + 1}_phase_label"] = label_for(labels, phase_idx)
            row[f"top{top_rank + 1}_prob"] = float(probabilities[pos, phase_idx])
        for key, values in metadata.items():
            value = values[pos]
            row[key] = str(value) if np.asarray(value).dtype.kind in {"U", "S", "O"} else int(value)
        row["selection_reason"] = selection_reason(row, threshold)
        rows.append(row)
    return rows


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.results_root / f"round_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and select hidden-label mixed-phase candidates for active learning."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--metrics-summary", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-margin", type=float, default=0.20)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--select-count", type=int, default=2_000)
    parser.add_argument("--shortlist-count", type=int, default=5_000)
    parser.add_argument("--similarity-threshold", type=float, default=0.98)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = make_output_dir(args)
    if path_has_files(output_dir) and not args.overwrite:
        print(
            "Refusing to overwrite an existing active-learning result directory. "
            "Choose --output-dir or pass --overwrite.\n"
            f"{output_dir}",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = resolve_device(args.device)
        model, checkpoint = load_checkpoint_model(args.checkpoint, device)
        threshold, threshold_source = resolve_threshold(args, checkpoint)
        dataset = CandidatePoolDataset(args.candidate_pool, args.max_samples, args.seed)
        if not dataset.labels:
            labels = [str(label) for label in checkpoint.get("phase_labels", [])]
        else:
            labels = dataset.labels

        probabilities, features, sample_indices = predict_pool(
            model,
            dataset,
            int(args.batch_size),
            int(args.num_workers),
            device,
        )
        scores = compute_uncertainty_scores(
            probabilities,
            threshold,
            float(args.threshold_margin),
            int(args.top_m),
        )
        full_order = np.argsort(-scores["uncertainty_score"])
        shortlist_count = min(int(args.shortlist_count), len(full_order))
        shortlist = full_order[:shortlist_count]
        selected, diversity_scores, skipped = select_diverse(
            shortlist,
            features,
            min(int(args.select_count), len(shortlist)),
            float(args.similarity_threshold),
        )
        selected_set = set(selected)
        ranked_rows = build_rows(
            sample_indices,
            probabilities,
            scores,
            labels,
            dataset.metadata,
            full_order,
            selected_set,
            diversity_scores,
            threshold,
        )
        selected_order = np.asarray(selected, dtype=np.int64)
        selected_rows = build_rows(
            sample_indices,
            probabilities,
            scores,
            labels,
            dataset.metadata,
            selected_order,
            selected_set,
            diversity_scores,
            threshold,
        )
        for idx, row in enumerate(selected_rows, start=1):
            row["selection_rank"] = idx

        write_csv(output_dir / "ranked_candidates.csv", ranked_rows)
        write_csv(output_dir / "selected_active_candidates.csv", selected_rows)
        summary = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(args.checkpoint),
            "candidate_pool": str(args.candidate_pool),
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "threshold_margin": float(args.threshold_margin),
            "top_m": int(args.top_m),
            "pool_sample_count": int(len(dataset)),
            "shortlist_count": int(shortlist_count),
            "requested_select_count": int(args.select_count),
            "selected_count": int(len(selected)),
            "similarity_threshold": float(args.similarity_threshold),
            "skipped_by_similarity": int(skipped),
            "mean_selected_uncertainty": float(np.mean(scores["uncertainty_score"][selected_order])) if len(selected) else 0.0,
            "scoring_note": "Scores use model probabilities and embeddings only; ground-truth pool labels are not read.",
        }
        write_json(output_dir / "active_learning_summary.json", summary)
        print(f"Saved active-learning ranking to {output_dir}")
        print(f"Selected {len(selected)} candidates from {len(dataset)} pool samples.")
    except Exception as exc:
        print(f"Active-learning scoring failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
