from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from model_multitask import ConvBlock


class MixedPhaseXRDConvNet(nn.Module):
    """1D-CNN baseline for multi-label mixed-phase XRD recognition."""

    def __init__(
        self,
        num_phase_classes: int,
        pooled_bins: int = 32,
        dropout: float = 0.20,
        hidden_dim: int = 512,
        head_hidden_dim: int | None = None,
        use_fraction_level_head: bool = True,
        fraction_level_count: int = 3,
    ) -> None:
        super().__init__()
        self.num_phase_classes = num_phase_classes
        self.pooled_bins = pooled_bins
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.use_fraction_level_head = use_fraction_level_head
        self.fraction_level_count = fraction_level_count

        self.backbone = nn.Sequential(
            ConvBlock(1, 32, kernel_size=11, pool=True),
            ConvBlock(32, 64, kernel_size=9, pool=True),
            ConvBlock(64, 128, kernel_size=7, pool=True),
            ConvBlock(128, 128, kernel_size=5, pool=True),
        )
        self.position_pool = nn.AdaptiveAvgPool1d(pooled_bins)
        feature_dim = 128 * pooled_bins

        self.shared_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if head_hidden_dim is None or head_hidden_dim <= 0:
            self.phase_head = nn.Linear(hidden_dim, num_phase_classes)
        else:
            self.phase_head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_phase_classes),
            )
        if use_fraction_level_head:
            self.fraction_level_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_phase_classes * fraction_level_count),
            )
        else:
            self.fraction_level_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.position_pool(features).flatten(1)
        shared = self.shared_projection(pooled)
        phase_logits = self.phase_head(shared)
        outputs = {"phase_logits": phase_logits}
        if self.fraction_level_head is not None:
            fraction_level_logits = self.fraction_level_head(shared)
            outputs["fraction_level_logits"] = fraction_level_logits.view(
                x.shape[0],
                self.num_phase_classes,
                self.fraction_level_count,
            )
        return outputs

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)["phase_logits"])


def load_single_phase_backbone(
    model: MixedPhaseXRDConvNet,
    checkpoint_path: Path,
    device: torch.device,
    include_phase_head: bool = False,
) -> dict[str, Any]:
    """Load compatible single-phase weights into the mixed-phase model.

    By default, only the convolutional backbone and shared projection are loaded.
    The softmax-trained single-phase phase head is not loaded unless explicitly
    requested, because the mixed-phase head is optimized with sigmoid BCE.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_shape: list[str] = []

    allowed_prefixes = ("backbone.", "shared_projection.")
    if include_phase_head:
        allowed_prefixes = allowed_prefixes + ("phase_head.",)

    for key, value in source_state.items():
        if not key.startswith(allowed_prefixes):
            continue
        if key not in target_state:
            continue
        if target_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        target_state[key] = value
        loaded_keys.append(key)

    model.load_state_dict(target_state)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_key_count": len(loaded_keys),
        "loaded_keys": loaded_keys,
        "skipped_shape_keys": skipped_shape,
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_best_metric_name": checkpoint.get("best_metric_name"),
        "source_best_metric_value": checkpoint.get("best_metric_value"),
    }
