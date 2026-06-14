from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiTaskXRDConvNet(nn.Module):
    """Shared 1D-CNN backbone with phase, crystal-system, and lattice heads."""

    def __init__(
        self,
        num_phase_classes: int,
        num_crystal_classes: int,
        lattice_dim: int = 3,
        pooled_bins: int = 32,
        dropout: float = 0.20,
        lattice_hidden_dims: Sequence[int] = (256,),
        phase_conditioned_lattice: bool = False,
        phase_embedding_dim: int = 64,
        detach_phase_for_lattice: bool = True,
    ) -> None:
        super().__init__()
        self.num_phase_classes = num_phase_classes
        self.num_crystal_classes = num_crystal_classes
        self.lattice_dim = lattice_dim
        self.pooled_bins = pooled_bins
        self.lattice_hidden_dims = list(lattice_hidden_dims)
        self.phase_conditioned_lattice = phase_conditioned_lattice
        self.phase_embedding_dim = phase_embedding_dim
        self.detach_phase_for_lattice = detach_phase_for_lattice

        self.backbone = nn.Sequential(
            ConvBlock(1, 32, kernel_size=11, pool=True),
            ConvBlock(32, 64, kernel_size=9, pool=True),
            ConvBlock(64, 128, kernel_size=7, pool=True),
            ConvBlock(128, 128, kernel_size=5, pool=True),
        )
        self.position_pool = nn.AdaptiveAvgPool1d(pooled_bins)
        feature_dim = 128 * pooled_bins

        self.shared_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phase_head = nn.Linear(512, num_phase_classes)
        self.crystal_head = nn.Linear(512, num_crystal_classes)
        if phase_conditioned_lattice:
            self.phase_embedding = nn.Sequential(
                nn.Linear(num_phase_classes, phase_embedding_dim),
                nn.LayerNorm(phase_embedding_dim),
                nn.GELU(),
            )
            lattice_input_dim = 512 + phase_embedding_dim
        else:
            self.phase_embedding = None
            lattice_input_dim = 512

        lattice_layers: list[nn.Module] = []
        in_dim = lattice_input_dim
        for hidden_dim in self.lattice_hidden_dims:
            lattice_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        lattice_layers.append(nn.Linear(in_dim, lattice_dim))
        self.lattice_head = nn.Sequential(*lattice_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.position_pool(features).flatten(1)
        shared = self.shared_projection(pooled)
        phase_logits = self.phase_head(shared)
        crystal_logits = self.crystal_head(shared)
        lattice_input = shared
        if self.phase_embedding is not None:
            phase_prob = torch.softmax(phase_logits, dim=1)
            if self.detach_phase_for_lattice:
                phase_prob = phase_prob.detach()
            phase_embedding = self.phase_embedding(phase_prob)
            lattice_input = torch.cat([shared, phase_embedding], dim=1)
        lattice_output = self.lattice_head(lattice_input)
        return {
            "phase_logits": phase_logits,
            "crystal_logits": crystal_logits,
            "lattice_norm": lattice_output,
            "lattice_raw": lattice_output,
        }
