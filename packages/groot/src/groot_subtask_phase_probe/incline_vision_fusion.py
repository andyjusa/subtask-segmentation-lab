from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


class ProjectedFusionProbe(nn.Module):
    """Compress vision and motion independently before joint classification."""

    def __init__(self, vision_dim: int, auxiliary_dim: int) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.vision = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.auxiliary = nn.Sequential(
            nn.Linear(auxiliary_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        vision = self.vision(features[:, : self.vision_dim])
        auxiliary = self.auxiliary(features[:, self.vision_dim :])
        return self.head(torch.cat([vision, auxiliary], dim=1))


class VisionOnlyProbe(nn.Module):
    """Reproduce the original R3b MLP while ignoring appended modalities."""

    def __init__(self, vision_dim: int) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.head = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features[:, : self.vision_dim])


class ResidualFusionProbe(nn.Module):
    """Use motion as a bounded correction to a strong vision classifier."""

    def __init__(self, vision_dim: int, auxiliary_dim: int) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.vision = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
        self.auxiliary = nn.Sequential(
            nn.Linear(auxiliary_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
        self.auxiliary_gate_logit = nn.Parameter(torch.full((5,), -2.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        vision = self.vision(features[:, : self.vision_dim])
        auxiliary = self.auxiliary(features[:, self.vision_dim :])
        gate = torch.sigmoid(self.auxiliary_gate_logit)
        return vision + gate * auxiliary


class TriModalResidualProbe(nn.Module):
    """Correct vision logits with separately gated full-hidden and motion branches."""

    def __init__(self, vision_dim: int, context_dim: int, auxiliary_dim: int) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.context_dim = context_dim
        self.vision = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
        self.auxiliary = nn.Sequential(
            nn.Linear(auxiliary_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
        self.context_gate_logit = nn.Parameter(torch.full((5,), -2.0))
        self.auxiliary_gate_logit = nn.Parameter(torch.full((5,), -2.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context_start = self.vision_dim
        auxiliary_start = context_start + self.context_dim
        vision = self.vision(features[:, :context_start])
        context = self.context(features[:, context_start:auxiliary_start])
        auxiliary = self.auxiliary(features[:, auxiliary_start:])
        context_gate = torch.sigmoid(self.context_gate_logit)
        auxiliary_gate = torch.sigmoid(self.auxiliary_gate_logit)
        return vision + context_gate * context + auxiliary_gate * auxiliary


def model_factories(
    vision_dim: int,
    auxiliary_dim: int,
    context_dim: int = 0,
) -> dict[str, Callable[[], nn.Module]]:
    factories: dict[str, Callable[[], nn.Module]] = {
        "vision_only": lambda: VisionOnlyProbe(vision_dim),
        "projected": lambda: ProjectedFusionProbe(vision_dim, auxiliary_dim),
        "residual": lambda: ResidualFusionProbe(vision_dim, auxiliary_dim),
    }
    if context_dim:
        factories = {
            "tri_residual": lambda: TriModalResidualProbe(
                vision_dim, context_dim, auxiliary_dim
            )
        }
    return factories


def append_causal_deltas(features: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Append current-minus-past features without reading future samples."""

    if features.ndim != 2:
        raise ValueError("features must have shape [time, dimension]")
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive")
    output = [features]
    for lag in lags:
        past = np.concatenate(
            [
                np.repeat(features[:1], min(lag, len(features)), axis=0),
                features[:-lag] if lag < len(features) else features[:0],
            ],
            axis=0,
        )
        output.append(features - past)
    return np.concatenate(output, axis=1).astype(np.float32, copy=False)
