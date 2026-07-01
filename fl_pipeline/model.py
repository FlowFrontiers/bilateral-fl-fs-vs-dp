"""MLP model and federated averaging."""

import copy
from typing import List, Optional

import torch
import torch.nn as nn


def default_hidden_dims(input_dim: int, num_classes: int) -> List[int]:
    """Original 646-param architecture: [(2/3)*input + classes, input]."""
    return [int((2 / 3) * input_dim + num_classes), input_dim]


class TrafficClassifier(nn.Module):
    """MLP for traffic classification with configurable hidden layers.

    Default hidden_dims reproduces the original 646-param architecture.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = default_hidden_dims(input_dim, num_classes)
        self.hidden_dims = hidden_dims

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def fedavg(models: List[nn.Module], weights: List[float]) -> TrafficClassifier:
    """Weighted federated averaging. Handles Opacus-wrapped models."""
    # Extract state dicts, unwrapping Opacus _module if present
    state_dicts = []
    for m in models:
        if hasattr(m, "_module"):
            state_dicts.append(m._module.state_dict())
        else:
            state_dicts.append(m.state_dict())

    # Weighted average
    avg_sd = {}
    for key in state_dicts[0]:
        avg_sd[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )

    # Create fresh model with averaged parameters
    ref = models[0]._module if hasattr(models[0], "_module") else models[0]
    new_model = TrafficClassifier(
        ref.model[0].in_features, ref.model[-1].out_features,
        hidden_dims=ref.hidden_dims,
    )
    new_model.load_state_dict(avg_sd)
    return new_model
