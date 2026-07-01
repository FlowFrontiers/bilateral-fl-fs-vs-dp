"""FedDPA: Fisher-based personalization masks + update-level DP.

Simplified from Yang (NeurIPS 2023). Each round per client:
1. Compute diagonal Fisher on subsampled local data → mask
2. Init local model: shared params from global, personal params from prev local
3. Train with mask-aware L2 (personal → prev local, shared → global)
4. Clip + noise shared-param delta only → send to server
5. Server aggregates shared deltas → new global
6. Evaluate personalized models (global shared + local personal)
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant


def compute_fisher_diagonal(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 5000,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Empirical Fisher diagonal: E[(dL/dtheta)^2] over subsampled data."""
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = torch.from_numpy(X[idx]).float().to(device)
    y_sub = torch.from_numpy(y[idx]).long().to(device)

    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    batch_size = 256
    for start in range(0, len(X_sub), batch_size):
        end = min(start + batch_size, len(X_sub))
        xb, yb = X_sub[start:end], y_sub[start:end]

        model.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.data ** 2 * len(xb)

    n = len(X_sub)
    for name in fisher:
        fisher[name] /= n

    return fisher


def generate_mask(
    fisher: Dict[str, torch.Tensor], threshold: float = 0.4
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Layer-wise min-max normalize, then threshold.

    Returns binary mask (1 = personal/keep local, 0 = shared/use global)
    and per-layer statistics.
    """
    mask = {}
    stats = {}

    for name, f in fisher.items():
        f_min, f_max = f.min(), f.max()
        if f_max - f_min > 1e-10:
            f_norm = (f - f_min) / (f_max - f_min)
        else:
            f_norm = torch.zeros_like(f)

        m = (f_norm >= threshold).float()
        mask[name] = m
        stats[name] = float(m.sum() / m.numel())

    total_personal = sum(m.sum().item() for m in mask.values())
    total_params = sum(m.numel() for m in mask.values())
    stats["_overall_personal_frac"] = total_personal / total_params if total_params > 0 else 0.0

    return mask, stats


def init_local_from_mask(
    global_model: nn.Module,
    mask: Dict[str, torch.Tensor],
    prev_local_params: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> nn.Module:
    """Initialize local model: shared from global, personal from prev local.

    Round 1 (prev_local_params=None): all params from global.
    Later rounds: personal params (mask=1) from previous local training.
    """
    local_model = copy.deepcopy(global_model).to(device)
    if prev_local_params is not None:
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                m = mask[name].to(device)
                if m.any():
                    prev = prev_local_params[name].to(device)
                    param.data = param.data * (1.0 - m) + prev * m
    local_model.train()
    return local_model


def feddpa_local_train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    global_params: Dict[str, torch.Tensor],
    prev_local_params: Optional[Dict[str, torch.Tensor]],
    mask: Dict[str, torch.Tensor],
    local_epochs: int,
    lr: float,
    lambda_reg: float,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Train with mask-aware L2 regularization.

    Personal params (mask=1): regularize toward prev_local (stability).
    Shared params (mask=0): regularize toward global (coherence).
    Round 1 fallback: both regularize toward global.
    """
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-move references to device
    g_ref = {n: p.to(device) for n, p in global_params.items()}
    if prev_local_params is not None:
        l_ref = {n: p.to(device) for n, p in prev_local_params.items()}
    else:
        l_ref = g_ref  # Round 1 fallback

    mask_d = {n: m.to(device) for n, m in mask.items()}

    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)

    n = len(X_t)
    for epoch in range(local_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)

            # Mask-aware L2 regularization
            l2 = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                m = mask_d[name]
                # Personal (m=1): toward previous local
                l2 = l2 + (m * (param - l_ref[name]) ** 2).sum()
                # Shared (m=0): toward global
                l2 = l2 + ((1.0 - m) * (param - g_ref[name]) ** 2).sum()

            total_loss = loss + lambda_reg * l2
            total_loss.backward()
            optimizer.step()

    return model


def clip_and_noise_shared_delta(
    local_model: nn.Module,
    global_model: nn.Module,
    mask: Dict[str, torch.Tensor],
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Compute shared-param delta only, clip, add noise.

    Personal params (mask=1) are zeroed out — they stay local, never sent to server.
    Returns noised shared delta dict and the pre-clip delta norm.
    """
    delta = {}
    global_params = dict(global_model.named_parameters())
    for name, param in local_model.named_parameters():
        shared_mask = (1.0 - mask[name]).to(device)
        delta[name] = (param.data - global_params[name].data.to(device)) * shared_mask

    # L2 norm of shared delta
    flat = torch.cat([d.flatten() for d in delta.values()])
    delta_norm = flat.norm(2).item()

    # Clip
    clip_factor = min(1.0, max_grad_norm / max(delta_norm, 1e-10))
    for name in delta:
        delta[name] = delta[name] * clip_factor

    # Noise on shared params only
    noise_std = noise_multiplier * max_grad_norm
    for name in delta:
        shared_mask = (1.0 - mask[name]).to(device)
        delta[name] = delta[name] + torch.randn_like(delta[name]) * noise_std * shared_mask

    return delta, delta_norm


def aggregate_shared_deltas(
    global_model: nn.Module,
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> nn.Module:
    """Server-side: apply weighted average of shared deltas to global model."""
    new_model = copy.deepcopy(global_model)
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            avg_delta = sum(w * d[name] for w, d in zip(weights, deltas))
            param.data += avg_delta
    return new_model


def build_personalized_model(
    global_model: nn.Module,
    mask: Dict[str, torch.Tensor],
    local_params: Dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Combine global (shared) + local (personal) into personalized model."""
    model = copy.deepcopy(global_model).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            m = mask[name].to(device)
            if m.any():
                local_p = local_params[name].to(device)
                param.data = param.data * (1.0 - m) + local_p * m
    return model


def calibrate_feddpa_noise(
    target_epsilon: float,
    delta: float,
    num_rounds: int,
) -> float:
    """Calibrate noise_multiplier for update-level DP over num_rounds.

    FedDPA: one noised release per round, sample_rate=1.0 (full participation).
    """
    low, high = 0.01, 200.0

    for _ in range(100):
        mid = (low + high) / 2
        acct = RDPAccountant()
        acct.history = [(mid, 1.0, num_rounds)]
        try:
            eps = acct.get_epsilon(delta)
        except Exception:
            eps = float("inf")

        if eps > target_epsilon:
            low = mid
        else:
            high = mid

        if high - low < 1e-6:
            break

    return high
