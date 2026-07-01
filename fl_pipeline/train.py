"""Training loops: standard FL, DP-SGD FL, FedDPA FL."""

import copy
import math
import time
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant

from .model import TrafficClassifier, fedavg
from .metrics import compute_metrics
from .dp import (
    compute_delta as compute_dp_delta,
    calibrate_noise_multiplier,
    make_dp_loader,
    setup_dp_training,
    get_epsilon,
)
from .feddpa import (
    compute_fisher_diagonal,
    generate_mask,
    init_local_from_mask,
    feddpa_local_train,
    clip_and_noise_shared_delta,
    aggregate_shared_deltas,
    build_personalized_model,
    calibrate_feddpa_noise,
)

HOME_KEYS = ["home_a", "home_b"]


def _get_device(config_name: str) -> torch.device:
    """Always use CPU for cross-platform reproducibility.

    MPS produces slightly different numerics than CPU, which would make
    results non-reproducible across machines.
    """
    return torch.device("cpu")


def _seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One epoch of local training. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model. Returns loss + metrics dict."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item()
            n_batches += 1
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), class_names
    )
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ── Standard FL (baseline + FS) ──────────────────────────────

def run_standard_fl(
    data: Dict, config, seed: int, config_name: str = "baseline_fl",
    return_model: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], "TrafficClassifier"]]:
    """Standard FedAvg — used for baseline_fl, fs_mild, fs_aggressive."""
    _seed_everything(seed)
    device = _get_device(config_name)

    num_features = data["num_features"]
    num_classes = data["num_classes"]
    class_names = data["class_names"]
    weights = [data["weights"]["home_a"], data["weights"]["home_b"]]

    global_model = TrafficClassifier(num_features, num_classes,
                                     hidden_dims=config.hidden_dims).to(device)
    n_params = sum(p.numel() for p in global_model.parameters())
    criterion = nn.CrossEntropyLoss()

    results = {"rounds": [], "config_name": config_name, "seed": seed,
               "hidden_dims": global_model.hidden_dims, "model_params": n_params}

    for r in range(1, config.num_rounds + 1):
        t0 = time.time()
        local_models = []
        round_info = {"round": r}

        for hk in HOME_KEYS:
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=config.lr)

            train_loss = 0.0
            for _ in range(config.local_epochs):
                train_loss = train_epoch(
                    local_model, data[hk]["train_loader"], optimizer, criterion, device
                )
            local_models.append(local_model)
            round_info[f"{hk}_train_loss"] = train_loss

        # FedAvg
        global_model = fedavg(local_models, weights).to(device)

        # Evaluate global model on each home's test set
        for hk in HOME_KEYS:
            m = evaluate(global_model, data[hk]["test_loader"], class_names, device)
            round_info[hk] = m

        round_info["time_s"] = time.time() - t0
        results["rounds"].append(round_info)
        _log_round(r, config.num_rounds, round_info)

    if return_model:
        return results, global_model
    return results


# ── DP-SGD FL ─────────────────────────────────────────────────

def run_dp_sgd_fl(
    data: Dict, config, seed: int,
    return_model: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], "TrafficClassifier"]]:
    """Federated learning with per-sample DP-SGD via Opacus."""
    _seed_everything(seed)
    device = torch.device("cpu")  # Opacus requires CPU

    num_features = data["num_features"]
    num_classes = data["num_classes"]
    class_names = data["class_names"]
    weights = [data["weights"]["home_a"], data["weights"]["home_b"]]

    # Calibrate noise per home (different sizes → different noise_multipliers)
    noise_mults = {}
    deltas_dp = {}
    for hk in HOME_KEYS:
        n_train = data[hk]["n_train"]
        delta = compute_dp_delta(n_train)
        sample_rate = config.batch_size / n_train
        steps_per_epoch = math.ceil(n_train / config.batch_size)
        total_steps = config.num_rounds * config.local_epochs * steps_per_epoch

        nm = calibrate_noise_multiplier(config.target_epsilon, delta, sample_rate, total_steps)
        noise_mults[hk] = nm
        deltas_dp[hk] = delta
        print(f"  {hk}: noise_multiplier={nm:.4f}, delta={delta:.2e}, "
              f"sample_rate={sample_rate:.6f}, total_steps={total_steps}")

    # Persistent accountants (accumulate across rounds)
    accountants = {hk: RDPAccountant() for hk in HOME_KEYS}

    global_model = TrafficClassifier(num_features, num_classes,
                                     hidden_dims=config.hidden_dims).to(device)
    n_params = sum(p.numel() for p in global_model.parameters())
    results = {"rounds": [], "config_name": "dp_sgd", "seed": seed,
               "hidden_dims": global_model.hidden_dims, "model_params": n_params,
               "noise_multipliers": {k: v for k, v in noise_mults.items()},
               "deltas": {k: v for k, v in deltas_dp.items()}}

    for r in range(1, config.num_rounds + 1):
        t0 = time.time()
        local_models = []
        round_info = {"round": r}

        for hk in HOME_KEYS:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()  # Opacus requires training mode
            optimizer = torch.optim.Adam(local_model.parameters(), lr=config.lr)

            # Fresh DataLoader for Opacus wrapping
            train_loader = make_dp_loader(
                data[hk]["X_train"], data[hk]["y_train"], config.batch_size
            )

            local_model, optimizer, train_loader, pe = setup_dp_training(
                local_model, optimizer, train_loader,
                noise_mults[hk], config.max_grad_norm, accountants[hk],
            )

            train_loss = 0.0
            for _ in range(config.local_epochs):
                train_loss = train_epoch(local_model, train_loader, optimizer,
                                         nn.CrossEntropyLoss(), device)

            eps = get_epsilon(pe, deltas_dp[hk])
            round_info[f"{hk}_epsilon"] = eps
            round_info[f"{hk}_train_loss"] = train_loss

            # Unwrap for FedAvg
            raw = local_model._module if hasattr(local_model, "_module") else local_model
            local_models.append(raw)

        global_model = fedavg(local_models, weights).to(device)

        for hk in HOME_KEYS:
            m = evaluate(global_model, data[hk]["test_loader"], class_names, device)
            round_info[hk] = m

        round_info["time_s"] = time.time() - t0
        results["rounds"].append(round_info)
        _log_round(r, config.num_rounds, round_info,
                   extra=f"eps_a={round_info.get('home_a_epsilon', '?'):.4f} "
                         f"eps_b={round_info.get('home_b_epsilon', '?'):.4f}")

    if return_model:
        return results, global_model
    return results


# ── FedDPA FL ─────────────────────────────────────────────────

def run_feddpa_fl(
    data: Dict, config, seed: int
) -> Dict[str, Any]:
    """Federated learning with FedDPA (Fisher masks + update-level DP).

    Key differences from standard FL:
    - Personal params (high Fisher) stay local, never noised or sent to server
    - Shared params (low Fisher) get clipped + noised → aggregated by server
    - Each home is evaluated on its personalized model (global shared + local personal)
    - Maintains per-home local state across rounds for personal param continuity
    """
    _seed_everything(seed)
    device = torch.device("cpu")

    num_features = data["num_features"]
    num_classes = data["num_classes"]
    class_names = data["class_names"]
    weights = [data["weights"]["home_a"], data["weights"]["home_b"]]

    # Calibrate update-level noise per home
    noise_mults = {}
    deltas_dp = {}
    for hk in HOME_KEYS:
        n_train = data[hk]["n_train"]
        delta = min(1e-5, 1.0 / (10 * n_train))
        nm = calibrate_feddpa_noise(config.target_epsilon, delta, config.num_rounds)
        noise_mults[hk] = nm
        deltas_dp[hk] = delta
        print(f"  FedDPA {hk}: noise_multiplier={nm:.4f}, delta={delta:.2e}")

    # Persistent accountants for update-level DP
    accountants = {hk: RDPAccountant() for hk in HOME_KEYS}

    global_model = TrafficClassifier(num_features, num_classes,
                                     hidden_dims=config.hidden_dims).to(device)
    n_params = sum(p.numel() for p in global_model.parameters())

    # Per-home state: previous round's local params (for personal param continuity)
    prev_local_params = {hk: None for hk in HOME_KEYS}

    results = {"rounds": [], "config_name": "feddpa", "seed": seed,
               "hidden_dims": global_model.hidden_dims, "model_params": n_params,
               "noise_multipliers": {k: v for k, v in noise_mults.items()},
               "deltas": {k: v for k, v in deltas_dp.items()}}

    for r in range(1, config.num_rounds + 1):
        t0 = time.time()
        round_info = {"round": r, "mask_stats": {}}

        shared_deltas = []
        round_masks = {}

        for hk in HOME_KEYS:
            # 1. Compute Fisher on current global model using local data → mask
            fisher = compute_fisher_diagonal(
                copy.deepcopy(global_model), data[hk]["X_train"], data[hk]["y_train"],
                n_samples=config.fisher_n_samples, device=device,
            )
            mask, mask_stats = generate_mask(fisher, config.fisher_threshold)
            round_masks[hk] = mask
            round_info["mask_stats"][hk] = mask_stats

            # 2. Init local model: shared from global, personal from prev local
            local_model = init_local_from_mask(
                global_model, mask, prev_local_params[hk], device,
            )

            # 3. Train with mask-aware L2 regularization
            global_params = {n: p.data.clone() for n, p in global_model.named_parameters()}
            local_model = feddpa_local_train(
                local_model, data[hk]["X_train"], data[hk]["y_train"],
                global_params, prev_local_params[hk], mask,
                config.local_epochs, config.lr, config.lambda_reg,
                config.batch_size, device,
            )

            # 4. Save local params for next round (before clipping/noising)
            prev_local_params[hk] = {
                n: p.data.clone().cpu() for n, p in local_model.named_parameters()
            }

            # 5. Clip + noise shared-param delta only
            delta_noised, delta_norm = clip_and_noise_shared_delta(
                local_model, global_model, mask,
                config.max_grad_norm, noise_mults[hk], device,
            )
            shared_deltas.append(delta_noised)
            round_info[f"{hk}_delta_norm"] = delta_norm

            # 6. Update DP accountant
            accountants[hk].step(noise_multiplier=noise_mults[hk], sample_rate=1.0)
            eps = accountants[hk].get_epsilon(deltas_dp[hk])
            round_info[f"{hk}_epsilon"] = eps

        # 7. Server: aggregate shared deltas → update global model
        global_model = aggregate_shared_deltas(
            global_model, shared_deltas, weights,
        ).to(device)

        # 8. Evaluate personalized models (global shared + local personal)
        for hk in HOME_KEYS:
            personal_model = build_personalized_model(
                global_model, round_masks[hk], prev_local_params[hk], device,
            )
            m = evaluate(personal_model, data[hk]["test_loader"], class_names, device)
            round_info[hk] = m

        round_info["time_s"] = time.time() - t0
        results["rounds"].append(round_info)
        pf_a = round_info["mask_stats"].get("home_a", {}).get("_overall_personal_frac", 0)
        pf_b = round_info["mask_stats"].get("home_b", {}).get("_overall_personal_frac", 0)
        _log_round(r, config.num_rounds, round_info,
                   extra=f"eps_a={round_info.get('home_a_epsilon', '?'):.4f} "
                         f"eps_b={round_info.get('home_b_epsilon', '?'):.4f} "
                         f"personal_a={pf_a:.1%} personal_b={pf_b:.1%}")

    return results


# ── Helpers ───────────────────────────────────────────────────

def _log_round(r: int, total: int, info: Dict, extra: str = ""):
    """Print round summary."""
    a = info.get("home_a", {})
    b = info.get("home_b", {})
    t = info.get("time_s", 0)
    parts = [
        f"  Round {r:2d}/{total}",
        f"A-F1={a.get('macro_f1', 0):.3f}",
        f"B-F1={b.get('macro_f1', 0):.3f}",
        f"A-worst={a.get('worst_group_f1', 0):.3f}",
        f"B-worst={b.get('worst_group_f1', 0):.3f}",
        f"({t:.1f}s)",
    ]
    if extra:
        parts.append(extra)
    print(" | ".join(parts))
