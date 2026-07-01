"""Membership Inference Attacks: loss-based and shadow-model."""

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


N_CAP = 2000  # Max samples per class for MIA evaluation


def compute_per_sample_loss(
    model: nn.Module, X: np.ndarray, y: np.ndarray,
    device: torch.device, batch_size: int = 512,
) -> np.ndarray:
    """Per-sample cross-entropy loss (lower = more memorized)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            end = min(start + batch_size, len(X_t))
            out = model(X_t[start:end])
            loss = criterion(out, y_t[start:end])
            losses.append(loss.cpu().numpy())

    return np.concatenate(losses)


def sample_balanced_indices(
    y_train: np.ndarray, y_test: np.ndarray, n_classes: int,
    seed: int, n_cap: int = N_CAP,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Precompute balanced member/non-member indices per class.

    For each class: n = min(n_train_c, n_test_c, n_cap).
    Returns {class_idx: (train_indices, test_indices)}.
    Reuse across configs for truly paired comparisons.
    """
    rng = np.random.RandomState(seed)
    indices = {}

    for c in range(n_classes):
        train_idx = np.where(y_train == c)[0]
        test_idx = np.where(y_test == c)[0]
        n = min(len(train_idx), len(test_idx), n_cap)

        if n < 10:
            continue  # Skip classes with too few samples

        chosen_train = rng.choice(train_idx, size=n, replace=False)
        chosen_test = rng.choice(test_idx, size=n, replace=False)
        indices[c] = (chosen_train, chosen_test)

    return indices


def compute_mia_metrics(
    member_losses: np.ndarray, nonmember_losses: np.ndarray,
) -> Dict[str, float]:
    """AUC, TPR@1%FPR, TPR@5%FPR, advantage from loss distributions.

    Convention: label=1 for members, score = -loss (members have lower loss).
    """
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)),
                             np.zeros(len(nonmember_losses))])

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    # TPR at fixed FPR thresholds
    tpr_at_1fpr = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
    tpr_at_5fpr = float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1])

    # Advantage: max(TPR - FPR)
    advantage = float(np.max(tpr - fpr))

    return {
        "auc": auc,
        "tpr_at_1fpr": tpr_at_1fpr,
        "tpr_at_5fpr": tpr_at_5fpr,
        "advantage": advantage,
    }


def run_mia_attack(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    indices: Dict[int, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str],
    device: torch.device,
) -> Dict[str, dict]:
    """Run MIA with precomputed balanced indices.

    Returns {"per_class": {class_name: metrics}, "macro": metrics}.
    """
    # Compute all losses once
    train_losses = compute_per_sample_loss(model, X_train, y_train, device)
    test_losses = compute_per_sample_loss(model, X_test, y_test, device)

    per_class = {}
    metric_keys = ["auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage"]

    for c, (train_idx, test_idx) in sorted(indices.items()):
        member_losses = train_losses[train_idx]
        nonmember_losses = test_losses[test_idx]
        m = compute_mia_metrics(member_losses, nonmember_losses)
        m["n_samples"] = len(train_idx)
        per_class[class_names[c]] = m

    # Macro-average across classes
    macro = {}
    for k in metric_keys:
        vals = [per_class[cn][k] for cn in per_class]
        macro[k] = float(np.mean(vals))

    return {"per_class": per_class, "macro": macro}


# ── Shadow-model MIA ─────────────────────────────────────────


def compute_attack_features(
    model: nn.Module, X: np.ndarray, y: np.ndarray,
    device: torch.device, batch_size: int = 512,
) -> np.ndarray:
    """Per-sample attack feature vector: [loss, confidence, entropy, true_class_prob, margin].

    Returns array of shape (n_samples, 5).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_features = []
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            end = min(start + batch_size, len(X_t))
            logits = model(X_t[start:end])
            loss = criterion(logits, y_t[start:end]).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_batch = y_t[start:end].cpu().numpy()

            confidence = probs.max(axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            true_class_prob = probs[np.arange(len(labels_batch)), labels_batch]
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            batch_feats = np.column_stack([loss, confidence, entropy, true_class_prob, margin])
            all_features.append(batch_feats)

    return np.concatenate(all_features, axis=0)


def train_attack_classifiers(
    shadow_features: np.ndarray,
    shadow_labels: np.ndarray,
    shadow_classes: np.ndarray,
    n_classes: int,
) -> Dict[int, LogisticRegression]:
    """Train per-class logistic regression attack models on shadow data.

    Args:
        shadow_features: (n_samples, 5) attack feature vectors
        shadow_labels: (n_samples,) binary member/non-member labels
        shadow_classes: (n_samples,) true class indices
        n_classes: number of traffic classes

    Returns: {class_idx: fitted LogisticRegression}
    """
    classifiers = {}
    for c in range(n_classes):
        mask = shadow_classes == c
        if mask.sum() < 20:
            continue
        X_c = shadow_features[mask]
        y_c = shadow_labels[mask]
        if len(np.unique(y_c)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_c, y_c)
        classifiers[c] = clf
    return classifiers


def evaluate_shadow_mia(
    attack_clfs: Dict[int, LogisticRegression],
    target_features_member: np.ndarray,
    target_features_nonmember: np.ndarray,
    member_classes: np.ndarray,
    nonmember_classes: np.ndarray,
    class_names: List[str],
    n_cap: int = N_CAP,
    seed: int = 42,
) -> Dict[str, dict]:
    """Evaluate shadow-model attack classifiers on target model outputs.

    Returns same structure as run_mia_attack: {"per_class": {...}, "macro": {...}}.
    """
    rng = np.random.RandomState(seed)
    per_class = {}
    metric_keys = ["auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage"]

    for c, clf in sorted(attack_clfs.items()):
        mem_mask = member_classes == c
        nonmem_mask = nonmember_classes == c
        mem_idx = np.where(mem_mask)[0]
        nonmem_idx = np.where(nonmem_mask)[0]
        n = min(len(mem_idx), len(nonmem_idx), n_cap)
        if n < 10:
            continue

        chosen_mem = rng.choice(mem_idx, size=n, replace=False)
        chosen_nonmem = rng.choice(nonmem_idx, size=n, replace=False)

        X_eval = np.concatenate([
            target_features_member[chosen_mem],
            target_features_nonmember[chosen_nonmem],
        ])
        y_eval = np.concatenate([np.ones(n), np.zeros(n)])

        # Attack classifier's predicted probability of membership
        scores = clf.predict_proba(X_eval)[:, 1]
        auc = float(roc_auc_score(y_eval, scores))
        fpr, tpr, _ = roc_curve(y_eval, scores)
        tpr_at_1fpr = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
        tpr_at_5fpr = float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1])
        advantage = float(np.max(tpr - fpr))

        per_class[class_names[c]] = {
            "auc": auc,
            "tpr_at_1fpr": tpr_at_1fpr,
            "tpr_at_5fpr": tpr_at_5fpr,
            "advantage": advantage,
            "n_samples": n,
        }

    macro = {}
    for k in metric_keys:
        vals = [per_class[cn][k] for cn in per_class]
        macro[k] = float(np.mean(vals)) if vals else 0.0
    macro["n_classes_evaluated"] = len(per_class)

    return {"per_class": per_class, "macro": macro}
