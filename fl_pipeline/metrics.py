"""Metrics: per-class F1, macro-F1, worst-group F1, paired 95% CI."""

from typing import Dict, List

import numpy as np
import scipy.stats
from sklearn.metrics import f1_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict:
    """Compute per-class F1, macro-F1, and worst-group F1."""
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    worst = float(min(per_class)) if len(per_class) > 0 else 0.0

    return {
        "macro_f1": float(macro),
        "worst_group_f1": worst,
        "per_class_f1": {
            name: float(f) for name, f in zip(class_names, per_class)
        },
    }


def paired_confidence_interval(
    values_a: List[float], values_b: List[float], confidence: float = 0.95
) -> Dict:
    """Paired t-test CI for the difference (A - B).

    Returns mean difference and confidence interval bounds.
    """
    diffs = np.array(values_a) - np.array(values_b)
    n = len(diffs)
    if n < 2:
        return {"mean_diff": float(np.mean(diffs)), "ci_low": float("nan"), "ci_high": float("nan")}
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return {
        "mean_diff": mean_diff,
        "ci_low": float(mean_diff - t_crit * se),
        "ci_high": float(mean_diff + t_crit * se),
    }
