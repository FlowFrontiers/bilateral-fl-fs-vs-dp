#!/usr/bin/env python3
"""Generate Figure 3: Round-by-round convergence curves.

Top panel: Combined macro-F1 (mean +/- std across 5 seeds).
Bottom panel: Home B System F1 (minority class collapse under DP-SGD).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIG_DIR = REPO_ROOT / "figures"
BASE = REPO_ROOT / "results" / "baseline_16"
SEEDS = [42, 123, 456, 789, 1024]
CONFIGS = {
    "baseline_fl": ("Baseline FL", "#2ca02c", "-"),
    "fs_mild": ("FS-mild", "#1f77b4", "--"),
    "dp_sgd": ("DP-SGD", "#d62728", "-."),
}
ROUNDS = 20


def load_per_round(config, metric_fn):
    """Load per-round metric values for all seeds. Returns (n_seeds, n_rounds) array."""
    all_seeds = []
    for s in SEEDS:
        d = json.loads((BASE / f"seed_{s}/{config}.json").read_text())
        vals = [metric_fn(r) for r in d["rounds"]]
        all_seeds.append(vals)
    return np.array(all_seeds)


def combined_macro(r):
    return (r["home_a"]["macro_f1"] + r["home_b"]["macro_f1"]) / 2


def home_b_system(r):
    return r["home_b"]["per_class_f1"]["System"]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
x = np.arange(1, ROUNDS + 1)

# Top panel: Combined macro-F1
for cfg, (label, color, ls) in CONFIGS.items():
    arr = load_per_round(cfg, combined_macro)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax1.plot(x, mean, ls, color=color, label=label, linewidth=1.2)
    ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

ax1.set_ylabel("Combined Macro-F1", fontsize=7.5)
ax1.legend(fontsize=6.5, loc="lower right")
ax1.set_ylim(0.45, 0.72)
ax1.tick_params(labelsize=7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Bottom panel: Home B System F1
for cfg, (label, color, ls) in CONFIGS.items():
    arr = load_per_round(cfg, home_b_system)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax2.plot(x, mean, ls, color=color, label=label, linewidth=1.2)
    ax2.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

ax2.set_ylabel("Home B System F1", fontsize=7.5)
ax2.set_xlabel("FL Round", fontsize=7.5)
ax2.set_ylim(-0.02, 0.85)
ax2.tick_params(labelsize=7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(h_pad=0.4)
for ext in ["pdf", "png"]:
    out = FIG_DIR / f"convergence.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'convergence.pdf'}")
