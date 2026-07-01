#!/usr/bin/env python3
"""Generate Figure 2: Per-seed FS-mild minus DP-SGD deltas."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIG_DIR = REPO_ROOT / "figures"
BASE = REPO_ROOT / "results" / "baseline_16"
SEEDS = [42, 123, 456, 789, 1024]


def load_final_metrics(config, seed):
    path = BASE / f"seed_{seed}" / f"{config}.json"
    with path.open() as f:
        d = json.load(f)
    last = d["rounds"][-1]
    ha, hb = last["home_a"], last["home_b"]
    macro = (ha["macro_f1"] + hb["macro_f1"]) / 2
    all_f1 = list(ha["per_class_f1"].values()) + list(hb["per_class_f1"].values())
    worst = min(all_f1)
    return macro, worst


# Compute deltas
macro_deltas, worst_deltas = [], []
for s in SEEDS:
    fs_m, fs_w = load_final_metrics("fs_mild", s)
    dp_m, dp_w = load_final_metrics("dp_sgd", s)
    macro_deltas.append(fs_m - dp_m)
    worst_deltas.append(fs_w - dp_w)

macro_deltas = np.array(macro_deltas)
worst_deltas = np.array(worst_deltas)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=False)

seed_labels = [str(s) for s in SEEDS]
x = np.arange(len(SEEDS))

# Worst-group F1 delta
ax1.bar(x, worst_deltas, color="#4878CF", edgecolor="white", linewidth=0.5, width=0.6)
ax1.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax1.set_title("Worst-group F1", fontsize=8, fontweight="bold")
ax1.set_ylabel(r"$\Delta$ (FS-mild $-$ DP-SGD)", fontsize=7)
ax1.set_xticks(x)
ax1.set_xticklabels(seed_labels, fontsize=6)
ax1.set_xlabel("Seed", fontsize=7)
ax1.tick_params(axis="y", labelsize=6)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Annotate mean
mean_w = worst_deltas.mean()
ax1.axhline(mean_w, color="#D65F5F", linewidth=1, linestyle="--", alpha=0.7)
ax1.text(len(SEEDS) - 0.5, mean_w + 0.003, f"mean={mean_w:.3f}",
         fontsize=5.5, color="#D65F5F", ha="right")

# Macro-F1 delta
ax2.bar(x, macro_deltas, color="#6ACC65", edgecolor="white", linewidth=0.5, width=0.6)
ax2.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax2.set_title("Combined Macro-F1", fontsize=8, fontweight="bold")
ax2.set_ylabel(r"$\Delta$ (FS-mild $-$ DP-SGD)", fontsize=7)
ax2.set_xticks(x)
ax2.set_xticklabels(seed_labels, fontsize=6)
ax2.set_xlabel("Seed", fontsize=7)
ax2.tick_params(axis="y", labelsize=6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotate mean
mean_m = macro_deltas.mean()
ax2.axhline(mean_m, color="#D65F5F", linewidth=1, linestyle="--", alpha=0.7)
ax2.text(len(SEEDS) - 0.5, mean_m + 0.003, f"mean={mean_m:.3f}",
         fontsize=5.5, color="#D65F5F", ha="right")

plt.tight_layout()
pdf_out = FIG_DIR / "paired_deltas.pdf"
png_out = FIG_DIR / "paired_deltas.png"
plt.savefig(pdf_out, bbox_inches="tight", dpi=300)
plt.savefig(png_out, bbox_inches="tight", dpi=300)
print(f"Saved {pdf_out}")
print(f"Saved {png_out}")
print(f"Worst-group deltas: {worst_deltas} (all positive: {all(worst_deltas > 0)})")
print(f"Macro deltas: {macro_deltas} (all positive: {all(macro_deltas > 0)})")
