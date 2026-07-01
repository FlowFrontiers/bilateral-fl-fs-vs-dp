#!/usr/bin/env python3
"""Generate Figure 1: Class distribution Home A vs Home B."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIG_DIR = REPO_ROOT / "figures"
sys.path.insert(0, str(REPO_ROOT))

from fl_pipeline.config import BASELINE_16, CORE_CLASSES, HOME_A_PATH, HOME_B_PATH  # noqa: E402
from fl_pipeline.data import load_and_filter  # noqa: E402

# Canonical counts from filtered data pipeline
_, audit_a = load_and_filter(HOME_A_PATH, BASELINE_16, CORE_CLASSES)
_, audit_b = load_and_filter(HOME_B_PATH, BASELINE_16, CORE_CLASSES)
counts_a = audit_a["class_counts_final"]
counts_b = audit_b["class_counts_final"]

classes = ["Network", "Web", "System", "Media", "Collaborative", "SocialNetwork"]
home_a = [counts_a[c] for c in classes]
home_b = [counts_b[c] for c in classes]

# Compute percentages
total_a, total_b = sum(home_a), sum(home_b)
pct_a = [100 * v / total_a for v in home_a]
pct_b = [100 * v / total_b for v in home_b]

fig, ax = plt.subplots(figsize=(3.5, 2.4))

x = np.arange(len(classes))
w = 0.35

bars_a = ax.bar(x - w / 2, pct_a, w, label=f"Home A ({sum(home_a)/1e3:.0f}K)", color="#4878CF", edgecolor="white", linewidth=0.5)
bars_b = ax.bar(x + w / 2, pct_b, w, label=f"Home B ({sum(home_b)/1e3:.0f}K)", color="#D65F5F", edgecolor="white", linewidth=0.5)

ax.set_ylabel("Share (%)", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=7, rotation=25, ha="right")
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0, 62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate the key skews
for i, (a, b) in enumerate(zip(pct_a, pct_b)):
    if abs(a - b) > 3:
        ratio = max(a, b) / min(a, b)
        ax.annotate(f"{ratio:.1f}x", xy=(x[i], max(a, b) + 1.5),
                    fontsize=5.5, ha="center", color="gray")

plt.tight_layout()
pdf_out = FIG_DIR / "class_distribution.pdf"
png_out = FIG_DIR / "class_distribution.png"
plt.savefig(pdf_out, bbox_inches="tight", dpi=300)
plt.savefig(png_out, bbox_inches="tight", dpi=300)
print(f"Saved {pdf_out}")
print(f"Saved {png_out}")
