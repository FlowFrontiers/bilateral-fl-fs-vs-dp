#!/usr/bin/env bash
# End-to-end reproduction script.
# Usage: bash scripts/reproduce.sh
#
# Prerequisites:
#   - Python 3.11+ installed
#   - git LFS installed and data/home_A.parquet + data/home_B.parquet fetched
#     (run `git lfs pull` if the hash check below fails)
#
# Total runtime estimate on a modern CPU (no GPU required):
#   Core (--core-only):
#   - Small model training:     ~4 hours  (5 configs × 5 seeds)
#   - Small equal-weight:       ~3 hours  (3 configs × 5 seeds)
#   - Small loss-based MIA:     ~5 hours  (retrains 3 configs × 5 seeds)
#   - Summary + figures:        ~1 minute
#   Extensions (full run):
#   - Small shadow-model MIA:   ~6 hours  (4 shadows × 3 configs × 5 seeds)
#   - Medium model training:    ~4 hours  (3 configs × 5 seeds)
#   - Medium loss-based MIA:    ~4 hours  (retrains 3 configs × 5 seeds)
#   - Medium shadow-model MIA:  ~7 hours  (4 shadows × 3 configs × 5 seeds)
#
# Core-only: ~8 hours; full (with shadow MIA + medium): ~25 hours
#
# To run only the core small-model results (skipping extensions):
#   bash scripts/reproduce.sh --core-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="42 123 456 789 1024"
FEATURE_SET="baseline_16"
MIA_CONFIGS="baseline_fl fs_mild dp_sgd"
N_SHADOWS=4
CORE_ONLY=false

if [[ "${1:-}" == "--core-only" ]]; then
    CORE_ONLY=true
    echo "Running core small-model reproduction only (no extensions)."
fi

echo "=== Step 1: Environment setup ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.lock.txt
else
    echo "  .venv exists, skipping creation."
fi
PYTHON=".venv/bin/python"

echo ""
echo "=== Step 2: Verify data files ==="
check_sha256() {
    file="$1"
    expected="$2"
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found. Run 'git lfs pull' or see data/README.md."
        exit 1
    fi
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: $file hash mismatch."
        echo "  expected: $expected"
        echo "  actual:   $actual"
        echo "If this is a Git LFS pointer file, run: git lfs pull"
        exit 1
    fi
}

check_sha256 data/home_A.parquet e34b01a1ac786e8decda51336c81d8163ace4cf5fbe01deea5612b727f510bcb
check_sha256 data/home_B.parquet b46b21c1d6b7b34ffd2028bff174f130940e09e35dd1c84ede8cdc4574698a43
echo "  Data files present and hashes match."

# ── Small model (primary results) ────────────────────────────

echo ""
echo "=== Step 3: Small model — size-proportional training ==="
$PYTHON -m fl_pipeline.run_experiment --feature-set $FEATURE_SET --seeds $SEEDS \
    --configs baseline_fl fs_mild fs_aggressive dp_sgd feddpa

echo ""
echo "=== Step 4: Small model — equal-weight ablation ==="
$PYTHON -m fl_pipeline.run_experiment --feature-set $FEATURE_SET --seeds $SEEDS \
    --equal-weight --configs $MIA_CONFIGS

echo ""
echo "=== Step 5: Small model — summaries ==="
$PYTHON -m fl_pipeline.analyze --feature-set $FEATURE_SET --save-summary
$PYTHON -m fl_pipeline.analyze --feature-set $FEATURE_SET --equal-weight --save-summary

echo ""
echo "=== Step 6: Small model — loss-based MIA ==="
$PYTHON -m fl_pipeline.run_mia --feature-set $FEATURE_SET --seeds $SEEDS \
    --configs $MIA_CONFIGS

echo ""
echo "=== Step 7: Extract paper numbers (small) ==="
$PYTHON scripts/extract_paper_numbers.py

if [ "$CORE_ONLY" = false ]; then

# ── Extensions (shadow MIA + medium model) ───────────────────

echo ""
echo "=== Step 8: Small model — shadow-model MIA ==="
$PYTHON -m fl_pipeline.run_shadow_mia --feature-set $FEATURE_SET --seeds $SEEDS \
    --configs $MIA_CONFIGS --n-shadows $N_SHADOWS

# ── Medium model ─────────────────────────────────────────────

echo ""
echo "=== Step 9: Medium model — training ==="
$PYTHON -m fl_pipeline.run_experiment --feature-set $FEATURE_SET --model-size medium \
    --seeds $SEEDS --configs $MIA_CONFIGS

echo ""
echo "=== Step 10: Medium model — summary ==="
$PYTHON -m fl_pipeline.analyze --feature-set ${FEATURE_SET}_medium --save-summary

echo ""
echo "=== Step 11: Medium model — loss-based MIA ==="
$PYTHON -m fl_pipeline.run_mia --feature-set $FEATURE_SET --model-size medium \
    --seeds $SEEDS --configs $MIA_CONFIGS

echo ""
echo "=== Step 12: Medium model — shadow-model MIA ==="
$PYTHON -m fl_pipeline.run_shadow_mia --feature-set $FEATURE_SET --model-size medium \
    --seeds $SEEDS --configs $MIA_CONFIGS --n-shadows $N_SHADOWS

echo ""
echo "=== Step 13: Extract paper numbers (medium) ==="
$PYTHON scripts/extract_paper_numbers.py --model-size medium

fi

# ── Figures ──────────────────────────────────────────────────

echo ""
echo "=== Generate figures ==="
$PYTHON scripts/fig1_class_distribution.py
$PYTHON scripts/fig2_paired_deltas.py
$PYTHON scripts/fig3_convergence.py

echo ""
echo "=== Done ==="
echo "Results:       results/"
echo "Paper numbers: results/paper_numbers.json"
if [ "$CORE_ONLY" = false ]; then
echo "               results/paper_numbers_baseline_16_medium.json"
fi
echo "Figures:       figures/*.pdf"
echo ""
