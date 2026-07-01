# Bilateral FL: Feature Suppression vs Differential Privacy

Reproducibility artifact for:

> **Privacy-Aware Traffic Analytics for Residential Service Management: A Two-Home Federated Study**

## Overview

This repository contains the complete pipeline for a controlled comparison of feature suppression (data minimization) and differential privacy (DP-SGD) in a bilateral federated learning setting with naturally non-IID home network traffic.

**Five configurations** evaluated across five random seeds:

| Config | Features | Privacy mechanism |
|--------|----------|-------------------|
| Baseline FL | 16 | None |
| FS-mild | 12 | Data minimization (drop PIAT) |
| FS-aggressive | 8 | Data minimization (drop PIAT + directional counts) |
| DP-SGD | 16 | Per-sample gradient clipping + Gaussian noise (eps=8.0) |
| FedDPA | 16 | Fisher-based adaptive DP (eps=8.0) |

## Repository Structure

```
bilateral-fl-fs-vs-dp/
├── LICENSE                 # MIT
├── CITATION.cff            # Citation metadata
├── repro_manifest.json     # Seeds, data checksums, expected outputs
├── requirements.lock.txt   # Pinned Python dependencies
├── .python-version         # Python 3.11.14 (pyenv)
├── data/                   # IP-deidentified flow parquet files (git LFS)
│   ├── README.md           # Schema, preprocessing, release notes
│   ├── home_A.parquet      # ~111 MB (git LFS)
│   └── home_B.parquet      #  ~63 MB (git LFS)
├── fl_pipeline/                  # Training and evaluation pipeline
│   ├── config.py           # Feature sets, hyperparameters, paths
│   ├── data.py             # Data loading, filtering, federated scaling
│   ├── model.py            # Configurable MLP (small: 646 params, medium: 10,822 params)
│   ├── train.py            # FedAvg training loop
│   ├── dp.py               # DP-SGD via Opacus
│   ├── feddpa.py           # FedDPA (Fisher-based adaptive DP)
│   ├── metrics.py          # Macro-F1, worst-group F1, per-class metrics
│   ├── mia.py              # MIA attacks (loss-based + shadow-model)
│   ├── run_experiment.py        # Main experiment runner (5 configs x 5 seeds)
│   ├── run_mia.py          # Loss-based MIA runner
│   ├── run_shadow_mia.py   # Shadow-model MIA runner
│   └── analyze.py          # Result aggregation and summary
├── results/                # Experiment outputs
│   ├── audit_log.json      # Preprocessing flow counts
│   ├── paper_numbers.json  # Canonical numbers for paper tables
│   ├── temporal_split_check*.json # Temporal holdout sanity checks
│   ├── baseline_16/        # Small model: per-seed results, summaries, MIA, shadow MIA
│   └── baseline_16_medium/ # Medium model: per-seed results, summaries, MIA, shadow MIA
├── figures/                # Generated figures (PDF + PNG)
└── scripts/
    ├── reproduce.sh        # End-to-end reproduction (setup → results → figures)
    ├── temporal_split_check.py # Earlier-80%/later-20% sanity check
    ├── extract_paper_numbers.py  # Extract canonical numbers from results
    ├── fig1_class_distribution.py
    ├── fig2_paired_deltas.py
    └── fig3_convergence.py
```

## Quick Start

### One-command reproduction

```bash
bash scripts/reproduce.sh             # full reproduction (core + extensions)
bash scripts/reproduce.sh --core-only  # small model + loss MIA only (skip shadow MIA + medium)
```

This creates a venv, verifies data, runs all experiments, and generates figures.
**Total time**: core-only ~8 hours; full ~25 hours on a modern CPU (estimated with `--n-shadows 4`, no GPU required).

### Step-by-step

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

# 2. Data ships with the repo via git LFS (run `git lfs pull` if needed; see data/README.md)
shasum -a 256 data/*.parquet   # verify against repro_manifest.json

# 3. Run experiments
python -m fl_pipeline.run_experiment --feature-set baseline_16 --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild fs_aggressive dp_sgd feddpa
python -m fl_pipeline.run_experiment --feature-set baseline_16 --seeds 42 123 456 789 1024 --equal-weight --configs baseline_fl fs_mild dp_sgd
python -m fl_pipeline.analyze --feature-set baseline_16 --save-summary
python -m fl_pipeline.analyze --feature-set baseline_16 --equal-weight --save-summary
python -m fl_pipeline.run_mia --feature-set baseline_16 --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild dp_sgd
python -m fl_pipeline.run_shadow_mia --feature-set baseline_16 --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild dp_sgd --n-shadows 4

# 3b. Medium model (optional robustness check, --model-size medium)
python -m fl_pipeline.run_experiment --feature-set baseline_16 --model-size medium --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild dp_sgd
python -m fl_pipeline.analyze --feature-set baseline_16_medium --save-summary
python -m fl_pipeline.run_mia --feature-set baseline_16 --model-size medium --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild dp_sgd
python -m fl_pipeline.run_shadow_mia --feature-set baseline_16 --model-size medium --seeds 42 123 456 789 1024 --configs baseline_fl fs_mild dp_sgd --n-shadows 4

# 4. Generate figures and paper numbers
python scripts/extract_paper_numbers.py   # → results/paper_numbers.json (small)
python scripts/extract_paper_numbers.py --model-size medium  # → results/paper_numbers_baseline_16_medium.json
python scripts/fig1_class_distribution.py # → figures/class_distribution.pdf
python scripts/fig2_paired_deltas.py       # → figures/paired_deltas.pdf
python scripts/fig3_convergence.py        # → figures/convergence.pdf

# Optional temporal holdout sanity checks
python scripts/temporal_split_check.py --seeds 42 123 456 789 1024
python scripts/temporal_split_check.py --model-size medium --seeds 42 123 456 789 1024
```

## Requirements

- Python 3.11+ (see `.python-version`)
- git LFS (to fetch the IP-deidentified dataset in `data/`)
- PyTorch 2.10+, Opacus 1.5+, scikit-learn 1.8+
- No GPU required (small model: 646 params; medium model: 10,822 params)
- See `requirements.lock.txt` for exact pinned versions

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
