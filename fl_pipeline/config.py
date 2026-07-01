"""FL experiment configuration and constants."""

from dataclasses import dataclass, field
from typing import List
import os

# ── Feature sets ──────────────────────────────────────────────

BASELINE_16 = [
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_packets",
    "dst2src_packets",
    "src2dst_bytes",
    "dst2src_bytes",
    "bidirectional_ps_min",
    "bidirectional_ps_max",
    "bidirectional_ps_mean",
    "bidirectional_ps_stddev",
    "bidirectional_piat_min",
    "bidirectional_piat_max",
    "bidirectional_piat_mean",
    "bidirectional_piat_stddev",
    "protocol",
]

ALT_16 = [
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_packets",
    "dst2src_packets",
    "src2dst_bytes",
    "dst2src_bytes",
    "bidirectional_ps_min",
    "bidirectional_ps_max",
    "bidirectional_ps_mean",
    "bidirectional_ps_stddev",
    "src2dst_ps_min",
    "src2dst_ps_max",
    "src2dst_ps_mean",
    "src2dst_ps_stddev",
    "protocol",
]

# FS-mild: drop the 4 distinguishing features
FS_MILD_BASELINE = [f for f in BASELINE_16 if not f.startswith("bidirectional_piat_")]
FS_MILD_ALT = [f for f in ALT_16 if not f.startswith("src2dst_ps_")]

# FS-aggressive: keep duration, bidir pkt/byte, bidir PS, protocol → 8 features
FS_AGGRESSIVE = [
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "bidirectional_ps_min",
    "bidirectional_ps_max",
    "bidirectional_ps_mean",
    "bidirectional_ps_stddev",
    "protocol",
]

FEATURE_SETS = {
    "baseline_16": BASELINE_16,
    "alt_16": ALT_16,
    "fs_mild_baseline": FS_MILD_BASELINE,
    "fs_mild_alt": FS_MILD_ALT,
    "fs_aggressive": FS_AGGRESSIVE,
}

# ── Constants ─────────────────────────────────────────────────

CORE_CLASSES = sorted(["Collaborative", "Media", "Network", "SocialNetwork", "System", "Web"])
SEEDS = [42, 123, 456, 789, 1024]
CONFIG_NAMES = ["baseline_fl", "fs_mild", "fs_aggressive", "dp_sgd", "feddpa"]

# Model sizes: None = original default (computed from input_dim)
MODEL_SIZES = {
    "small": None,
    "medium": [128, 64],
}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HOME_A_PATH = os.path.join(DATA_DIR, "home_A.parquet")
HOME_B_PATH = os.path.join(DATA_DIR, "home_B.parquet")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def get_feature_cols(config_name: str, base_feature_set: str = "baseline_16") -> List[str]:
    """Return the feature column list for a given config name and base set."""
    if config_name in ("baseline_fl", "dp_sgd", "feddpa"):
        return FEATURE_SETS[base_feature_set]
    elif config_name == "fs_mild":
        prefix = base_feature_set.split("_")[0]  # "baseline" or "alt"
        return FEATURE_SETS[f"fs_mild_{prefix}"]
    elif config_name == "fs_aggressive":
        return FS_AGGRESSIVE
    else:
        raise ValueError(f"Unknown config: {config_name}")


@dataclass
class ExperimentConfig:
    """Configuration for a single FL experiment run."""
    # Hyperparameters
    num_rounds: int = 20
    local_epochs: int = 5
    batch_size: int = 256
    lr: float = 0.001
    test_ratio: float = 0.2
    # DP
    target_epsilon: float = 8.0
    max_grad_norm: float = 1.0
    # FedDPA
    fisher_threshold: float = 0.4
    fisher_n_samples: int = 5000
    lambda_reg: float = 0.05
    # FedAvg
    equal_weight: bool = False
    # Model architecture
    hidden_dims: List[int] = None  # None = original default [16, 16]
