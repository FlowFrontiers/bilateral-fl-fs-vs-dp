#!/usr/bin/env python3
"""CLI entry point: iterate seeds x configs, save JSON results."""

import argparse
import json
import os
import sys
import time

from .config import (
    ExperimentConfig, SEEDS, CONFIG_NAMES, CORE_CLASSES,
    HOME_A_PATH, HOME_B_PATH, RESULTS_DIR, MODEL_SIZES,
    get_feature_cols,
)
from .data import prepare_federated_data
from .train import run_standard_fl, run_dp_sgd_fl, run_feddpa_fl


def run_single(
    config_name: str,
    seed: int,
    feature_set: str,
    equal_weight: bool,
    cfg: ExperimentConfig,
) -> dict:
    """Run one config x seed combination."""
    feature_cols = get_feature_cols(config_name, feature_set)
    print(f"\n{'='*60}")
    print(f"Config: {config_name} | Seed: {seed} | Features: {len(feature_cols)} "
          f"| Equal weight: {equal_weight}")
    print(f"{'='*60}")

    data = prepare_federated_data(
        HOME_A_PATH, HOME_B_PATH, feature_cols, CORE_CLASSES,
        seed=seed, test_ratio=cfg.test_ratio, batch_size=cfg.batch_size,
        equal_weight=equal_weight,
    )

    # Save/refresh audit log every run (ensures it matches current data)
    audit_path = os.path.join(RESULTS_DIR, "audit_log.json")
    with open(audit_path, "w") as f:
        json.dump(data["audit"], f, indent=2)

    print(f"  Home A: {data['home_a']['n_train']} train / {data['home_a']['n_test']} test")
    print(f"  Home B: {data['home_b']['n_train']} train / {data['home_b']['n_test']} test")
    print(f"  Weights: A={data['weights']['home_a']:.3f}, B={data['weights']['home_b']:.3f}")

    t0 = time.time()

    if config_name in ("baseline_fl", "fs_mild", "fs_aggressive"):
        results = run_standard_fl(data, cfg, seed, config_name)
    elif config_name == "dp_sgd":
        results = run_dp_sgd_fl(data, cfg, seed)
    elif config_name == "feddpa":
        results = run_feddpa_fl(data, cfg, seed)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    results["total_time_s"] = time.time() - t0
    results["feature_cols"] = feature_cols
    results["weights"] = data["weights"]
    results["audit"] = data["audit"]

    return results


def save_results(results: dict, feature_set: str, seed: int,
                 config_name: str, equal_weight: bool,
                 model_size: str = "small"):
    """Save results JSON to the structured directory."""
    subdir = feature_set if model_size == "small" else f"{feature_set}_{model_size}"
    if equal_weight:
        seed_dir = os.path.join(RESULTS_DIR, subdir, f"equal_weight_seed_{seed}")
    else:
        seed_dir = os.path.join(RESULTS_DIR, subdir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    path = os.path.join(seed_dir, f"{config_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run FL experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to run (default: all 5)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to run (default: all 5)")
    parser.add_argument("--feature-set", default="baseline_16",
                        choices=["baseline_16", "alt_16"])
    parser.add_argument("--equal-weight", action="store_true",
                        help="Use equal 0.5/0.5 FedAvg weights")
    parser.add_argument("--model-size", default="small",
                        choices=list(MODEL_SIZES.keys()),
                        help="Model architecture size (default: small)")
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    configs = args.configs or CONFIG_NAMES
    hidden_dims = MODEL_SIZES[args.model_size]
    cfg = ExperimentConfig(equal_weight=args.equal_weight, hidden_dims=hidden_dims)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_runs = len(seeds) * len(configs)
    run_idx = 0

    for seed in seeds:
        for config_name in configs:
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total_runs}")
            results = run_single(
                config_name, seed, args.feature_set, args.equal_weight, cfg
            )
            save_results(results, args.feature_set, seed,
                         config_name, args.equal_weight, args.model_size)

    print(f"\nAll done. Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
