#!/usr/bin/env python3
"""Run loss-based MIA evaluation on trained FL models."""

import argparse
import json
import os
import sys

import numpy as np

from .config import (
    SEEDS, HOME_A_PATH, HOME_B_PATH, CORE_CLASSES, RESULTS_DIR,
    MODEL_SIZES, ExperimentConfig, get_feature_cols,
)
from .data import prepare_federated_data
from .train import run_standard_fl, run_dp_sgd_fl
from .mia import sample_balanced_indices, run_mia_attack, N_CAP
from .metrics import paired_confidence_interval

HOME_KEYS = ["home_a", "home_b"]


def train_model(data, config, seed, config_name):
    """Train and return (results, model) for the given config."""
    if config_name == "dp_sgd":
        return run_dp_sgd_fl(data, config, seed, return_model=True)
    else:
        return run_standard_fl(data, config, seed,
                               config_name=config_name, return_model=True)


def run_seed(seed, configs, feature_set, config_obj):
    """Run MIA for all configs on one seed with shared indices."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # 1. Prepare data once per seed
    feature_cols = get_feature_cols("baseline_fl", feature_set)
    data = prepare_federated_data(
        HOME_A_PATH, HOME_B_PATH, feature_cols, CORE_CLASSES,
        seed=seed, batch_size=config_obj.batch_size,
    )

    # 2. Precompute balanced indices per home (reused across configs)
    indices = {}
    for hk in HOME_KEYS:
        indices[hk] = sample_balanced_indices(
            data[hk]["y_train"], data[hk]["y_test"],
            data["num_classes"], seed,
        )
        n_total = sum(len(ti) for ti, _ in indices[hk].values())
        print(f"  {hk}: {len(indices[hk])} classes, "
              f"{n_total} members + {n_total} non-members")

    # 3. Train each config and evaluate MIA
    seed_results = {}
    device_for_mia = None  # Will use model's device

    for config_name in configs:
        print(f"\n  --- {config_name} ---")

        # Need correct feature set per config
        feat_cols = get_feature_cols(config_name, feature_set)
        if feat_cols != feature_cols:
            # FS configs use fewer features — re-prepare data
            data_cfg = prepare_federated_data(
                HOME_A_PATH, HOME_B_PATH, feat_cols, CORE_CLASSES,
                seed=seed, batch_size=config_obj.batch_size,
            )
            # Recompute indices for this feature set (same seed → same splits)
            indices_cfg = {}
            for hk in HOME_KEYS:
                indices_cfg[hk] = sample_balanced_indices(
                    data_cfg[hk]["y_train"], data_cfg[hk]["y_test"],
                    data_cfg["num_classes"], seed,
                )
        else:
            data_cfg = data
            indices_cfg = indices

        results, model = train_model(data_cfg, config_obj, seed, config_name)

        import torch
        device = next(model.parameters()).device

        mia_result = {"config": config_name, "seed": seed}
        for hk in HOME_KEYS:
            mia = run_mia_attack(
                model,
                data_cfg[hk]["X_train"], data_cfg[hk]["y_train"],
                data_cfg[hk]["X_test"], data_cfg[hk]["y_test"],
                indices_cfg[hk], data_cfg["class_names"], device,
            )
            mia_result[hk] = mia
            print(f"  {hk} macro: AUC={mia['macro']['auc']:.4f}  "
                  f"TPR@1%={mia['macro']['tpr_at_1fpr']:.4f}  "
                  f"TPR@5%={mia['macro']['tpr_at_5fpr']:.4f}")

        # Equal-home average
        eq = {}
        for k in ["auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage"]:
            eq[k] = float((mia_result["home_a"]["macro"][k]
                           + mia_result["home_b"]["macro"][k]) / 2)
        mia_result["equal_avg"] = eq

        seed_results[config_name] = mia_result

    return seed_results


def aggregate_results(all_seed_results, configs):
    """Aggregate across seeds into summary with paired CIs."""
    summary = {"n_cap": N_CAP, "per_config": {}, "paired_ci": {}}
    metric_keys = ["auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage"]

    # Per-config aggregation
    for config_name in configs:
        cfg_summary = {}
        for scope in HOME_KEYS + ["equal_avg"]:
            vals = {}
            for k in metric_keys:
                if scope in HOME_KEYS:
                    seed_vals = [r[config_name][scope]["macro"][k]
                                 for r in all_seed_results]
                else:
                    seed_vals = [r[config_name][scope][k]
                                 for r in all_seed_results]
                vals[f"{k}_mean"] = float(np.mean(seed_vals))
                vals[f"{k}_std"] = float(np.std(seed_vals))
            cfg_summary[scope] = vals
        summary["per_config"][config_name] = cfg_summary

    # Paired CIs: fs_mild vs baseline_fl, fs_mild vs dp_sgd
    if "fs_mild" in configs:
        ci_metrics = ["auc", "tpr_at_1fpr", "tpr_at_5fpr"]
        ci_short = {"auc": "auc", "tpr_at_1fpr": "tpr1", "tpr_at_5fpr": "tpr5"}

        for baseline_name in ["baseline_fl", "dp_sgd"]:
            if baseline_name not in configs:
                continue
            short_base = "baseline" if baseline_name == "baseline_fl" else "dp_sgd"
            for mk in ci_metrics:
                # Use equal_avg for paired comparison
                fs_vals = [r["fs_mild"]["equal_avg"][mk]
                           for r in all_seed_results]
                base_vals = [r[baseline_name]["equal_avg"][mk]
                             for r in all_seed_results]
                ci = paired_confidence_interval(fs_vals, base_vals)
                key = f"{ci_short[mk]}_fs_mild_minus_{short_base}"
                summary["paired_ci"][key] = ci

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run MIA evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--configs", nargs="+",
                        default=["baseline_fl", "fs_mild", "dp_sgd"])
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--model-size", default="small",
                        choices=list(MODEL_SIZES.keys()),
                        help="Model architecture size (default: small)")
    args = parser.parse_args()

    hidden_dims = MODEL_SIZES[args.model_size]
    config_obj = ExperimentConfig(hidden_dims=hidden_dims)
    subdir = args.feature_set if args.model_size == "small" else f"{args.feature_set}_{args.model_size}"
    out_dir = os.path.join(RESULTS_DIR, subdir, "mia")
    os.makedirs(out_dir, exist_ok=True)

    all_seed_results = []

    for seed in args.seeds:
        seed_results = run_seed(seed, args.configs, args.feature_set, config_obj)

        # Save per-seed, per-config JSONs
        for config_name, mia_result in seed_results.items():
            fname = os.path.join(out_dir, f"seed_{seed}_{config_name}.json")
            with open(fname, "w") as f:
                json.dump(mia_result, f, indent=2, default=str)
            print(f"  Saved: {fname}")

        all_seed_results.append(seed_results)

    # Aggregate
    summary = aggregate_results(all_seed_results, args.configs)
    summary_path = os.path.join(out_dir, "mia_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  MIA SUMMARY ({len(args.seeds)} seeds)")
    print(f"{'='*60}")
    for cn in args.configs:
        eq = summary["per_config"][cn]["equal_avg"]
        print(f"\n  {cn}:")
        print(f"    AUC:      {eq['auc_mean']:.4f} +/- {eq['auc_std']:.4f}")
        print(f"    TPR@1%:   {eq['tpr_at_1fpr_mean']:.4f} +/- {eq['tpr_at_1fpr_std']:.4f}")
        print(f"    TPR@5%:   {eq['tpr_at_5fpr_mean']:.4f} +/- {eq['tpr_at_5fpr_std']:.4f}")
        print(f"    Adv:      {eq['advantage_mean']:.4f} +/- {eq['advantage_std']:.4f}")

    if summary["paired_ci"]:
        print(f"\n  Paired CIs (fs_mild minus baseline):")
        for key, ci in summary["paired_ci"].items():
            print(f"    {key}: {ci['mean_diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")

    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
