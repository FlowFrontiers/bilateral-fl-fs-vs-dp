#!/usr/bin/env python3
"""Run shadow-model MIA evaluation on trained FL models.

For each seed × config:
  1. Train target model on full data
  2. Train N shadow FL models on random 50% subsets (preserving bilateral structure)
  3. Collect attack features from shadow models (known members/non-members)
  4. Train per-home, per-class LR attack classifiers on shadow data
  5. Evaluate attack classifiers on target model's train (members) vs test (non-members)
"""

import argparse
import json
import os
import time

import numpy as np

from .config import (
    SEEDS, HOME_A_PATH, HOME_B_PATH, CORE_CLASSES, RESULTS_DIR,
    MODEL_SIZES, ExperimentConfig, get_feature_cols,
)
from .data import prepare_federated_data
from .train import run_standard_fl, run_dp_sgd_fl
from .mia import (
    compute_attack_features, train_attack_classifiers,
    evaluate_shadow_mia, N_CAP,
)
from .metrics import paired_confidence_interval

HOME_KEYS = ["home_a", "home_b"]


def train_model(data, config, seed, config_name):
    """Train and return (results, model) for the given config."""
    if config_name == "dp_sgd":
        return run_dp_sgd_fl(data, config, seed, return_model=True)
    else:
        return run_standard_fl(data, config, seed,
                               config_name=config_name, return_model=True)


def make_shadow_data(data, home_key, shadow_ratio, shadow_seed):
    """Split a home's training data into shadow-train and shadow-out.

    Returns (X_shadow_train, y_shadow_train, X_shadow_out, y_shadow_out).
    """
    rng = np.random.RandomState(shadow_seed)
    X_train = data[home_key]["X_train"]
    y_train = data[home_key]["y_train"]
    n = len(X_train)
    n_shadow = int(n * shadow_ratio)

    perm = rng.permutation(n)
    shadow_idx = perm[:n_shadow]
    out_idx = perm[n_shadow:]

    return X_train[shadow_idx], y_train[shadow_idx], X_train[out_idx], y_train[out_idx]


def train_shadow_fl(data, shadow_splits, config, seed, config_name):
    """Train one shadow FL model using shadow subsets of training data.

    Replaces each home's train data/loader with the shadow subset,
    keeping test data and all other structure intact.
    """
    import copy
    import torch
    from .data import _make_loader

    shadow_data = copy.copy(data)
    for hk in HOME_KEYS:
        X_sh, y_sh = shadow_splits[hk]
        shadow_data[hk] = dict(data[hk])
        shadow_data[hk]["X_train"] = X_sh
        shadow_data[hk]["y_train"] = y_sh
        shadow_data[hk]["n_train"] = len(X_sh)
        shadow_data[hk]["train_loader"] = _make_loader(
            X_sh, y_sh, config.batch_size, shuffle=True, seed=seed,
        )

    # Recompute weights for shadow sizes
    n_a = shadow_data["home_a"]["n_train"]
    n_b = shadow_data["home_b"]["n_train"]
    n_total = n_a + n_b
    shadow_data["weights"] = {
        "home_a": n_a / n_total,
        "home_b": n_b / n_total,
    }

    _, model = train_model(shadow_data, config, seed, config_name)
    return model


def run_seed(seed, configs, feature_set, config_obj, n_shadows, shadow_ratio):
    """Run shadow-model MIA for all configs on one seed."""
    print(f"\n{'='*60}")
    print(f"  SHADOW MIA — SEED {seed}")
    print(f"{'='*60}")

    import torch
    device = torch.device("cpu")
    seed_results = {}

    for config_name in configs:
        print(f"\n  --- {config_name} ---")

        # Prepare data for this config
        feat_cols = get_feature_cols(config_name, feature_set)
        data = prepare_federated_data(
            HOME_A_PATH, HOME_B_PATH, feat_cols, CORE_CLASSES,
            seed=seed, batch_size=config_obj.batch_size,
        )

        # 1. Train target model on full data
        print(f"  Training target model...")
        t0 = time.time()
        _, target_model = train_model(data, config_obj, seed, config_name)
        print(f"  Target trained in {time.time()-t0:.0f}s")

        # 2. Compute target attack features (for evaluation in step 5)
        target_features = {}
        for hk in HOME_KEYS:
            target_features[hk] = {
                "member": compute_attack_features(
                    target_model, data[hk]["X_train"], data[hk]["y_train"], device),
                "nonmember": compute_attack_features(
                    target_model, data[hk]["X_test"], data[hk]["y_test"], device),
                "member_classes": data[hk]["y_train"],
                "nonmember_classes": data[hk]["y_test"],
            }

        # 3. Train shadow models and collect attack features
        # Per-home, per-class: accumulate shadow features separately
        shadow_features = {hk: {"feats": [], "labels": [], "classes": []}
                           for hk in HOME_KEYS}

        for s_idx in range(n_shadows):
            shadow_seed = seed * 1000 + s_idx
            print(f"  Shadow {s_idx+1}/{n_shadows}...", end=" ")
            t0 = time.time()

            # Split each home's train data
            shadow_splits = {}
            shadow_out = {}
            home_offsets = {"home_a": 0, "home_b": 7919}  # prime offset to avoid collisions
            for hk in HOME_KEYS:
                X_sh, y_sh, X_out, y_out = make_shadow_data(
                    data, hk, shadow_ratio, shadow_seed + home_offsets[hk])
                shadow_splits[hk] = (X_sh, y_sh)
                shadow_out[hk] = (X_out, y_out)

            # Train shadow FL model
            shadow_model = train_shadow_fl(
                data, shadow_splits, config_obj, shadow_seed, config_name)

            # Collect attack features per home
            for hk in HOME_KEYS:
                X_sh, y_sh = shadow_splits[hk]
                X_out, y_out = shadow_out[hk]

                feats_member = compute_attack_features(
                    shadow_model, X_sh, y_sh, device)
                feats_nonmember = compute_attack_features(
                    shadow_model, X_out, y_out, device)

                shadow_features[hk]["feats"].append(feats_member)
                shadow_features[hk]["feats"].append(feats_nonmember)
                shadow_features[hk]["labels"].append(np.ones(len(feats_member)))
                shadow_features[hk]["labels"].append(np.zeros(len(feats_nonmember)))
                shadow_features[hk]["classes"].append(y_sh)
                shadow_features[hk]["classes"].append(y_out)

            print(f"({time.time()-t0:.0f}s)")

        # 4. Train per-home, per-class attack classifiers
        attack_clfs = {}
        for hk in HOME_KEYS:
            all_feats = np.concatenate(shadow_features[hk]["feats"])
            all_labels = np.concatenate(shadow_features[hk]["labels"])
            all_classes = np.concatenate(shadow_features[hk]["classes"])
            attack_clfs[hk] = train_attack_classifiers(
                all_feats, all_labels, all_classes, data["num_classes"])
            n_clfs = len(attack_clfs[hk])
            print(f"  {hk}: {n_clfs} per-class attack classifiers trained")

        # 5. Evaluate on target model
        mia_result = {
            "attack": "shadow_model",
            "n_shadows": n_shadows,
            "shadow_ratio": shadow_ratio,
            "config": config_name,
            "seed": seed,
        }

        for hk in HOME_KEYS:
            mia = evaluate_shadow_mia(
                attack_clfs[hk],
                target_features[hk]["member"],
                target_features[hk]["nonmember"],
                target_features[hk]["member_classes"],
                target_features[hk]["nonmember_classes"],
                data["class_names"],
                n_cap=N_CAP,
                seed=seed,
            )
            mia_result[hk] = mia
            n_eval = mia['macro']['n_classes_evaluated']
            print(f"  {hk} macro ({n_eval} classes): AUC={mia['macro']['auc']:.4f}  "
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
    summary = {"attack": "shadow_model", "per_config": {}, "paired_ci": {}}
    metric_keys = ["auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage"]

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
                fs_vals = [r["fs_mild"]["equal_avg"][mk]
                           for r in all_seed_results]
                base_vals = [r[baseline_name]["equal_avg"][mk]
                             for r in all_seed_results]
                ci = paired_confidence_interval(fs_vals, base_vals)
                key = f"{ci_short[mk]}_fs_mild_minus_{short_base}"
                summary["paired_ci"][key] = ci

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run shadow-model MIA evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--configs", nargs="+",
                        default=["baseline_fl", "fs_mild", "dp_sgd"])
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--model-size", default="small",
                        choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--n-shadows", type=int, default=4,
                        help="Number of shadow models (default: 4)")
    parser.add_argument("--shadow-ratio", type=float, default=0.5,
                        help="Fraction of train data per shadow (default: 0.5)")
    args = parser.parse_args()

    if not (0 < args.shadow_ratio < 1):
        parser.error("--shadow-ratio must be between 0 and 1 (exclusive)")
    if args.n_shadows < 1:
        parser.error("--n-shadows must be >= 1")

    hidden_dims = MODEL_SIZES[args.model_size]
    config_obj = ExperimentConfig(hidden_dims=hidden_dims)
    subdir = args.feature_set if args.model_size == "small" else f"{args.feature_set}_{args.model_size}"
    out_dir = os.path.join(RESULTS_DIR, subdir, "shadow_mia")
    os.makedirs(out_dir, exist_ok=True)

    all_seed_results = []

    for seed in args.seeds:
        seed_results = run_seed(
            seed, args.configs, args.feature_set, config_obj,
            args.n_shadows, args.shadow_ratio,
        )

        # Save per-seed, per-config JSONs
        for config_name, mia_result in seed_results.items():
            fname = os.path.join(out_dir, f"seed_{seed}_{config_name}.json")
            with open(fname, "w") as f:
                json.dump(mia_result, f, indent=2, default=str)
            print(f"  Saved: {fname}")

        all_seed_results.append(seed_results)

    # Aggregate
    summary = aggregate_results(all_seed_results, args.configs)
    summary["n_shadows"] = args.n_shadows
    summary["shadow_ratio"] = args.shadow_ratio
    summary_path = os.path.join(out_dir, "shadow_mia_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SHADOW MIA SUMMARY ({len(args.seeds)} seeds, {args.n_shadows} shadows)")
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
