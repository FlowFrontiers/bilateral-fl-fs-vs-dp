#!/usr/bin/env python3
"""Load results, evaluate pre-registered evaluation criteria, print summary."""

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

from .config import SEEDS, RESULTS_DIR
from .metrics import paired_confidence_interval


def load_results(feature_set: str, equal_weight: bool = False) -> Dict[str, Dict[int, dict]]:
    """Load all result JSONs. Returns {config_name: {seed: results}}."""
    base = os.path.join(RESULTS_DIR, feature_set)
    all_results = {}

    for entry in sorted(os.listdir(base)):
        entry_path = os.path.join(base, entry)
        if not os.path.isdir(entry_path):
            continue

        is_equal = entry.startswith("equal_weight_")
        if is_equal != equal_weight:
            continue

        seed_str = entry.replace("equal_weight_seed_", "").replace("seed_", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        for fname in sorted(os.listdir(entry_path)):
            if not fname.endswith(".json"):
                continue
            config_name = fname.replace(".json", "")
            with open(os.path.join(entry_path, fname)) as f:
                data = json.load(f)
            all_results.setdefault(config_name, {})[seed] = data

    return all_results


def extract_final_metrics(results: dict) -> dict:
    """Extract metrics from the last round."""
    last = results["rounds"][-1]
    return {
        "home_a_macro_f1": last["home_a"]["macro_f1"],
        "home_b_macro_f1": last["home_b"]["macro_f1"],
        "home_a_worst_f1": last["home_a"]["worst_group_f1"],
        "home_b_worst_f1": last["home_b"]["worst_group_f1"],
        "home_a_per_class": last["home_a"]["per_class_f1"],
        "home_b_per_class": last["home_b"]["per_class_f1"],
    }


def combined_worst_group_f1(metrics: dict) -> float:
    """Worst-group F1 across both homes."""
    all_f1s = list(metrics["home_a_per_class"].values()) + list(metrics["home_b_per_class"].values())
    return min(all_f1s) if all_f1s else 0.0


def combined_macro_f1(metrics: dict) -> float:
    """Average of both homes' macro-F1."""
    return (metrics["home_a_macro_f1"] + metrics["home_b_macro_f1"]) / 2


def evaluate_criteria(all_results: Dict[str, Dict[int, dict]]) -> dict:
    """Evaluate the pre-registered evaluation criteria.

    Primary comparison: FS-mild vs DP-SGD (required).
    FedDPA is optional — included if available, reported separately if not.

    Pass: FS-mild beats DP-SGD (and FedDPA if present) on worst-group F1
          in >=4/5 seeds, AND paired 95% CI for macro-F1 delta excludes 0.
    Marginal: one condition met.
    Fail: neither.
    """
    required = {"fs_mild", "dp_sgd"}
    if not required.issubset(all_results.keys()):
        missing = required - set(all_results.keys())
        return {"decision": "INCOMPLETE", "missing": list(missing)}

    has_feddpa = "feddpa" in all_results

    common = set(all_results["fs_mild"].keys()) & set(all_results["dp_sgd"].keys())
    if has_feddpa:
        feddpa_seeds = set(all_results["feddpa"].keys())
        common_with_feddpa = common & feddpa_seeds
    seeds = sorted(common)

    if len(seeds) < 3:
        return {"decision": "INCOMPLETE", "reason": f"Only {len(seeds)} common seeds"}

    # Per-seed comparisons
    fs_beats_dpsgd_worst = 0
    fs_beats_feddpa_worst = 0
    fs_macro, dpsgd_macro = [], []
    # Paired lists for FedDPA CI: fs_macro values only for seeds where FedDPA exists
    fs_macro_for_feddpa, feddpa_macro = [], []

    for seed in seeds:
        fs = extract_final_metrics(all_results["fs_mild"][seed])
        dp = extract_final_metrics(all_results["dp_sgd"][seed])

        fs_worst = combined_worst_group_f1(fs)
        dp_worst = combined_worst_group_f1(dp)

        if fs_worst > dp_worst:
            fs_beats_dpsgd_worst += 1

        fs_macro_val = combined_macro_f1(fs)
        fs_macro.append(fs_macro_val)
        dpsgd_macro.append(combined_macro_f1(dp))

        if has_feddpa and seed in all_results["feddpa"]:
            dpa = extract_final_metrics(all_results["feddpa"][seed])
            dpa_worst = combined_worst_group_f1(dpa)
            if fs_worst > dpa_worst:
                fs_beats_feddpa_worst += 1
            fs_macro_for_feddpa.append(fs_macro_val)
            feddpa_macro.append(combined_macro_f1(dpa))

    threshold = max(1, math.ceil(0.8 * len(seeds)))  # 4/5 for 5 seeds

    # Condition 1: FS-mild beats DP-SGD on worst-group F1 in >= threshold seeds
    cond1_dpsgd = fs_beats_dpsgd_worst >= threshold
    # If FedDPA present with enough seeds, also require beating it
    # Use separate threshold based on FedDPA's own seed count
    if has_feddpa and len(common_with_feddpa) >= 3:
        feddpa_threshold = max(1, math.ceil(0.8 * len(common_with_feddpa)))
        cond1 = cond1_dpsgd and (fs_beats_feddpa_worst >= feddpa_threshold)
    else:
        cond1 = cond1_dpsgd

    # Condition 2: Paired 95% CI for macro-F1 delta (FS-mild minus DP-SGD) excludes 0
    ci_dpsgd = paired_confidence_interval(fs_macro, dpsgd_macro)
    cond2 = bool(ci_dpsgd["ci_low"] > 0)

    if cond1 and cond2:
        decision = "PASS"
    elif cond1 or cond2:
        decision = "MARGINAL"
    else:
        decision = "FAIL"

    criteria = {
        "decision": decision,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "threshold_needed": threshold,
        "condition_1_worst_group": cond1,
        "fs_beats_dpsgd_worst": fs_beats_dpsgd_worst,
        "macro_f1_ci_vs_dpsgd": ci_dpsgd,
        "condition_2_macro_ci": cond2,
        "feddpa_included": has_feddpa,
    }
    if has_feddpa:
        ci_feddpa = paired_confidence_interval(
            fs_macro_for_feddpa, feddpa_macro
        ) if feddpa_macro else None
        criteria["fs_beats_feddpa_worst"] = fs_beats_feddpa_worst
        criteria["feddpa_seeds_used"] = len(feddpa_macro)
        criteria["macro_f1_ci_vs_feddpa"] = ci_feddpa

    return criteria


def print_summary(all_results: Dict[str, Dict[int, dict]], feature_set: str):
    """Print formatted summary of all results."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — {feature_set}")
    print(f"{'='*70}\n")

    for config_name in ["baseline_fl", "fs_mild", "fs_aggressive", "dp_sgd", "feddpa"]:
        if config_name not in all_results:
            continue

        seeds_data = all_results[config_name]
        seeds = sorted(seeds_data.keys())
        n = len(seeds)

        metrics_list = [extract_final_metrics(seeds_data[s]) for s in seeds]

        macro_a = [m["home_a_macro_f1"] for m in metrics_list]
        macro_b = [m["home_b_macro_f1"] for m in metrics_list]
        worst_all = [combined_worst_group_f1(m) for m in metrics_list]
        combined = [combined_macro_f1(m) for m in metrics_list]

        print(f"  {config_name} ({n} seeds)")
        print(f"    Macro-F1 Home A: {np.mean(macro_a):.3f} +/- {np.std(macro_a):.3f}")
        print(f"    Macro-F1 Home B: {np.mean(macro_b):.3f} +/- {np.std(macro_b):.3f}")
        print(f"    Combined Macro:  {np.mean(combined):.3f} +/- {np.std(combined):.3f}")
        print(f"    Worst-group F1:  {np.mean(worst_all):.3f} +/- {np.std(worst_all):.3f}")

        # Per-class breakdown (mean across seeds)
        class_names = sorted(metrics_list[0]["home_a_per_class"].keys())
        print(f"    Per-class F1 (Home A / Home B, mean):")
        for cls in class_names:
            a_vals = [m["home_a_per_class"][cls] for m in metrics_list]
            b_vals = [m["home_b_per_class"][cls] for m in metrics_list]
            print(f"      {cls:20s}  A={np.mean(a_vals):.3f}  B={np.mean(b_vals):.3f}")
        print()

    # Evaluation criteria
    criteria = evaluate_criteria(all_results)
    print(f"{'='*70}")
    print(f"  EVALUATION CRITERIA: {criteria['decision']}")
    print(f"{'='*70}")
    for k, v in criteria.items():
        if k != "decision":
            print(f"    {k}: {v}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze FL results")
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--equal-weight", action="store_true")
    parser.add_argument("--save-summary", action="store_true",
                        help="Save summary.json alongside results")
    args = parser.parse_args()

    all_results = load_results(args.feature_set, args.equal_weight)
    if not all_results:
        print(f"No results found in {RESULTS_DIR}/{args.feature_set}/")
        sys.exit(1)

    print_summary(all_results, args.feature_set)

    if args.save_summary:
        criteria = evaluate_criteria(all_results)
        fname = "summary_equal_weight.json" if args.equal_weight else "summary.json"
        summary_path = os.path.join(RESULTS_DIR, args.feature_set, fname)
        with open(summary_path, "w") as f:
            json.dump(criteria, f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
