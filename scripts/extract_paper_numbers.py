#!/usr/bin/env python3
"""Extract canonical paper numbers from FL experiment JSON results.

Usage:
  python scripts/extract_paper_numbers.py
  python scripts/extract_paper_numbers.py --feature-set baseline_16 --model-size medium

Outputs:
  - results/paper_numbers.json (default small model on baseline_16)
  - results/paper_numbers_<feature_set>_<model_size>.json (non-default runs)
  - concise console summary for table rows / text claims
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def ci95(vals_a: List[float], vals_b: List[float]) -> Dict[str, float]:
    """Paired t CI for (A - B), matching paper pipeline semantics."""
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n < 2:
        return {"mean_diff": mean_diff, "ci_low": float("nan"), "ci_high": float("nan")}

    # t_{0.975,4}=2.776 for n=5. We keep dynamic lookup for robustness.
    # Avoid scipy dependency: hardcode small-n values used in this project.
    t_crit_lookup = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    t_crit = t_crit_lookup.get(n, 1.96)
    se = float(np.std(diffs, ddof=1) / math.sqrt(n))
    margin = t_crit * se
    return {
        "mean_diff": mean_diff,
        "ci_low": float(mean_diff - margin),
        "ci_high": float(mean_diff + margin),
    }


def fmt3(x: float) -> str:
    return f"{x:.3f}".lstrip("0")


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def config_has_all_seed_files(base: Path, cfg: str, seeds: List[int]) -> bool:
    return all((base / f"seed_{s}" / f"{cfg}.json").exists() for s in seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical paper numbers from result JSONs")
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--model-size", default="small", choices=["small", "medium"])
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results"
    subdir = args.feature_set if args.model_size == "small" else f"{args.feature_set}_{args.model_size}"
    base = repo_root / "results" / subdir
    mia_base = base / "mia"

    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    summary_path = base / "summary.json"
    summary_equal_path = base / "summary_equal_weight.json"
    mia_summary_path = mia_base / "mia_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    summary_equal = load_json(summary_equal_path) if summary_equal_path.exists() else {}
    mia_summary = load_json(mia_summary_path) if mia_summary_path.exists() else {}

    seeds = args.seeds
    default_configs_all = ["baseline_fl", "fs_mild", "fs_aggressive", "dp_sgd", "feddpa"]
    configs_all = [c for c in default_configs_all if config_has_all_seed_files(base, c, seeds)]
    if not configs_all:
        raise RuntimeError(f"No complete config result sets found in {base} for seeds {seeds}")

    configs_primary = [c for c in ["baseline_fl", "fs_mild", "dp_sgd"] if c in configs_all]
    homes = ["home_a", "home_b"]

    # Optional import for canonical dataset counts.
    sys.path.insert(0, str(repo_root))
    from fl_pipeline.config import BASELINE_16, CORE_CLASSES, HOME_A_PATH, HOME_B_PATH  # noqa: E402
    from fl_pipeline.data import load_and_filter  # noqa: E402

    classes = sorted(CORE_CLASSES)
    df_a, audit_a = load_and_filter(HOME_A_PATH, BASELINE_16, CORE_CLASSES)
    df_b, audit_b = load_and_filter(HOME_B_PATH, BASELINE_16, CORE_CLASSES)
    del df_a, df_b

    class_counts_a = audit_a["class_counts_final"]
    class_counts_b = audit_b["class_counts_final"]
    total_a = int(audit_a["after_core_filter"])
    total_b = int(audit_b["after_core_filter"])

    results: Dict[str, dict] = {
        "meta": {
            "feature_set": args.feature_set,
            "model_size": args.model_size,
            "results_subdir": subdir,
            "seeds": seeds,
            "configs_all": configs_all,
            "configs_primary": configs_primary,
        },
        "dataset": {
            "home_a_total": total_a,
            "home_b_total": total_b,
            "total": total_a + total_b,
            "home_a_after_pkt_filter": int(audit_a["after_pkt_filter"]),
            "home_b_after_pkt_filter": int(audit_b["after_pkt_filter"]),
            "rows": [],
        },
        "main_table": {},
        "runtime_table": {},
        "per_class_table": {},
        "criteria": {
            "primary": summary,
            "equal_weight": summary_equal,
        },
        "mia_table": {} if mia_summary else None,
        "mia_home_table": {} if mia_summary else None,
        "mia_paired_ci": mia_summary.get("paired_ci", {}) if mia_summary else None,
    }

    for c in classes:
        a = int(class_counts_a[c])
        b = int(class_counts_b[c])
        pct_a = 100.0 * a / total_a
        pct_b = 100.0 * b / total_b
        results["dataset"]["rows"].append(
            {
                "class": c,
                "home_a_count": a,
                "home_a_pct": float(pct_a),
                "home_b_count": b,
                "home_b_pct": float(pct_b),
                # Ratio = larger share / smaller share (always >= 1), matching paper table caption.
                "ratio_larger_over_smaller": float(max(pct_a, pct_b) / min(pct_a, pct_b)),
            }
        )

    # Main table stats.
    for cfg in configs_all:
        per_seed = []
        for s in seeds:
            p = base / f"seed_{s}" / f"{cfg}.json"
            d = load_json(p)
            r = d["rounds"][-1]
            a = float(r["home_a"]["macro_f1"])
            b = float(r["home_b"]["macro_f1"])
            c = (a + b) / 2
            w = min(
                list(r["home_a"]["per_class_f1"].values())
                + list(r["home_b"]["per_class_f1"].values())
            )
            per_seed.append({"seed": s, "a_macro": a, "b_macro": b, "combined": c, "worst": float(w)})

        results["main_table"][cfg] = {
            "a_macro": mean_std([x["a_macro"] for x in per_seed]),
            "b_macro": mean_std([x["b_macro"] for x in per_seed]),
            "combined": mean_std([x["combined"] for x in per_seed]),
            "worst": mean_std([x["worst"] for x in per_seed]),
            "per_seed": per_seed,
        }

        # Runtime stats (median + IQR to handle outlier rounds from execution pauses)
        round_times = []
        n_rounds = None
        for s in seeds:
            p = base / f"seed_{s}" / f"{cfg}.json"
            d = load_json(p)
            ts = [float(r.get("time_s", 0.0)) for r in d["rounds"]]
            round_times.extend(ts)
            if n_rounds is None:
                n_rounds = len(ts)
            else:
                assert n_rounds == len(ts), (
                    f"Seed {s} has {len(ts)} rounds but expected {n_rounds} for {cfg}"
                )
        rt = np.array(round_times)
        median_round = float(np.median(rt))
        results["runtime_table"][cfg] = {
            "round_time_s": {
                "median": median_round,
                "q25": float(np.percentile(rt, 25)),
                "q75": float(np.percentile(rt, 75)),
            },
            "run_time_min": float(median_round * n_rounds / 60.0),
        }

    # Per-class table (BL/FS/DP only).
    for cfg in configs_primary:
        results["per_class_table"][cfg] = {"home_a": {}, "home_b": {}}
        for home in homes:
            for c in classes:
                vals = []
                for s in seeds:
                    p = base / f"seed_{s}" / f"{cfg}.json"
                    d = load_json(p)
                    vals.append(float(d["rounds"][-1][home]["per_class_f1"][c]))
                results["per_class_table"][cfg][home][c] = float(np.mean(vals))

    # MIA table (equal-home averages already pre-aggregated in mia_summary).
    if mia_summary:
        for cfg in configs_primary:
            if cfg not in mia_summary.get("per_config", {}):
                continue
            eq = mia_summary["per_config"][cfg]["equal_avg"]
            results["mia_table"][cfg] = {
                "auc": {"mean": float(eq["auc_mean"]), "std": float(eq["auc_std"])},
                "tpr_at_1fpr": {"mean": float(eq["tpr_at_1fpr_mean"]), "std": float(eq["tpr_at_1fpr_std"])},
                "tpr_at_5fpr": {"mean": float(eq["tpr_at_5fpr_mean"]), "std": float(eq["tpr_at_5fpr_std"])},
                "advantage": {"mean": float(eq["advantage_mean"]), "std": float(eq["advantage_std"])},
            }
            results["mia_home_table"][cfg] = {}
            for home in ["home_a", "home_b"]:
                h = mia_summary["per_config"][cfg][home]
                results["mia_home_table"][cfg][home] = {
                    "auc": {"mean": float(h["auc_mean"]), "std": float(h["auc_std"])},
                    "tpr_at_5fpr": {"mean": float(h["tpr_at_5fpr_mean"]), "std": float(h["tpr_at_5fpr_std"])},
                }

    # Save canonical dump.
    if args.output:
        out_json = Path(args.output)
    elif args.feature_set == "baseline_16" and args.model_size == "small":
        out_json = out_dir / "paper_numbers.json"
    else:
        out_json = out_dir / f"paper_numbers_{subdir}.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    # Console summary for quick copy/paste validation.
    print(f"Saved: {out_json}")
    print("\nMain table rows (mean±std):")
    for cfg in configs_all:
        row = results["main_table"][cfg]
        print(
            f"  {cfg:13s} "
            f"A {fmt3(row['a_macro']['mean'])}±{fmt3(row['a_macro']['std'])} | "
            f"B {fmt3(row['b_macro']['mean'])}±{fmt3(row['b_macro']['std'])} | "
            f"C {fmt3(row['combined']['mean'])}±{fmt3(row['combined']['std'])} | "
            f"W {fmt3(row['worst']['mean'])}±{fmt3(row['worst']['std'])}"
        )

    print("\nPer-class means (BL/FS/DP):")
    for c in classes:
        bl_a = results["per_class_table"]["baseline_fl"]["home_a"][c]
        fs_a = results["per_class_table"]["fs_mild"]["home_a"][c]
        dp_a = results["per_class_table"]["dp_sgd"]["home_a"][c]
        bl_b = results["per_class_table"]["baseline_fl"]["home_b"][c]
        fs_b = results["per_class_table"]["fs_mild"]["home_b"][c]
        dp_b = results["per_class_table"]["dp_sgd"]["home_b"][c]
        print(
            f"  {c:14s} "
            f"A[{fmt3(bl_a)}, {fmt3(fs_a)}, {fmt3(dp_a)}] "
            f"B[{fmt3(bl_b)}, {fmt3(fs_b)}, {fmt3(dp_b)}]"
        )

    if results["mia_table"]:
        print("\nMIA table rows (equal-home mean±std):")
        for cfg in configs_primary:
            if cfg not in results["mia_table"]:
                continue
            r = results["mia_table"][cfg]
            print(
                f"  {cfg:11s} "
                f"AUC {fmt3(r['auc']['mean'])}±{fmt3(r['auc']['std'])} | "
                f"TPR1 {fmt3(r['tpr_at_1fpr']['mean'])}±{fmt3(r['tpr_at_1fpr']['std'])} | "
                f"TPR5 {fmt3(r['tpr_at_5fpr']['mean'])}±{fmt3(r['tpr_at_5fpr']['std'])} | "
                f"Adv {fmt3(r['advantage']['mean'])}±{fmt3(r['advantage']['std'])}"
            )
    else:
        print("\nMIA summary not found for this result set (skipped MIA table extraction).")

    print("\nRuntime rows (median round × n_rounds):")
    for cfg in configs_all:
        r = results["runtime_table"][cfg]
        rt = r["round_time_s"]
        rm = r["run_time_min"]
        print(
            f"  {cfg:11s} "
            f"round {rt['median']:.1f}s [{rt['q25']:.1f}, {rt['q75']:.1f}] | "
            f"run ~{rm:.1f}min"
        )

    print("\nEvaluation criteria snapshots:")
    if summary:
        print(
            "  primary:",
            summary["decision"],
            f"macro CI [{summary['macro_f1_ci_vs_dpsgd']['ci_low']:+.4f}, {summary['macro_f1_ci_vs_dpsgd']['ci_high']:+.4f}]",
        )
    else:
        print("  primary: missing summary.json")
    if summary_equal:
        print(
            "  equal-weight:",
            summary_equal["decision"],
            f"macro CI [{summary_equal['macro_f1_ci_vs_dpsgd']['ci_low']:+.4f}, {summary_equal['macro_f1_ci_vs_dpsgd']['ci_high']:+.4f}]",
        )
    else:
        print("  equal-weight: missing summary_equal_weight.json")


if __name__ == "__main__":
    main()
