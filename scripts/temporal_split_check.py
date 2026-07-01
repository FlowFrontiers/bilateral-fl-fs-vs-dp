#!/usr/bin/env python3
"""Internal sanity check (reviewer request): does the FS-mild > DP-SGD worst-group
ordering survive a TEMPORAL split instead of the random stratified split?

Earlier-80% / later-20% per home (by bidirectional_first_seen_ms), then the SAME
FedAvg pipeline (baseline_fl, fs_mild, dp_sgd). Writes JSON incrementally.

Run:  cd bilateral-fl-fs-vs-dp && ../.venv/bin/python scripts/temporal_split_check.py --seeds 42 123 456 789 1024
Output: results/temporal_split_check.json

Medium model:
  ../.venv/bin/python scripts/temporal_split_check.py --model-size medium --seeds 42 123 456 789 1024
Output: results/temporal_split_check_medium.json
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl_pipeline.config import (BASELINE_16, FS_MILD_BASELINE, CORE_CLASSES, MODEL_SIZES,
                                ExperimentConfig, HOME_A_PATH, HOME_B_PATH, RESULTS_DIR)
from fl_pipeline.train import run_standard_fl, run_dp_sgd_fl

TIME_COL = "bidirectional_first_seen_ms"


def _load_sorted(path, feature_cols):
    cols = list(dict.fromkeys(feature_cols + ["category", "confidence",
                                              "bidirectional_packets", TIME_COL]))
    df = pd.read_parquet(path, columns=cols)
    df = df[df["bidirectional_packets"] >= 2]
    df = df[df["confidence"] == "DPI"]
    df = df[df["category"].isin(CORE_CLASSES)]
    return df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)


def temporal_data(feature_cols, seed, test_ratio=0.2, batch_size=256):
    da, db = _load_sorted(HOME_A_PATH, feature_cols), _load_sorted(HOME_B_PATH, feature_cols)
    le = LabelEncoder(); le.fit(sorted(CORE_CLASSES))

    def split(df):
        k = int(len(df) * (1 - test_ratio))
        tr, te = df.iloc[:k], df.iloc[k:]
        return (tr[feature_cols].values.astype(np.float32), le.transform(tr["category"].values),
                te[feature_cols].values.astype(np.float32), le.transform(te["category"].values))

    Xa_tr, ya_tr, Xa_te, ya_te = split(da)
    Xb_tr, yb_tr, Xb_te, yb_te = split(db)

    na, nb = len(Xa_tr), len(Xb_tr); ntot = na + nb
    s = Xa_tr.sum(0) + Xb_tr.sum(0); sq = (Xa_tr ** 2).sum(0) + (Xb_tr ** 2).sum(0)
    mean = s / ntot; var = sq / ntot - mean ** 2
    std = np.sqrt(np.maximum(var, 0)); std[std == 0] = 1.0
    for arr in (Xa_tr, Xa_te, Xb_tr, Xb_te): arr[:] = (arr - mean) / std

    def mk(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)

    def support(y):  # per-class test-set counts (catch vanished classes)
        return {le.classes_[i]: int((y == i).sum()) for i in range(len(le.classes_))}

    return {
        "home_a": {"X_train": Xa_tr, "y_train": ya_tr, "X_test": Xa_te, "y_test": ya_te,
                   "train_loader": mk(Xa_tr, ya_tr, True), "test_loader": mk(Xa_te, ya_te, False),
                   "n_train": na, "n_test": len(Xa_te)},
        "home_b": {"X_train": Xb_tr, "y_train": yb_tr, "X_test": Xb_te, "y_test": yb_te,
                   "train_loader": mk(Xb_tr, yb_tr, True), "test_loader": mk(Xb_te, yb_te, False),
                   "n_train": nb, "n_test": len(Xb_te)},
        "class_names": [str(c) for c in le.classes_], "num_features": len(feature_cols),
        "num_classes": len(CORE_CLASSES),
        "weights": {"home_a": na / ntot, "home_b": nb / ntot},
        "test_support": {"home_a": support(ya_te), "home_b": support(yb_te)},
    }


def combined(results):
    last = results["rounds"][-1]
    a, b = last["home_a"], last["home_b"]
    return {"combined_macro_f1": (a["macro_f1"] + b["macro_f1"]) / 2,
            "worst_group_f1": min(a["worst_group_f1"], b["worst_group_f1"]),
            "home_a": a, "home_b": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--model-size", choices=["small", "medium"], default="small")
    args = ap.parse_args()
    cfg = ExperimentConfig(hidden_dims=MODEL_SIZES[args.model_size])
    suffix = "" if args.model_size == "small" else "_medium"
    out_path = os.path.join(RESULTS_DIR, f"temporal_split_check{suffix}.json")
    out = {"split": "temporal_80_20", "model": args.model_size, "seeds": {}}

    for seed in args.seeds:
        d_bl = temporal_data(BASELINE_16, seed)
        r_bl = combined(run_standard_fl(d_bl, cfg, seed, "baseline_fl"))
        d_fs = temporal_data(FS_MILD_BASELINE, seed)
        r_fs = combined(run_standard_fl(d_fs, cfg, seed, "fs_mild"))
        d_dp = temporal_data(BASELINE_16, seed)
        r_dp = combined(run_dp_sgd_fl(d_dp, cfg, seed))
        out["seeds"][str(seed)] = {
            "baseline_fl": r_bl, "fs_mild": r_fs, "dp_sgd": r_dp,
            "fs_beats_dp_worst": bool(r_fs["worst_group_f1"] > r_dp["worst_group_f1"]),
            "test_support": d_dp["test_support"],
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[seed {seed}] FS worst={r_fs['worst_group_f1']:.3f} vs DP worst={r_dp['worst_group_f1']:.3f} "
              f"-> FS>DP: {out['seeds'][str(seed)]['fs_beats_dp_worst']}  (saved {out_path})")

    wins = sum(v["fs_beats_dp_worst"] for v in out["seeds"].values())
    print(f"\nORDERING SURVIVES (FS-mild worst > DP-SGD worst): {wins}/{len(out['seeds'])} seeds")


if __name__ == "__main__":
    main()
