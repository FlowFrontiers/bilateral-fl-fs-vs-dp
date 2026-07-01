"""Data loading, filtering, splitting, and federated-safe scaling."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Dict, List, Tuple, Any


def load_and_filter(
    parquet_path: str,
    feature_cols: List[str],
    core_classes: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load parquet and apply 3 filters. Returns filtered DataFrame + audit."""
    cols_to_load = list(dict.fromkeys(
        feature_cols + ["category", "confidence", "bidirectional_packets"]
    ))
    df = pd.read_parquet(parquet_path, columns=cols_to_load)
    audit = {"raw_count": len(df), "class_counts_raw": df["category"].value_counts().to_dict()}

    # Filter 1: bidirectional_packets >= 2
    df = df[df["bidirectional_packets"] >= 2]
    audit["after_pkt_filter"] = len(df)

    # Filter 2: confidence == 'DPI'
    df = df[df["confidence"] == "DPI"]
    audit["after_dpi_filter"] = len(df)

    # Filter 3: CORE classes only
    df = df[df["category"].isin(core_classes)]
    audit["after_core_filter"] = len(df)
    audit["class_counts_final"] = df["category"].value_counts().to_dict()

    # Drop columns used only for filtering
    drop_cols = {"confidence"}
    if "bidirectional_packets" not in feature_cols:
        drop_cols.add("bidirectional_packets")
    df = df.drop(columns=list(drop_cols & set(df.columns)))

    return df, audit


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False
    )


def prepare_federated_data(
    home_a_path: str,
    home_b_path: str,
    feature_cols: List[str],
    core_classes: List[str],
    seed: int,
    test_ratio: float = 0.2,
    batch_size: int = 256,
    equal_weight: bool = False,
) -> Dict[str, Any]:
    """Full data pipeline: load → filter → split → federated-safe scale → DataLoaders.

    Scaling uses sufficient statistics (n, sum, sum_sq) per home — no raw data exchange.
    """
    df_a, audit_a = load_and_filter(home_a_path, feature_cols, core_classes)
    df_b, audit_b = load_and_filter(home_b_path, feature_cols, core_classes)

    # Deterministic label encoder (alphabetical)
    le = LabelEncoder()
    le.fit(sorted(core_classes))

    X_a = df_a[feature_cols].values.astype(np.float32)
    y_a = le.transform(df_a["category"].values)
    X_b = df_b[feature_cols].values.astype(np.float32)
    y_b = le.transform(df_b["category"].values)

    # Stratified 80/20 split per home
    X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
        X_a, y_a, test_size=test_ratio, random_state=seed, stratify=y_a
    )
    X_b_tr, X_b_te, y_b_tr, y_b_te = train_test_split(
        X_b, y_b, test_size=test_ratio, random_state=seed, stratify=y_b
    )

    # Federated-safe scaling: sufficient statistics only
    n_a, n_b = len(X_a_tr), len(X_b_tr)
    sum_a, sum_sq_a = X_a_tr.sum(axis=0), (X_a_tr ** 2).sum(axis=0)
    sum_b, sum_sq_b = X_b_tr.sum(axis=0), (X_b_tr ** 2).sum(axis=0)

    n_total = n_a + n_b
    global_mean = (sum_a + sum_b) / n_total
    global_var = (sum_sq_a + sum_sq_b) / n_total - global_mean ** 2
    global_std = np.sqrt(np.maximum(global_var, 0))
    global_std[global_std == 0] = 1.0

    # Apply scaling
    for arr in (X_a_tr, X_a_te, X_b_tr, X_b_te):
        arr[:] = (arr - global_mean) / global_std

    # FedAvg weights
    if equal_weight:
        w_a, w_b = 0.5, 0.5
    else:
        w_a = n_a / n_total
        w_b = n_b / n_total

    return {
        "home_a": {
            "X_train": X_a_tr, "y_train": y_a_tr,
            "X_test": X_a_te, "y_test": y_a_te,
            "train_loader": _make_loader(X_a_tr, y_a_tr, batch_size, True, seed),
            "test_loader": _make_loader(X_a_te, y_a_te, batch_size, False, seed),
            "n_train": n_a, "n_test": len(X_a_te),
        },
        "home_b": {
            "X_train": X_b_tr, "y_train": y_b_tr,
            "X_test": X_b_te, "y_test": y_b_te,
            "train_loader": _make_loader(X_b_tr, y_b_tr, batch_size, True, seed),
            "test_loader": _make_loader(X_b_te, y_b_te, batch_size, False, seed),
            "n_train": n_b, "n_test": len(X_b_te),
        },
        "class_names": [str(c) for c in le.classes_],
        "num_features": len(feature_cols),
        "num_classes": len(core_classes),
        "scaler": {"mean": global_mean.tolist(), "std": global_std.tolist()},
        "weights": {"home_a": float(w_a), "home_b": float(w_b)},
        "audit": {"home_a": audit_a, "home_b": audit_b},
    }
