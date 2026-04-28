from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix

from run_anchor_hybrid_probe import (
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    make_scaffold_split,
    split_is_usable,
    tune_alpha,
)


DEFAULT_K_VALUES = "1,3,5,10"


def parse_k_values(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one K value is required.")
    if any(k < 1 for k in values):
        raise ValueError("K values must be positive integers.")
    return values


def resolve_n_jobs(requested: int | None, cap: int) -> int:
    available = max(1, int(os.cpu_count() or 1))
    if requested is not None:
        return max(1, min(int(requested), available))
    return max(1, min(cap, available - 8))


def fit_rf(x_train: np.ndarray, y_train: np.ndarray, seed: int, n_jobs: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_knn(x_train: np.ndarray, y_train: np.ndarray, k: int, n_jobs: int) -> KNeighborsClassifier:
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="jaccard",
        n_jobs=n_jobs,
    )
    model.fit(x_train.astype(bool), y_train)
    return model


def run_task(
    task_cfg: dict,
    k_values: list[int],
    seed: int,
    rf_n_jobs: int,
    knn_n_jobs: int,
) -> list[dict]:
    df = pd.read_csv(task_cfg["path"])
    raw_label = task_cfg.get("raw_label_col", task_cfg["label"])
    df = df[[task_cfg["smiles_col"], raw_label]].dropna().copy()
    df = df.rename(columns={task_cfg["smiles_col"]: "smiles", raw_label: "label"})
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")

    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        return [
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "split": "scaffold",
                "model": "skipped",
                "k": np.nan,
                "status": f"skipped_{reason}",
            }
        ]

    x = morgan_fingerprint_matrix(df["smiles"].tolist(), radius=2, n_bits=2048)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    rf = fit_rf(x_train, y_train, seed=seed, n_jobs=rf_n_jobs)
    rf_valid = rf.predict_proba(x_valid)[:, 1]
    rf_test = rf.predict_proba(x_test)[:, 1]

    rows: list[dict] = []
    for k in k_values:
        k_eff = min(k, len(x_train))
        knn = fit_knn(x_train, y_train, k=k_eff, n_jobs=knn_n_jobs)
        knn_valid = knn.predict_proba(x_valid.astype(bool))[:, 1]
        knn_test = knn.predict_proba(x_test.astype(bool))[:, 1]
        alpha, valid_metrics = tune_alpha(y_valid, rf_valid, knn_valid)
        hybrid_test = alpha * rf_test + (1.0 - alpha) * knn_test

        common = {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "split": "scaffold",
            "k": k,
            "effective_k": k_eff,
            "seed": seed,
            "status": "ok",
            "n_samples": len(df),
            "n_positive": int(y.sum()),
            "train_size": int(len(train_idx)),
            "valid_size": int(len(valid_idx)),
            "test_size": int(len(test_idx)),
        }
        rows.append(
            {
                **common,
                "model": "retrieval_only",
                "alpha_rf": np.nan,
                "valid_auprc_at_alpha": np.nan,
                **evaluate_probs(y_test, knn_test),
            }
        )
        rows.append(
            {
                **common,
                "model": "retrieval_plus_reasoning",
                "alpha_rf": alpha,
                "valid_auprc_at_alpha": valid_metrics["auprc"],
                **evaluate_probs(y_test, hybrid_test),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=None)
    parser.add_argument("--knn-n-jobs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "anchor_k_sensitivity")
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    rf_n_jobs = resolve_n_jobs(args.rf_n_jobs, cap=96)
    knn_n_jobs = resolve_n_jobs(args.knn_n_jobs, cap=64)

    all_rows = []
    for task_cfg in TASKS:
        print(f"RUN {task_cfg['dataset']}::{task_cfg['label']} K={k_values}")
        all_rows.extend(
            run_task(
                task_cfg,
                k_values=k_values,
                seed=args.seed,
                rf_n_jobs=rf_n_jobs,
                knn_n_jobs=knn_n_jobs,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "results.csv", index=False)
    summary = {
        "num_rows": int(len(df)),
        "k_values": k_values,
        "seed": args.seed,
        "rf_n_jobs": rf_n_jobs,
        "knn_n_jobs": knn_n_jobs,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not df.empty:
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
