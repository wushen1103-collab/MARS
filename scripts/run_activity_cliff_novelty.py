from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.anchor_reliability import compute_anchor_features
from admet_shift_reliability.features import morgan_fingerprint_matrix

from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, fit_rf, load_task_frame, make_scaffold_split

BUCKETS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
BUCKET_NAMES = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]


def bucketize_similarity(similarity: np.ndarray) -> np.ndarray:
    return np.asarray(pd.cut(similarity, bins=BUCKETS, labels=BUCKET_NAMES, include_lowest=True, right=False).astype(str))


def fit_meta(valid_probs: np.ndarray, valid_anchor: np.ndarray, valid_y: np.ndarray, seed: int):
    x_valid = np.column_stack([valid_probs, valid_anchor, np.abs(valid_probs - valid_anchor)])
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int | None) -> tuple[list[dict], list[dict], pd.DataFrame]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    x = morgan_fingerprint_matrix(df["smiles"].tolist())
    xb = x.astype(bool)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    rf_valid = rf.predict_proba(x[valid_idx])[:, 1]
    rf_test = rf.predict_proba(x[test_idx])[:, 1]
    anchor_valid = compute_anchor_features(xb[train_idx], y[train_idx], xb[valid_idx], n_neighbors=10)["anchor_prob"]
    anchor_test_features = compute_anchor_features(xb[train_idx], y[train_idx], xb[test_idx], n_neighbors=10)
    anchor_test = anchor_test_features["anchor_prob"]
    meta = fit_meta(rf_valid, anchor_valid, y[valid_idx], seed=seed)
    meta_x = np.column_stack([rf_test, anchor_test, np.abs(rf_test - anchor_test)])
    meta_test = meta.predict_proba(meta_x)[:, 1]

    nn = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=1)
    nn.fit(xb[train_idx])
    distance, index = nn.kneighbors(xb[test_idx], return_distance=True)
    nearest_sim = 1.0 - distance[:, 0]
    nearest_train_label = y[train_idx][index[:, 0]]
    is_activity_cliff = (nearest_sim >= 0.7) & (nearest_train_label != y[test_idx])
    buckets = bucketize_similarity(nearest_sim)

    preds = {"rf": rf_test, "retrieval_only": anchor_test, "retrieval_plus_reasoning": meta_test}
    bucket_rows = []
    cliff_rows = []
    for name, probs in preds.items():
        for bucket_name in BUCKET_NAMES:
            mask = buckets == bucket_name
            if mask.sum() < 2 or np.unique(y[test_idx][mask]).size < 2:
                continue
            bucket_rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": name,
                    "novelty_bucket": bucket_name,
                    "n": int(mask.sum()),
                    "positive_rate": float(np.mean(y[test_idx][mask])),
                    **evaluate_probs(y[test_idx][mask], probs[mask]),
                }
            )
        for group_name, mask in [("activity_cliff", is_activity_cliff), ("non_cliff", ~is_activity_cliff)]:
            if mask.sum() < 2 or np.unique(y[test_idx][mask]).size < 2:
                continue
            cliff_rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "positive_rate": float(np.mean(y[test_idx][mask])),
                    **evaluate_probs(y[test_idx][mask], probs[mask]),
                }
            )

    sample_df = pd.DataFrame(
        {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "smiles": df.iloc[test_idx]["smiles"].to_numpy(),
            "y_true": y[test_idx],
            "nearest_train_similarity": nearest_sim,
            "nearest_train_label": nearest_train_label,
            "novelty_bucket": buckets,
            "is_activity_cliff": is_activity_cliff,
            "rf_prob": rf_test,
            "retrieval_prob": anchor_test,
            "reasoning_prob": meta_test,
            "anchor_distance_mean": anchor_test_features["anchor_distance_mean"],
            "anchor_disagreement": anchor_test_features["anchor_disagreement"],
        }
    )
    return bucket_rows, cliff_rows, sample_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "activity_cliff_novelty")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_bucket, all_cliff, sample_frames = [], [], []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}")
        bucket_rows, cliff_rows, sample_df = run_task(task, seed=args.seed, rf_n_jobs=args.rf_n_jobs)
        all_bucket.extend(bucket_rows)
        all_cliff.extend(cliff_rows)
        sample_frames.append(sample_df)
    pd.DataFrame(all_bucket).to_csv(args.output_dir / "novelty_bucket_metrics.csv", index=False)
    pd.DataFrame(all_cliff).to_csv(args.output_dir / "activity_cliff_metrics.csv", index=False)
    pd.concat(sample_frames, ignore_index=True).to_csv(args.output_dir / "test_sample_novelty_annotations.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"bucket_rows": len(all_bucket), "cliff_rows": len(all_cliff)}, indent=2))


if __name__ == "__main__":
    main()
