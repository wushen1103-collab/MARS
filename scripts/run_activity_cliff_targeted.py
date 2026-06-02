from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    load_task_frame,
    make_scaffold_split,
)


SIM_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
SIM_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">=0.95"]


def nearest_opposite_label_similarity(x_train_bool: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y_train), dtype=float)
    for cls in [0, 1]:
        query_idx = np.where(y_train == cls)[0]
        opp_idx = np.where(y_train != cls)[0]
        if len(query_idx) == 0 or len(opp_idx) == 0:
            continue
        nn = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=1)
        nn.fit(x_train_bool[opp_idx])
        distance, _ = nn.kneighbors(x_train_bool[query_idx], return_distance=True)
        out[query_idx] = 1.0 - distance[:, 0]
    return out


def nearest_train_similarity(x_train_bool: np.ndarray, x_query_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=1)
    nn.fit(x_train_bool)
    distance, index = nn.kneighbors(x_query_bool, return_distance=True)
    return 1.0 - distance[:, 0], index[:, 0]


def fit_weighted_rf(x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int, n_jobs: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def safe_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan, "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan}
    return evaluate_probs(y_true, probs)


def run_task(task_cfg: dict, seed: int, n_jobs: int, alpha: float) -> tuple[list[dict], pd.DataFrame]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    x = morgan_fingerprint_matrix(df["smiles"].tolist())
    xb = x.astype(bool)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    train_risk_sim = nearest_opposite_label_similarity(xb[train_idx], y[train_idx])
    train_risk = np.clip((train_risk_sim - 0.5) / 0.5, 0.0, 1.0)
    sample_weight = 1.0 + alpha * train_risk

    base = fit_weighted_rf(x[train_idx], y[train_idx], sample_weight=None, seed=seed, n_jobs=n_jobs)
    weighted = fit_weighted_rf(x[train_idx], y[train_idx], sample_weight=sample_weight, seed=seed, n_jobs=n_jobs)
    base_prob = base.predict_proba(x[test_idx])[:, 1]
    weighted_prob = weighted.predict_proba(x[test_idx])[:, 1]

    test_sim, nearest = nearest_train_similarity(xb[train_idx], xb[test_idx])
    nearest_label = y[train_idx][nearest]
    activity_cliff = (test_sim >= 0.7) & (nearest_label != y[test_idx])
    buckets = pd.cut(test_sim, bins=SIM_BINS, labels=SIM_LABELS, include_lowest=True, right=False).astype(str)

    rows = []
    for model_name, probs in {"rf_morgan": base_prob, "cliff_weighted_rf": weighted_prob}.items():
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "model": model_name,
                "group": "overall",
                "n": int(len(test_idx)),
                "mean_train_cliff_weight": float(np.mean(sample_weight)),
                **safe_metrics(y[test_idx], probs),
            }
        )
        for group_name, mask in [("activity_cliff", activity_cliff), ("non_cliff", ~activity_cliff)]:
            if mask.sum() < 2:
                continue
            rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": model_name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "mean_train_cliff_weight": float(np.mean(sample_weight)),
                    **safe_metrics(y[test_idx][mask], probs[mask]),
                }
            )
        for bucket in SIM_LABELS:
            mask = np.asarray(buckets == bucket)
            if mask.sum() < 2:
                continue
            rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": model_name,
                    "group": f"sim_{bucket}",
                    "n": int(mask.sum()),
                    "mean_train_cliff_weight": float(np.mean(sample_weight)),
                    **safe_metrics(y[test_idx][mask], probs[mask]),
                }
            )

    sample_df = pd.DataFrame(
        {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "smiles": df.iloc[test_idx]["smiles"].to_numpy(),
            "y_true": y[test_idx],
            "nearest_train_similarity": test_sim,
            "nearest_train_label": nearest_label,
            "is_activity_cliff": activity_cliff,
            "rf_prob": base_prob,
            "cliff_weighted_prob": weighted_prob,
        }
    )
    return rows, sample_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "activity_cliff_targeted")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    samples = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}", flush=True)
        task_rows, sample_df = run_task(task, seed=args.seed, n_jobs=args.n_jobs, alpha=args.alpha)
        rows.extend(task_rows)
        samples.append(sample_df)
    pd.DataFrame(rows).to_csv(args.output_dir / "activity_cliff_targeted_metrics.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(args.output_dir / "activity_cliff_targeted_predictions.csv", index=False)
    summary = {"tasks": len(TASKS), "rows": len(rows), "alpha": args.alpha}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

