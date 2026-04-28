from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    error_detection_auroc,
    evaluate_probs,
    filter_valid_smiles,
    fit_rf,
    load_task_frame,
    make_scaffold_split,
    risk_coverage_auc,
)


SIM_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
SIM_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">=0.95"]


def nearest_train_similarity(x_train_bool: np.ndarray, x_query_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=1)
    nn.fit(x_train_bool)
    distance, index = nn.kneighbors(x_query_bool, return_distance=True)
    return 1.0 - distance[:, 0], index[:, 0]


def fit_reasoner(valid_prob: np.ndarray, valid_anchor: dict[str, np.ndarray], valid_y: np.ndarray, seed: int) -> LogisticRegression:
    x_valid = np.column_stack(
        [
            valid_prob,
            valid_anchor["anchor_prob"],
            np.abs(valid_prob - valid_anchor["anchor_prob"]),
            valid_anchor["anchor_disagreement"],
            valid_anchor["anchor_distance_mean"],
        ]
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def apply_reasoner(model: LogisticRegression, prob: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    x = np.column_stack(
        [
            prob,
            anchor["anchor_prob"],
            np.abs(prob - anchor["anchor_prob"]),
            anchor["anchor_disagreement"],
            anchor["anchor_distance_mean"],
        ]
    )
    return model.predict_proba(x)[:, 1]


def confidence_features(prob: np.ndarray, anchor: dict[str, np.ndarray], max_sim: np.ndarray) -> np.ndarray:
    margin = np.abs(prob - 0.5) * 2.0
    return np.column_stack(
        [
            prob,
            margin,
            anchor["anchor_prob"],
            np.abs(prob - anchor["anchor_prob"]),
            anchor["anchor_disagreement"],
            anchor["anchor_distance_mean"],
            anchor["anchor_distance_min"],
            max_sim,
        ]
    )


def fit_error_model(x_valid: np.ndarray, valid_prob: np.ndarray, valid_y: np.ndarray, seed: int) -> LogisticRegression | None:
    valid_correct = ((valid_prob >= 0.5).astype(int) == valid_y).astype(int)
    if np.unique(valid_correct).size < 2:
        return None
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_correct)
    return model


def evaluate_confidence(y_true: np.ndarray, probs: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    return {
        "error_detection_auroc": error_detection_auroc(y_true, probs, confidence),
        "risk_coverage_auc": risk_coverage_auc(y_true, probs, confidence),
        "selective_error_80": selective_error(y_true, probs, confidence, coverage=0.8),
        "selective_error_50": selective_error(y_true, probs, confidence, coverage=0.5),
    }


def selective_error(y_true: np.ndarray, probs: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    n_keep = max(1, int(np.ceil(len(y_true) * coverage)))
    keep = np.argsort(-confidence)[:n_keep]
    pred = (probs[keep] >= 0.5).astype(int)
    return float(np.mean(pred != y_true[keep]))


def safe_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan, "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan}
    return evaluate_probs(y_true, probs)


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (probs >= left) & (probs <= right) if idx == n_bins - 1 else (probs >= left) & (probs < right)
        if np.any(mask):
            value += abs(float(np.mean(y_true[mask])) - float(np.mean(probs[mask]))) * (int(np.sum(mask)) / len(y_true))
    return float(value)


def bootstrap_delta(y_true: np.ndarray, base_prob: np.ndarray, new_prob: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = {"auprc": [], "auroc": [], "brier": [], "ece": []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y_true[idx]).size < 2:
            continue
        rows["auprc"].append(average_precision_score(y_true[idx], new_prob[idx]) - average_precision_score(y_true[idx], base_prob[idx]))
        rows["auroc"].append(roc_auc_score(y_true[idx], new_prob[idx]) - roc_auc_score(y_true[idx], base_prob[idx]))
        rows["brier"].append(brier_score_loss(y_true[idx], base_prob[idx]) - brier_score_loss(y_true[idx], new_prob[idx]))
        rows["ece"].append(ece(y_true[idx], base_prob[idx]) - ece(y_true[idx], new_prob[idx]))
    out: dict[str, float] = {}
    for metric, values in rows.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            out[f"{metric}_delta_mean"] = np.nan
            out[f"{metric}_delta_ci_low"] = np.nan
            out[f"{metric}_delta_ci_high"] = np.nan
            out[f"{metric}_p_delta_gt0"] = np.nan
        else:
            out[f"{metric}_delta_mean"] = float(np.mean(arr))
            out[f"{metric}_delta_ci_low"] = float(np.quantile(arr, 0.025))
            out[f"{metric}_delta_ci_high"] = float(np.quantile(arr, 0.975))
            out[f"{metric}_p_delta_gt0"] = float(np.mean(arr > 0))
    return out


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int, n_boot: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
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

    anchor_valid = compute_anchor_features(xb[train_idx], y[train_idx], xb[valid_idx], n_neighbors=15)
    anchor_test = compute_anchor_features(xb[train_idx], y[train_idx], xb[test_idx], n_neighbors=15)
    reasoner = fit_reasoner(rf_valid, anchor_valid, y[valid_idx], seed=seed)
    reasoning_valid = apply_reasoner(reasoner, rf_valid, anchor_valid)
    reasoning_test = apply_reasoner(reasoner, rf_test, anchor_test)

    valid_sim, _ = nearest_train_similarity(xb[train_idx], xb[valid_idx])
    test_sim, test_nn = nearest_train_similarity(xb[train_idx], xb[test_idx])
    nearest_label = y[train_idx][test_nn]
    activity_cliff = (test_sim >= 0.7) & (nearest_label != y[test_idx])
    buckets = pd.cut(test_sim, bins=SIM_BINS, labels=SIM_LABELS, include_lowest=True, right=False).astype(str)

    leakage_rows = [
        {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "train_size": int(len(train_idx)),
            "valid_size": int(len(valid_idx)),
            "test_size": int(len(test_idx)),
            "max_train_similarity_mean": float(np.mean(test_sim)),
            "max_train_similarity_p90": float(np.quantile(test_sim, 0.9)),
            "frac_similarity_ge_0_70": float(np.mean(test_sim >= 0.70)),
            "frac_similarity_ge_0_85": float(np.mean(test_sim >= 0.85)),
            "frac_similarity_ge_0_95": float(np.mean(test_sim >= 0.95)),
            "activity_cliff_frac": float(np.mean(activity_cliff)),
        }
    ]

    preds = {"rf_morgan": rf_test, "anchor_reasoning": reasoning_test}
    bucket_rows = []
    for model_name, probs in preds.items():
        for bucket in SIM_LABELS:
            mask = np.asarray(buckets == bucket)
            if mask.sum() < 2:
                continue
            bucket_rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": model_name,
                    "max_similarity_bucket": bucket,
                    "n": int(mask.sum()),
                    **safe_metrics(y[test_idx][mask], probs[mask]),
                }
            )
        for group_name, mask in [("activity_cliff", activity_cliff), ("non_cliff", ~activity_cliff)]:
            if mask.sum() < 2:
                continue
            bucket_rows.append(
                {
                    "dataset": task_cfg["dataset"],
                    "label": task_cfg["label"],
                    "model": model_name,
                    "max_similarity_bucket": group_name,
                    "n": int(mask.sum()),
                    **safe_metrics(y[test_idx][mask], probs[mask]),
                }
            )

    x_rel_valid = confidence_features(reasoning_valid, anchor_valid, valid_sim)
    x_rel_test = confidence_features(reasoning_test, anchor_test, test_sim)
    error_model = fit_error_model(x_rel_valid, reasoning_valid, y[valid_idx], seed=seed)
    margin_conf = np.abs(reasoning_test - 0.5) * 2.0
    anchor_conf = np.clip(1.0 - 0.5 * anchor_test["anchor_disagreement"] - 0.5 * anchor_test["anchor_distance_mean"], 0.0, 1.0)
    learned_conf = error_model.predict_proba(x_rel_test)[:, 1] if error_model is not None else margin_conf

    uq_rows = []
    for conf_name, confidence in {
        "prob_margin": margin_conf,
        "anchor_distance_disagreement": anchor_conf,
        "learned_shift_error_model": learned_conf,
    }.items():
        uq_rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "model": "anchor_reasoning",
                "confidence": conf_name,
                **evaluate_confidence(y[test_idx], reasoning_test, confidence),
            }
        )

    bootstrap_rows = [
        {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "comparison": "anchor_reasoning_vs_rf_morgan",
            **bootstrap_delta(y[test_idx], rf_test, reasoning_test, n_boot=n_boot, seed=seed),
        }
    ]
    return leakage_rows, bucket_rows, uq_rows, bootstrap_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "leakage_shift_uq_bootstrap_20260422")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    leakage_rows: list[dict] = []
    bucket_rows: list[dict] = []
    uq_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}", flush=True)
        leakage, buckets, uq, boot = run_task(task, seed=args.seed, rf_n_jobs=args.rf_n_jobs, n_boot=args.n_boot)
        leakage_rows.extend(leakage)
        bucket_rows.extend(buckets)
        uq_rows.extend(uq)
        bootstrap_rows.extend(boot)

    pd.DataFrame(leakage_rows).to_csv(args.output_dir / "leakage_summary.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(args.output_dir / "shift_bucket_metrics.csv", index=False)
    pd.DataFrame(uq_rows).to_csv(args.output_dir / "shift_uq_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "paired_bootstrap_anchor_vs_rf.csv", index=False)
    summary = {
        "tasks": len(TASKS),
        "leakage_rows": len(leakage_rows),
        "bucket_rows": len(bucket_rows),
        "uq_rows": len(uq_rows),
        "bootstrap_rows": len(bootstrap_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

