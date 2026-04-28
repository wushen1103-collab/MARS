from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    fit_rf,
    load_task_frame,
    make_scaffold_split,
    split_is_usable,
)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    if scores.size == 0:
        return float("inf")
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), scores.size)
    return float(scores[rank - 1])


def conformal_binary_metrics(valid_y: np.ndarray, valid_prob: np.ndarray, test_y: np.ndarray, test_prob: np.ndarray, alpha: float) -> dict[str, float]:
    valid_scores = np.where(valid_y == 1, 1.0 - valid_prob, valid_prob)
    qhat = conformal_quantile(valid_scores, alpha=alpha)
    include_zero = test_prob <= qhat
    include_one = (1.0 - test_prob) <= qhat
    set_size = include_zero.astype(int) + include_one.astype(int)
    true_in_set = np.where(test_y == 1, include_one, include_zero)
    singleton = set_size == 1
    singleton_pred = np.where(include_one & ~include_zero, 1, 0)
    singleton_correct = singleton_pred[singleton] == test_y[singleton] if np.any(singleton) else np.asarray([], dtype=bool)
    return {
        "alpha": float(alpha),
        "qhat": float(qhat),
        "coverage": float(np.mean(true_in_set)),
        "target_coverage": float(1.0 - alpha),
        "avg_set_size": float(np.mean(set_size)),
        "singleton_rate": float(np.mean(singleton)),
        "ambiguous_rate": float(np.mean(set_size == 2)),
        "empty_rate": float(np.mean(set_size == 0)),
        "singleton_error": float(1.0 - np.mean(singleton_correct)) if singleton_correct.size else np.nan,
    }


def wilson_upper(errors: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    phat = errors / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    radius = z * math.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n)))
    return float((centre + radius) / denom)


def risk_control_threshold(valid_y: np.ndarray, valid_prob: np.ndarray, target_risk: float) -> tuple[float, dict[str, float]]:
    conf = np.maximum(valid_prob, 1.0 - valid_prob)
    pred = (valid_prob >= 0.5).astype(int)
    error = (pred != valid_y).astype(int)
    best_threshold = 1.01
    best_coverage = 0.0
    best_empirical = np.nan
    best_upper = np.nan
    for threshold in np.unique(np.sort(conf)):
        keep = conf >= threshold
        n_keep = int(np.sum(keep))
        if n_keep == 0:
            continue
        n_err = int(np.sum(error[keep]))
        upper = wilson_upper(n_err, n_keep)
        if upper <= target_risk and n_keep / len(valid_y) >= best_coverage:
            best_threshold = float(threshold)
            best_coverage = float(n_keep / len(valid_y))
            best_empirical = float(n_err / n_keep)
            best_upper = float(upper)
    return best_threshold, {
        "valid_target_risk": float(target_risk),
        "valid_selected_coverage": float(best_coverage),
        "valid_selected_empirical_risk": float(best_empirical),
        "valid_selected_wilson_upper": float(best_upper),
    }


def risk_control_metrics(valid_y: np.ndarray, valid_prob: np.ndarray, test_y: np.ndarray, test_prob: np.ndarray, target_risk: float) -> dict[str, float]:
    threshold, valid_stats = risk_control_threshold(valid_y, valid_prob, target_risk=target_risk)
    test_conf = np.maximum(test_prob, 1.0 - test_prob)
    keep = test_conf >= threshold
    pred = (test_prob >= 0.5).astype(int)
    n_keep = int(np.sum(keep))
    out = {
        "target_risk": float(target_risk),
        "threshold": float(threshold),
        "test_coverage": float(n_keep / len(test_y)),
        "test_error": float(np.mean(pred[keep] != test_y[keep])) if n_keep else np.nan,
        "test_kept": n_keep,
        "test_total": int(len(test_y)),
    }
    out.update(valid_stats)
    return out


def classwise_ece(y_true: np.ndarray, prob_one: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    values = []
    out = {}
    for class_id, class_prob in [(0, 1.0 - prob_one), (1, prob_one)]:
        class_true = (y_true == class_id).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for idx in range(n_bins):
            left, right = bins[idx], bins[idx + 1]
            mask = (class_prob >= left) & (class_prob <= right) if idx == n_bins - 1 else (class_prob >= left) & (class_prob < right)
            if np.any(mask):
                ece += abs(float(np.mean(class_true[mask])) - float(np.mean(class_prob[mask]))) * (int(np.sum(mask)) / len(y_true))
        out[f"class_{class_id}_ece"] = float(ece)
        values.append(ece)
    out["classwise_ece"] = float(np.mean(values))
    return out


def adaptive_ece(y_true: np.ndarray, prob_one: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(prob_one, kind="mergesort")
    chunks = np.array_split(order, min(n_bins, len(order)))
    value = 0.0
    for chunk in chunks:
        if len(chunk) > 0:
            value += abs(float(np.mean(y_true[chunk])) - float(np.mean(prob_one[chunk]))) * (len(chunk) / len(y_true))
    return float(value)


def fit_reasoner(valid_prob: np.ndarray, valid_anchor: dict[str, np.ndarray], valid_y: np.ndarray, seed: int) -> LogisticRegression | None:
    if np.unique(valid_y).size < 2:
        return None
    x_valid = np.column_stack([valid_prob, valid_anchor["anchor_prob"], np.abs(valid_prob - valid_anchor["anchor_prob"]), valid_anchor["anchor_disagreement"], valid_anchor["anchor_distance_mean"]])
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def apply_reasoner(model: LogisticRegression | None, prob: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    if model is None:
        return np.clip(0.5 * prob + 0.5 * anchor["anchor_prob"], 0.0, 1.0)
    x = np.column_stack([prob, anchor["anchor_prob"], np.abs(prob - anchor["anchor_prob"]), anchor["anchor_disagreement"], anchor["anchor_distance_mean"]])
    return model.predict_proba(x)[:, 1]


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int, alphas: list[float], target_risks: list[float]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        return ([{"dataset": task_cfg["dataset"], "label": task_cfg["label"], "status": f"skipped_{reason}"}], [], [], [])
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

    models = {"rf_morgan": (rf_valid, rf_test), "retrieval_only": (anchor_valid["anchor_prob"], anchor_test["anchor_prob"]), "anchor_reasoning": (reasoning_valid, reasoning_test)}
    base_rows: list[dict] = []
    conformal_rows: list[dict] = []
    risk_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for model_name, (valid_prob, test_prob) in models.items():
        base_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "model": model_name, "split": "scaffold", "seed": seed, "status": "ok", "valid_size": int(len(valid_idx)), "test_size": int(len(test_idx)), **evaluate_probs(y[test_idx], test_prob)})
        calibration_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "model": model_name, "split": "scaffold", "seed": seed, "adaptive_ece": adaptive_ece(y[test_idx], test_prob), **classwise_ece(y[test_idx], test_prob)})
        for alpha in alphas:
            conformal_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "model": model_name, "split": "scaffold", "seed": seed, **conformal_binary_metrics(y[valid_idx], valid_prob, y[test_idx], test_prob, alpha=alpha)})
        for target_risk in target_risks:
            risk_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "model": model_name, "split": "scaffold", "seed": seed, **risk_control_metrics(y[valid_idx], valid_prob, y[test_idx], test_prob, target_risk=target_risk)})
    return base_rows, conformal_rows, risk_rows, calibration_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--alphas", default="0.05,0.1,0.2")
    parser.add_argument("--target-risks", default="0.05,0.1,0.2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "conformal_risk_control_20260422")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    target_risks = [float(item) for item in args.target_risks.split(",") if item.strip()]

    base_rows: list[dict] = []
    conformal_rows: list[dict] = []
    risk_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}", flush=True)
        base, conformal, risk, calibration = run_task(task, args.seed, args.rf_n_jobs, alphas, target_risks)
        base_rows.extend(base)
        conformal_rows.extend(conformal)
        risk_rows.extend(risk)
        calibration_rows.extend(calibration)

    pd.DataFrame(base_rows).to_csv(args.output_dir / "base_model_metrics.csv", index=False)
    pd.DataFrame(conformal_rows).to_csv(args.output_dir / "conformal_set_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(args.output_dir / "risk_control_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.output_dir / "classwise_calibration_metrics.csv", index=False)
    summary = {"base_rows": len(base_rows), "conformal_rows": len(conformal_rows), "risk_rows": len(risk_rows), "calibration_rows": len(calibration_rows), "tasks": len(TASKS), "alphas": alphas, "target_risks": target_risks}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
