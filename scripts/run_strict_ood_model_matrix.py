from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import (  # noqa: E402
    compute_anchor_features,
    error_detection_auroc,
    risk_coverage_auc,
)
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_leakage_shift_uq_bootstrap import nearest_train_similarity  # noqa: E402
from run_realistic_ood_splits import (  # noqa: E402
    make_fingerprint_density_split,
    make_molecular_weight_reverse_split,
    make_pca_cluster_split,
    split_usable,
)
from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    fit_rf,
    load_task_frame,
)


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (y_prob >= left) & (y_prob <= right) if idx == n_bins - 1 else (y_prob >= left) & (y_prob < right)
        if np.any(mask):
            value += abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask]))) * (int(np.sum(mask)) / len(y_true))
    return float(value)


def safe_eval(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan, "positive_rate": np.nan}
    if np.unique(y_true).size < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "brier": float(brier_score_loss(y_true, probs)),
            "ece": expected_calibration_error(y_true, probs),
            "positive_rate": float(np.mean(y_true)),
        }
    return evaluate_probs(y_true, probs)


def fit_reasoner(
    valid_prob: np.ndarray,
    valid_anchor: dict[str, np.ndarray],
    valid_y: np.ndarray,
    seed: int,
) -> LogisticRegression | None:
    if np.unique(valid_y).size < 2:
        return None
    x_valid = np.column_stack(
        [
            valid_prob,
            valid_anchor["anchor_prob"],
            np.abs(valid_prob - valid_anchor["anchor_prob"]),
            valid_anchor["anchor_disagreement"],
            valid_anchor["anchor_distance_mean"],
            valid_anchor["anchor_distance_min"],
        ]
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def apply_reasoner(model: LogisticRegression | None, prob: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    if model is None:
        return np.clip(0.5 * prob + 0.5 * anchor["anchor_prob"], 0.0, 1.0)
    x = np.column_stack(
        [
            prob,
            anchor["anchor_prob"],
            np.abs(prob - anchor["anchor_prob"]),
            anchor["anchor_disagreement"],
            anchor["anchor_distance_mean"],
            anchor["anchor_distance_min"],
        ]
    )
    return model.predict_proba(x)[:, 1]


def confidence_feature_block(prob: np.ndarray, anchor: dict[str, np.ndarray], max_sim: np.ndarray) -> np.ndarray:
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
    ).astype(np.float32)


def fit_error_model(
    valid_prob: np.ndarray,
    valid_anchor: dict[str, np.ndarray],
    valid_max_sim: np.ndarray,
    valid_y: np.ndarray,
    seed: int,
) -> LogisticRegression | None:
    valid_correct = ((valid_prob >= 0.5).astype(int) == valid_y).astype(int)
    if np.unique(valid_correct).size < 2:
        return None
    x_valid = confidence_feature_block(valid_prob, valid_anchor, valid_max_sim)
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_correct)
    return model


def confidence_metrics(y_true: np.ndarray, probs: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    return {
        "error_rate": float(np.mean(pred != y_true)),
        "error_detection_auroc": error_detection_auroc(y_true, probs, confidence),
        "risk_coverage_auc": risk_coverage_auc(y_true, probs, confidence),
        "selective_error_50": selective_error(y_true, probs, confidence, 0.5),
        "selective_error_80": selective_error(y_true, probs, confidence, 0.8),
    }


def selective_error(y_true: np.ndarray, probs: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    n_keep = max(1, int(np.ceil(len(y_true) * coverage)))
    keep = np.argsort(-confidence, kind="mergesort")[:n_keep]
    return float(np.mean((probs[keep] >= 0.5).astype(int) != y_true[keep]))


def fit_rf_ensemble(x_train: np.ndarray, y_train: np.ndarray, seeds: list[int], rf_n_jobs: int) -> list[RandomForestClassifier]:
    return [fit_rf(x_train, y_train, seed=seed, rf_n_jobs=rf_n_jobs) for seed in seeds]


def ensemble_probs(models: list[RandomForestClassifier], x: np.ndarray) -> np.ndarray:
    probs = np.stack([model.predict_proba(x)[:, 1] for model in models], axis=0)
    return probs.mean(axis=0)


def make_split(split_name: str, smiles: list[str], x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    if split_name in {"pca_cluster", "umap"}:
        return make_pca_cluster_split(x, y, seed)
    if split_name in {"fingerprint_density", "lohi"}:
        return make_fingerprint_density_split(x, y, seed)
    if split_name == "molecular_weight_reverse":
        return make_molecular_weight_reverse_split(smiles, y, seed)
    raise ValueError(f"Unknown strict OOD split: {split_name}")


def run_one(task_cfg: dict, split_name: str, seed: int, rf_n_jobs: int, ensemble_size: int) -> tuple[list[dict], list[dict]]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    smiles = df["smiles"].tolist()
    x = morgan_fingerprint_matrix(smiles)
    xb = x.astype(bool)
    try:
        split = make_split(split_name, smiles, x, y, seed)
    except Exception as exc:
        return ([{"dataset": task_cfg["dataset"], "label": task_cfg["label"], "split": split_name, "seed": seed, "model": "skipped", "status": f"split_failed:{type(exc).__name__}:{exc}"}], [])
    if not split_usable(split, y):
        return ([{"dataset": task_cfg["dataset"], "label": task_cfg["label"], "split": split_name, "seed": seed, "model": "skipped", "status": "skipped_single_class_partition"}], [])

    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    rf_valid = rf.predict_proba(x[valid_idx])[:, 1]
    rf_test = rf.predict_proba(x[test_idx])[:, 1]

    ens_seeds = [seed + 1009 * idx for idx in range(max(1, ensemble_size))]
    rf_ensemble = fit_rf_ensemble(x[train_idx], y[train_idx], seeds=ens_seeds, rf_n_jobs=rf_n_jobs)
    ens_test = ensemble_probs(rf_ensemble, x[test_idx])

    anchor_valid = compute_anchor_features(xb[train_idx], y[train_idx], xb[valid_idx], n_neighbors=15)
    anchor_test = compute_anchor_features(xb[train_idx], y[train_idx], xb[test_idx], n_neighbors=15)
    reasoner = fit_reasoner(rf_valid, anchor_valid, y[valid_idx], seed=seed)
    reasoning_valid = apply_reasoner(reasoner, rf_valid, anchor_valid)
    reasoning_test = apply_reasoner(reasoner, rf_test, anchor_test)

    valid_sim, _ = nearest_train_similarity(xb[train_idx], xb[valid_idx])
    test_sim, _ = nearest_train_similarity(xb[train_idx], xb[test_idx])
    error_model = fit_error_model(reasoning_valid, anchor_valid, valid_sim, y[valid_idx], seed=seed)
    learned_conf = (
        error_model.predict_proba(confidence_feature_block(reasoning_test, anchor_test, test_sim))[:, 1]
        if error_model is not None
        else np.abs(reasoning_test - 0.5) * 2.0
    )

    prediction_models = {
        "rf_morgan": rf_test,
        "rf_ensemble": ens_test,
        "retrieval_only": anchor_test["anchor_prob"],
        "anchor_reasoning": reasoning_test,
    }

    metric_rows: list[dict] = []
    confidence_rows: list[dict] = []
    for model_name, probs in prediction_models.items():
        metric_rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "split": split_name,
                "seed": seed,
                "model": model_name,
                "status": "ok",
                "n_samples": int(len(df)),
                "n_positive": int(y.sum()),
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "test_size": int(len(test_idx)),
                "test_max_similarity_mean": float(np.mean(test_sim)),
                "test_frac_similarity_ge_0_70": float(np.mean(test_sim >= 0.70)),
                **safe_eval(y[test_idx], probs),
            }
        )
        margin_conf = np.abs(probs - 0.5) * 2.0
        confidence_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "split": split_name, "seed": seed, "model": model_name, "confidence": "prob_margin", **confidence_metrics(y[test_idx], probs, margin_conf)})
        if model_name == "anchor_reasoning":
            confidence_rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "split": split_name, "seed": seed, "model": model_name, "confidence": "learned_shift_error_model", **confidence_metrics(y[test_idx], probs, learned_conf)})
    return metric_rows, confidence_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="fingerprint_density,molecular_weight_reverse,pca_cluster")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "strict_ood_model_matrix")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    seeds = parse_seeds(args.seeds)
    metric_rows: list[dict] = []
    confidence_rows: list[dict] = []
    for seed in seeds:
        for task in TASKS:
            for split_name in split_names:
                print(f"RUN {task['dataset']}::{task['label']} split={split_name} seed={seed}", flush=True)
                metrics, conf = run_one(task, split_name, seed, args.rf_n_jobs, args.ensemble_size)
                metric_rows.extend(metrics)
                confidence_rows.extend(conf)
                pd.DataFrame(metric_rows).to_csv(args.output_dir / "strict_ood_model_metrics.csv", index=False)
                pd.DataFrame(confidence_rows).to_csv(args.output_dir / "strict_ood_confidence_metrics.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    ok = metrics_df[metrics_df["status"] == "ok"].copy() if not metrics_df.empty else pd.DataFrame()
    summary = {
        "rows": int(len(metric_rows)),
        "confidence_rows": int(len(confidence_rows)),
        "ok_rows": int(len(ok)),
        "tasks": int(ok[["dataset", "label"]].drop_duplicates().shape[0]) if not ok.empty else 0,
        "splits": split_names,
        "seeds": seeds,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
