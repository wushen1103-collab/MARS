from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.anchor_reliability import (
    compute_anchor_features,
    error_detection_auroc,
    risk_coverage_auc,
)
from admet_shift_reliability.datasets import load_task_frame
from admet_shift_reliability.features import morgan_fingerprint_matrix
from admet_shift_reliability.reliability_benchmark import (
    binary_entropy_confidence,
    resolve_rf_n_jobs,
    selective_error_at_coverage,
)
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


TASKS = [
    {
        "dataset": "bbbp",
        "source": "csv",
        "path": ROOT / "data" / "raw" / "BBBP.csv",
        "smiles_col": "smiles",
        "label": "p_np",
    },
    {
        "dataset": "clintox",
        "source": "csv",
        "path": ROOT / "data" / "raw" / "clintox.csv.gz",
        "smiles_col": "smiles",
        "label": "CT_TOX",
    },
    {
        "dataset": "tox21",
        "source": "csv",
        "path": ROOT / "data" / "raw" / "tox21.csv.gz",
        "smiles_col": "smiles",
        "label": "NR-AhR",
    },
    {
        "dataset": "tox21",
        "source": "csv",
        "path": ROOT / "data" / "raw" / "tox21.csv.gz",
        "smiles_col": "smiles",
        "label": "SR-MMP",
    },
    {
        "dataset": "ames",
        "source": "tdc_tox",
        "tdc_name": "AMES",
        "cache_path": ROOT / "data" / "raw" / "AMES_tdc.csv.gz",
        "label": "AMES",
    },
    {
        "dataset": "herg",
        "source": "tdc_tox",
        "tdc_name": "hERG",
        "cache_path": ROOT / "data" / "raw" / "hERG_tdc.csv.gz",
        "label": "hERG",
    },
    {
        "dataset": "dili",
        "source": "tdc_tox",
        "tdc_name": "DILI",
        "cache_path": ROOT / "data" / "raw" / "DILI_tdc.csv.gz",
        "label": "DILI",
    },
]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
        "positive_rate": float(np.mean(y_true)),
    }


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    valid = []
    for smi in df[smiles_col].astype(str):
        valid.append(Chem.MolFromSmiles(smi) is not None)
    out = df.loc[valid].copy()
    out = out.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
    return out


def make_scaffold_split(smiles: list[str]) -> dict[str, list[int]]:
    return BemisMurckoScaffoldSplitter(valid_frac=0.1, test_frac=0.2).split(smiles)


def split_is_usable(split: dict[str, list[int]], y: np.ndarray) -> tuple[bool, str]:
    for part_name in ("train", "valid", "test"):
        idx = split[part_name]
        if len(idx) == 0:
            return False, f"{part_name}_empty"
        values = y[idx]
        if part_name == "train" and len(np.unique(values)) < 2:
            return False, "train_single_class"
        if part_name in {"valid", "test"} and len(np.unique(values)) < 2:
            return False, f"{part_name}_single_class"
    return True, "ok"


def fit_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    rf_n_jobs: int | None = None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        n_jobs=resolve_rf_n_jobs(requested=rf_n_jobs, cpu_count=os.cpu_count()),
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def positive_class_prob_from_estimator(estimator, x: np.ndarray) -> np.ndarray:
    probs = estimator.predict_proba(x)
    if probs.shape[1] == 2:
        class_to_col = {cls: idx for idx, cls in enumerate(estimator.classes_)}
        return probs[:, class_to_col[1]]
    if estimator.classes_[0] == 1:
        return np.ones(x.shape[0], dtype=np.float64)
    return np.zeros(x.shape[0], dtype=np.float64)


def rf_tree_confidence(rf: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    tree_probs = np.stack([positive_class_prob_from_estimator(tree, x) for tree in rf.estimators_], axis=0)
    tree_std = np.std(tree_probs, axis=0)
    conf = 1.0 - np.clip(tree_std / 0.5, 0.0, 1.0)
    return conf.astype(np.float32)


def maybe_fit_reliability_model(x_valid: np.ndarray, y_valid_correct: np.ndarray, seed: int) -> LogisticRegression | None:
    if np.unique(y_valid_correct).size < 2:
        return None
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_valid, y_valid_correct)
    return model


def heuristic_anchor_confidence(
    rf_prob: np.ndarray,
    anchor_feat: dict[str, np.ndarray],
) -> np.ndarray:
    score = 1.0 - (
        0.4 * anchor_feat["anchor_disagreement"]
        + 0.4 * np.abs(rf_prob - anchor_feat["anchor_prob"])
        + 0.2 * anchor_feat["anchor_distance_mean"]
    )
    score = np.clip(score, 0.0, 1.0)
    margin = np.abs(rf_prob - 0.5) * 2.0
    return 0.5 * score + 0.5 * margin


def reliability_feature_block(
    rf_prob: np.ndarray,
    margin_conf: np.ndarray,
    entropy_conf: np.ndarray,
    tree_conf: np.ndarray,
    anchor_feat: dict[str, np.ndarray],
) -> np.ndarray:
    return np.column_stack(
        [
            rf_prob,
            margin_conf,
            entropy_conf,
            tree_conf,
            anchor_feat["anchor_prob"],
            np.abs(rf_prob - anchor_feat["anchor_prob"]),
            anchor_feat["anchor_disagreement"],
            anchor_feat["anchor_distance_mean"],
            anchor_feat["anchor_distance_min"],
            anchor_feat["anchor_neighbor_label_mean"],
        ]
    ).astype(np.float32)


def run_task(task_cfg: dict, seed: int = 42, rf_n_jobs: int | None = None) -> list[dict]:
    df = load_task_frame(task_cfg)
    df = df.dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")

    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        return [{
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "method": "skipped",
            "status": f"skipped_{reason}",
        }]

    x = morgan_fingerprint_matrix(df["smiles"].tolist(), radius=2, n_bits=2048)
    x_bool = x.astype(bool)

    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    rf_valid = rf.predict_proba(x[valid_idx])[:, 1]
    rf_test = rf.predict_proba(x[test_idx])[:, 1]

    margin_valid = np.abs(rf_valid - 0.5) * 2.0
    margin_test = np.abs(rf_test - 0.5) * 2.0
    entropy_valid = binary_entropy_confidence(rf_valid)
    entropy_test = binary_entropy_confidence(rf_test)
    tree_valid = rf_tree_confidence(rf, x[valid_idx])
    tree_test = rf_tree_confidence(rf, x[test_idx])

    anchor_valid = compute_anchor_features(x_bool[train_idx], y[train_idx], x_bool[valid_idx], n_neighbors=15)
    anchor_test = compute_anchor_features(x_bool[train_idx], y[train_idx], x_bool[test_idx], n_neighbors=15)
    heuristic_valid = heuristic_anchor_confidence(rf_valid, anchor_valid)
    heuristic_test = heuristic_anchor_confidence(rf_test, anchor_test)

    valid_correct = ((rf_valid >= 0.5).astype(np.int64) == y[valid_idx]).astype(np.int64)
    x_rel_valid = reliability_feature_block(rf_valid, margin_valid, entropy_valid, tree_valid, anchor_valid)
    x_rel_test = reliability_feature_block(rf_test, margin_test, entropy_test, tree_test, anchor_test)
    rel_model = maybe_fit_reliability_model(x_rel_valid, valid_correct, seed=seed)
    learned_test = rel_model.predict_proba(x_rel_test)[:, 1] if rel_model is not None else margin_test

    methods = {
        "margin": margin_test,
        "entropy": entropy_test,
        "tree_conf": tree_test,
        "anchor_heuristic": heuristic_test,
        "learned": learned_test,
    }

    base_metrics = evaluate_probs(y[test_idx], rf_test)
    rows = []
    for method_name, confidence in methods.items():
        row = {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "method": method_name,
            "status": "ok",
            "n_samples": len(df),
            "n_positive": int(y.sum()),
            "train_size": int(len(train_idx)),
            "valid_size": int(len(valid_idx)),
            "test_size": int(len(test_idx)),
            "base_auroc": base_metrics["auroc"],
            "base_auprc": base_metrics["auprc"],
            "base_brier": base_metrics["brier"],
            "base_ece": base_metrics["ece"],
            "error_detection_auroc": error_detection_auroc(y[test_idx], rf_test, confidence),
            "risk_coverage_auc": risk_coverage_auc(y[test_idx], rf_test, confidence),
        }
        for cov in (0.2, 0.5, 0.8):
            err, kept = selective_error_at_coverage(y[test_idx], rf_test, confidence, coverage=cov)
            suffix = str(cov).replace(".", "")
            row[f"selective_error_cov{suffix}"] = err
            row[f"kept_cov{suffix}"] = kept
        rows.append(row)

    return rows


def select_tasks(task_specs: list[str] | None) -> list[dict]:
    if not task_specs:
        return TASKS
    wanted = set()
    for spec in task_specs:
        if ":" not in spec:
            raise ValueError(f"Task spec must look like dataset:label, got: {spec}")
        dataset, label = spec.split(":", 1)
        wanted.add((dataset, label))
    return [task for task in TASKS if (task["dataset"], task["label"]) in wanted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reliability_benchmark")
    parser.add_argument("--task", action="append", default=None, help="dataset:label, repeatable")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tasks = select_tasks(args.task)
    for task_cfg in tasks:
        print(f"RUN {task_cfg['dataset']}::{task_cfg['label']}")
        rows.extend(run_task(task_cfg, seed=args.seed, rf_n_jobs=args.rf_n_jobs))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"results_seed{args.seed}.csv", index=False)

    summary = {}
    if not df.empty:
        ok_df = df[df["status"] == "ok"].copy()
        if not ok_df.empty:
            winners = []
            for (dataset, label), sub_df in ok_df.groupby(["dataset", "label"]):
                best_err = sub_df.sort_values("error_detection_auroc", ascending=False).iloc[0]
                best_rc = sub_df.sort_values("risk_coverage_auc", ascending=True).iloc[0]
                winners.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "best_error_detection": best_err["method"],
                        "best_risk_coverage": best_rc["method"],
                    }
                )
            summary["winners"] = winners
            print(ok_df.round(4).to_string(index=False))

    (output_dir / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

