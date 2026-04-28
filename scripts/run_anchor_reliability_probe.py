from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.datasets import load_task_frame
from admet_shift_reliability.anchor_reliability import (
    compute_anchor_features,
    error_detection_auroc,
    risk_coverage_auc,
)
from admet_shift_reliability.features import morgan_fingerprint_matrix
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


TASKS = [
    {"dataset": "bbbp", "path": ROOT / "data" / "raw" / "BBBP.csv", "smiles_col": "smiles", "label": "p_np"},
    {"dataset": "clintox", "path": ROOT / "data" / "raw" / "clintox.csv.gz", "smiles_col": "smiles", "label": "CT_TOX"},
    {"dataset": "tox21", "path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "label": "NR-AhR"},
    {"dataset": "tox21", "path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "label": "SR-MMP"},
    {"dataset": "ames", "source": "tdc_tox", "tdc_name": "AMES", "cache_path": ROOT / "data" / "raw" / "AMES_tdc.csv.gz", "label": "AMES"},
    {"dataset": "herg", "source": "tdc_tox", "tdc_name": "hERG", "cache_path": ROOT / "data" / "raw" / "hERG_tdc.csv.gz", "label": "hERG"},
    {"dataset": "dili", "source": "tdc_tox", "tdc_name": "DILI", "cache_path": ROOT / "data" / "raw" / "DILI_tdc.csv.gz", "label": "DILI"},
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


def load_probe_frame(task_cfg: dict) -> pd.DataFrame:
    frame = load_task_frame(task_cfg)
    frame = frame.dropna().copy()
    frame["label"] = frame["label"].astype(int)
    return frame


def fit_rf(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        n_jobs=max(1, min(192, (os.cpu_count() or 8) - 8)),
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_logreg_classifier(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression | None:
    if np.unique(y_train).size < 2:
        return None
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def build_predictive_features(
    rf_prob: np.ndarray,
    anchor_feat: dict[str, np.ndarray],
) -> np.ndarray:
    anchor_prob = anchor_feat["anchor_prob"]
    return np.column_stack(
        [
            rf_prob,
            anchor_prob,
            np.abs(rf_prob - anchor_prob),
            anchor_feat["anchor_disagreement"],
            anchor_feat["anchor_distance_mean"],
            anchor_feat["anchor_distance_min"],
            anchor_feat["anchor_neighbor_label_mean"],
        ]
    ).astype(np.float32)


def build_reliability_features(
    meta_prob: np.ndarray,
    predictive_features: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            meta_prob,
            np.abs(meta_prob - 0.5) * 2.0,
            predictive_features,
        ]
    ).astype(np.float32)


def heuristic_anchor_confidence(
    meta_prob: np.ndarray,
    rf_prob: np.ndarray,
    anchor_feat: dict[str, np.ndarray],
) -> np.ndarray:
    score = 1.0 - (
        0.4 * anchor_feat["anchor_disagreement"]
        + 0.4 * np.abs(rf_prob - anchor_feat["anchor_prob"])
        + 0.2 * anchor_feat["anchor_distance_mean"]
    )
    score = np.clip(score, 0.0, 1.0)
    return 0.5 * score + 0.5 * (np.abs(meta_prob - 0.5) * 2.0)


def maybe_predict(model: LogisticRegression | None, x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if model is None:
        return fallback
    return model.predict_proba(x)[:, 1]


def run_task(task_cfg: dict, seed: int = 42) -> dict[str, float | int | str]:
    df = load_probe_frame(task_cfg)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    x = morgan_fingerprint_matrix(df["smiles"].tolist(), radius=2, n_bits=2048)
    x_bool = x.astype(bool)

    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        return {
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "status": f"skipped_{reason}",
        }

    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    train_base_idx, train_meta_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=seed,
        stratify=y[train_idx],
    )

    x_base = x[train_base_idx]
    x_base_bool = x_bool[train_base_idx]
    y_base = y[train_base_idx]

    rf = fit_rf(x_base, y_base, seed=seed)

    rf_meta = rf.predict_proba(x[train_meta_idx])[:, 1]
    rf_valid = rf.predict_proba(x[valid_idx])[:, 1]
    rf_test = rf.predict_proba(x[test_idx])[:, 1]

    anchor_meta = compute_anchor_features(x_base_bool, y_base, x_bool[train_meta_idx], n_neighbors=15)
    anchor_valid = compute_anchor_features(x_base_bool, y_base, x_bool[valid_idx], n_neighbors=15)
    anchor_test = compute_anchor_features(x_base_bool, y_base, x_bool[test_idx], n_neighbors=15)

    x_meta_features = build_predictive_features(rf_meta, anchor_meta)
    x_valid_features = build_predictive_features(rf_valid, anchor_valid)
    x_test_features = build_predictive_features(rf_test, anchor_test)

    predictive_meta_model = fit_logreg_classifier(x_meta_features, y[train_meta_idx], seed=seed)
    meta_valid = maybe_predict(predictive_meta_model, x_valid_features, fallback=0.5 * rf_valid + 0.5 * anchor_valid["anchor_prob"])
    meta_test = maybe_predict(predictive_meta_model, x_test_features, fallback=0.5 * rf_test + 0.5 * anchor_test["anchor_prob"])

    valid_correct = ((meta_valid >= 0.5).astype(np.int64) == y[valid_idx]).astype(np.int64)
    reliability_valid_x = build_reliability_features(meta_valid, x_valid_features)
    reliability_test_x = build_reliability_features(meta_test, x_test_features)
    reliability_model = fit_logreg_classifier(reliability_valid_x, valid_correct, seed=seed)

    learned_conf_test = maybe_predict(
        reliability_model,
        reliability_test_x,
        fallback=np.abs(meta_test - 0.5) * 2.0,
    )
    margin_conf_test = np.abs(meta_test - 0.5) * 2.0
    heuristic_conf_test = heuristic_anchor_confidence(meta_prob=meta_test, rf_prob=rf_test, anchor_feat=anchor_test)

    row: dict[str, float | int | str] = {
        "dataset": task_cfg["dataset"],
        "label": task_cfg["label"],
        "status": "ok",
        "n_samples": len(df),
        "n_positive": int(y.sum()),
        "train_base_size": int(len(train_base_idx)),
        "train_meta_size": int(len(train_meta_idx)),
        "valid_size": int(len(valid_idx)),
        "test_size": int(len(test_idx)),
        "valid_correct_rate": float(np.mean(valid_correct)),
    }

    for prefix, probs in (
        ("rf", rf_test),
        ("anchor", anchor_test["anchor_prob"]),
        ("meta", meta_test),
    ):
        for key, value in evaluate_probs(y[test_idx], probs).items():
            row[f"{prefix}_{key}"] = value

    for prefix, conf in (
        ("margin", margin_conf_test),
        ("heuristic", heuristic_conf_test),
        ("learned", learned_conf_test),
    ):
        row[f"{prefix}_error_detection_auroc"] = error_detection_auroc(y[test_idx], meta_test, conf)
        row[f"{prefix}_risk_coverage_auc"] = risk_coverage_auc(y[test_idx], meta_test, conf)

    return row


def main() -> None:
    output_dir = ROOT / "outputs" / "anchor_reliability_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_cfg in TASKS:
        print(f"RUN {task_cfg['dataset']}::{task_cfg['label']}")
        rows.append(run_task(task_cfg))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_rows": len(df),
                "num_success": int((df["status"] == "ok").sum()) if not df.empty else 0,
            },
            indent=2,
        )
    )
    if not df.empty:
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

