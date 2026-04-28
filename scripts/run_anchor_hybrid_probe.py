from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")


TASKS = [
    {"dataset": "bbbp", "path": ROOT / "data" / "raw" / "BBBP.csv", "smiles_col": "smiles", "label": "p_np"},
    {"dataset": "clintox", "path": ROOT / "data" / "raw" / "clintox.csv.gz", "smiles_col": "smiles", "label": "CT_TOX"},
    {"dataset": "tox21", "path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "label": "NR-AhR"},
    {"dataset": "tox21", "path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "label": "SR-MMP"},
    {"dataset": "ames", "path": ROOT / "data" / "raw" / "AMES_tdc.csv.gz", "smiles_col": "Drug", "label": "AMES", "raw_label_col": "Y"},
    {"dataset": "herg", "path": ROOT / "data" / "raw" / "hERG_tdc.csv.gz", "smiles_col": "Drug", "label": "hERG", "raw_label_col": "Y"},
    {"dataset": "dili", "path": ROOT / "data" / "raw" / "DILI_tdc.csv.gz", "smiles_col": "Drug", "label": "DILI", "raw_label_col": "Y"},
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


def fit_knn(x_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    model = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="jaccard",
        n_jobs=max(1, min(64, (os.cpu_count() or 8) // 2)),
    )
    model.fit(x_train.astype(bool), y_train)
    return model


def tune_alpha(y_valid: np.ndarray, rf_valid: np.ndarray, knn_valid: np.ndarray) -> tuple[float, dict[str, float]]:
    best_alpha = 0.0
    best_metrics = None
    best_key = None
    for alpha in np.linspace(0.0, 1.0, 21):
        probs = alpha * rf_valid + (1.0 - alpha) * knn_valid
        metrics = evaluate_probs(y_valid, probs)
        key = (metrics["auprc"], metrics["auroc"], -metrics["brier"])
        if best_key is None or key > best_key:
            best_key = key
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics


def run_task(task_cfg: dict, seed: int = 42) -> list[dict]:
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
        return [{
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "status": f"skipped_{reason}",
        }]

    x = morgan_fingerprint_matrix(df["smiles"].tolist(), radius=2, n_bits=2048)
    x_train, y_train = x[split["train"]], y[split["train"]]
    x_valid, y_valid = x[split["valid"]], y[split["valid"]]
    x_test, y_test = x[split["test"]], y[split["test"]]

    rf = fit_rf(x_train, y_train, seed=seed)
    knn = fit_knn(x_train, y_train)

    rf_valid = rf.predict_proba(x_valid)[:, 1]
    knn_valid = knn.predict_proba(x_valid.astype(bool))[:, 1]
    alpha, valid_metrics = tune_alpha(y_valid, rf_valid, knn_valid)

    rf_test = rf.predict_proba(x_test)[:, 1]
    knn_test = knn.predict_proba(x_test.astype(bool))[:, 1]
    hybrid_test = alpha * rf_test + (1.0 - alpha) * knn_test

    rows = []
    for model_name, probs in (("rf", rf_test), ("knn", knn_test), ("hybrid", hybrid_test)):
        rows.append({
            "dataset": task_cfg["dataset"],
            "label": task_cfg["label"],
            "split": "scaffold",
            "model": model_name,
            "alpha_rf": alpha if model_name == "hybrid" else np.nan,
            "valid_auprc_at_alpha": valid_metrics["auprc"] if model_name == "hybrid" else np.nan,
            "n_samples": len(df),
            "n_positive": int(y.sum()),
            "train_size": len(split["train"]),
            "valid_size": len(split["valid"]),
            "test_size": len(split["test"]),
            **evaluate_probs(y_test, probs),
        })
    return rows


def main() -> None:
    output_dir = ROOT / "outputs" / "anchor_hybrid_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task_cfg in TASKS:
        print(f"RUN {task_cfg['dataset']}::{task_cfg['label']}")
        all_rows.extend(run_task(task_cfg))

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps({"num_rows": len(df)}, indent=2))

    if not df.empty:
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

