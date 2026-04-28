from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")


DATASETS = [
    {
        "name": "bbbp",
        "path": ROOT / "data" / "raw" / "BBBP.csv",
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
    },
    {
        "name": "clintox",
        "path": ROOT / "data" / "raw" / "clintox.csv.gz",
        "smiles_col": "smiles",
        "label_cols": ["CT_TOX", "FDA_APPROVED"],
    },
    {
        "name": "tox21",
        "path": ROOT / "data" / "raw" / "tox21.csv.gz",
        "smiles_col": "smiles",
        "label_cols": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    },
]


@dataclass
class SplitIndices:
    train: list[int]
    valid: list[int]
    test: list[int]


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


def has_valid_binary_labels(values: np.ndarray) -> bool:
    uniques = np.unique(values)
    return len(uniques) == 2 and set(uniques.tolist()) == {0, 1}


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    valid = []
    for smi in df[smiles_col].astype(str):
        valid.append(Chem.MolFromSmiles(smi) is not None)
    out = df.loc[valid].copy()
    out = out.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
    return out


def make_random_split(y: np.ndarray, seed: int) -> SplitIndices:
    indices = np.arange(len(y))
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=0.125,
        random_state=seed,
        stratify=y[train_valid_idx],
    )
    return SplitIndices(
        train=train_idx.tolist(),
        valid=valid_idx.tolist(),
        test=test_idx.tolist(),
    )


def make_scaffold_split(smiles: list[str]) -> SplitIndices:
    split = BemisMurckoScaffoldSplitter(valid_frac=0.1, test_frac=0.2).split(smiles)
    return SplitIndices(**split)


def fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if name == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=max(1, min(192, (os.cpu_count() or 8) - 8)),
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="jaccard",
            n_jobs=max(1, min(64, (os.cpu_count() or 8) // 2)),
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    model.fit(x_train, y_train)
    return model


def evaluate_binary_classifier(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, probs)),
        "auprc": float(average_precision_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "ece": expected_calibration_error(y_test, probs),
        "positive_rate": float(np.mean(y_test)),
    }


def split_is_usable(split: SplitIndices, y: np.ndarray) -> tuple[bool, str]:
    for part_name in ("train", "valid", "test"):
        idx = getattr(split, part_name)
        if len(idx) == 0:
            return False, f"{part_name}_empty"
        values = y[idx]
        if part_name == "train" and len(np.unique(values)) < 2:
            return False, "train_single_class"
        if part_name in {"valid", "test"} and len(np.unique(values)) < 2:
            return False, f"{part_name}_single_class"
    return True, "ok"


def run_task(dataset_cfg: dict, label_col: str, seed: int) -> list[dict]:
    df = pd.read_csv(dataset_cfg["path"])
    df = df[[dataset_cfg["smiles_col"], label_col]].dropna()
    df = df.rename(columns={dataset_cfg["smiles_col"]: "smiles", label_col: "label"})
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")

    if len(df) < 200:
        return [{
            "dataset": dataset_cfg["name"],
            "label": label_col,
            "status": "skipped_too_small",
            "n_samples": len(df),
        }]

    y = df["label"].to_numpy()
    if not has_valid_binary_labels(y):
        return [{
            "dataset": dataset_cfg["name"],
            "label": label_col,
            "status": "skipped_non_binary",
            "n_samples": len(df),
        }]

    x = morgan_fingerprint_matrix(df["smiles"].tolist(), radius=2, n_bits=2048)
    results: list[dict] = []

    for split_name, split in (
        ("random", make_random_split(y, seed)),
        ("scaffold", make_scaffold_split(df["smiles"].tolist())),
    ):
        usable, reason = split_is_usable(split, y)
        if not usable:
            results.append({
                "dataset": dataset_cfg["name"],
                "label": label_col,
                "split": split_name,
                "status": f"skipped_{reason}",
                "n_samples": len(df),
                "n_positive": int(np.sum(y)),
            })
            continue

        x_train = x[split.train]
        y_train = y[split.train]
        x_test = x[split.test]
        y_test = y[split.test]

        for model_name in ("logreg", "rf", "knn"):
            if model_name == "knn":
                x_train_model = x_train.astype(bool)
                x_test_model = x_test.astype(bool)
            else:
                x_train_model = x_train
                x_test_model = x_test

            model = fit_model(model_name, x_train_model, y_train, seed)
            metrics = evaluate_binary_classifier(model, x_test_model, y_test)
            results.append({
                "dataset": dataset_cfg["name"],
                "label": label_col,
                "split": split_name,
                "model": model_name,
                "status": "ok",
                "n_samples": len(df),
                "n_positive": int(np.sum(y)),
                "train_size": len(split.train),
                "valid_size": len(split.valid),
                "test_size": len(split.test),
                **metrics,
            })

    return results


def main() -> None:
    seed = 42
    output_dir = ROOT / "outputs" / "fingerprint_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for dataset_cfg in DATASETS:
        for label_col in dataset_cfg["label_cols"]:
            print(f"RUN {dataset_cfg['name']}::{label_col}")
            task_results = run_task(dataset_cfg, label_col, seed)
            all_results.extend(task_results)

    result_df = pd.DataFrame(all_results)
    result_path = output_dir / "results.csv"
    result_df.to_csv(result_path, index=False)

    summary = {
        "num_rows": len(result_df),
        "num_success": int((result_df["status"] == "ok").sum()) if not result_df.empty else 0,
        "datasets": sorted(result_df["dataset"].dropna().unique().tolist()) if not result_df.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not result_df.empty:
        ok_df = result_df[result_df["status"] == "ok"].copy()
        if not ok_df.empty:
            print("\n=== AGGREGATED MEAN BY MODEL/SPLIT ===")
            agg = ok_df.groupby(["split", "model"])[["auroc", "auprc", "brier", "ece"]].mean()
            print(agg.round(4).to_string())
            print("\n=== FULL RESULTS ===")
            cols = ["dataset", "label", "split", "model", "auroc", "auprc", "brier", "ece", "n_samples", "n_positive"]
            print(ok_df[cols].round(4).to_string(index=False))
        else:
            print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
