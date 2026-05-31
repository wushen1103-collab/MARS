from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix

from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, fit_rf, load_task_frame


def mol_weight(smiles: list[str]) -> np.ndarray:
    return np.asarray([Descriptors.MolWt(Chem.MolFromSmiles(smi)) for smi in smiles], dtype=np.float64)


def random_train_valid(train_idx: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train, valid = train_test_split(train_idx, test_size=0.125, random_state=seed, stratify=y[train_idx])
    return np.asarray(train), np.asarray(valid)


def make_molecular_weight_reverse_split(smiles: list[str], y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    mw = mol_weight(smiles)
    order = np.argsort(mw)
    test_idx = order[int(len(order) * 0.8) :]
    train_pool = order[: int(len(order) * 0.8)]
    train_idx, valid_idx = random_train_valid(train_pool, y, seed)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def make_pca_cluster_split(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    reducer = PCA(n_components=min(10, x.shape[1]), random_state=seed)
    emb = reducer.fit_transform(x)
    labels = KMeans(n_clusters=5, random_state=seed, n_init=10).fit_predict(emb)
    counts = pd.Series(labels).value_counts()
    test_cluster = int(counts.index[-1])
    test_idx = np.where(labels == test_cluster)[0]
    train_pool = np.where(labels != test_cluster)[0]
    train_idx, valid_idx = random_train_valid(train_pool, y, seed)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def make_fingerprint_density_split(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    density = x.sum(axis=1)
    order = np.argsort(density)
    test_idx = order[: int(len(order) * 0.2)]
    train_pool = order[int(len(order) * 0.2) :]
    train_idx, valid_idx = random_train_valid(train_pool, y, seed)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def make_umap_split(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Compatibility alias for fixed outputs generated before the split was renamed."""
    return make_pca_cluster_split(x, y, seed)


def make_lohi_split(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Compatibility alias for fixed outputs generated before the split was renamed."""
    return make_fingerprint_density_split(x, y, seed)


def split_usable(split: dict[str, np.ndarray], y: np.ndarray) -> bool:
    return all(len(split[k]) > 0 and np.unique(y[split[k]]).size > 1 for k in ("train", "valid", "test"))


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int | None) -> list[dict]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    smiles = df["smiles"].tolist()
    x = morgan_fingerprint_matrix(smiles)
    splitters = {
        "pca_cluster": make_pca_cluster_split,
        "molecular_weight_reverse": make_molecular_weight_reverse_split,
        "fingerprint_density": make_fingerprint_density_split,
    }
    rows = []
    for split_name, splitter in splitters.items():
        split = splitter(smiles, y, seed) if split_name == "molecular_weight_reverse" else splitter(x, y, seed)
        if not split_usable(split, y):
            rows.append({"dataset": task_cfg["dataset"], "label": task_cfg["label"], "split": split_name, "status": "skipped_single_class"})
            continue
        rf = fit_rf(x[split["train"]], y[split["train"]], seed=seed, rf_n_jobs=rf_n_jobs)
        probs = rf.predict_proba(x[split["test"]])[:, 1]
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "split": split_name,
                "model": "rf_morgan",
                "status": "ok",
                "train_size": int(len(split["train"])),
                "valid_size": int(len(split["valid"])),
                "test_size": int(len(split["test"])),
                **evaluate_probs(y[split["test"]], probs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "realistic_ood_splits")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}")
        rows.extend(run_task(task, args.seed, args.rf_n_jobs))
    pd.DataFrame(rows).to_csv(args.output_dir / "realistic_ood_results.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
