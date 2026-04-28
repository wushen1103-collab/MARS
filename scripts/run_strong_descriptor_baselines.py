from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix
from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, load_task_frame, make_scaffold_split


DESC_FUNCS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
]


def descriptor_matrix(smiles: list[str]) -> np.ndarray:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append([float(fn(mol)) for fn in DESC_FUNCS])
    return np.asarray(rows, dtype=np.float32)


def run_task(task_cfg: dict, seed: int, n_jobs: int) -> list[dict]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    fp = morgan_fingerprint_matrix(df["smiles"].tolist())
    desc = descriptor_matrix(df["smiles"].tolist())
    features = {
        "chemprop_rdkit_proxy": np.concatenate([fp, desc], axis=1),
        "xgb_rdkit": desc,
    }
    rows = []
    for name, x in features.items():
        if name == "chemprop_rdkit_proxy":
            model = RandomForestClassifier(n_estimators=800, min_samples_leaf=1, class_weight="balanced", n_jobs=n_jobs, random_state=seed)
        else:
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, l2_regularization=0.01, random_state=seed)
        model.fit(x[split["train"]], y[split["train"]])
        probs = model.predict_proba(x[split["test"]])[:, 1]
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "split": "scaffold",
                "model": name,
                "train_size": len(split["train"]),
                "valid_size": len(split["valid"]),
                "test_size": len(split["test"]),
                **evaluate_probs(y[split["test"]], probs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "strong_descriptor_baselines")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}")
        rows.extend(run_task(task, args.seed, args.n_jobs))
    pd.DataFrame(rows).to_csv(args.output_dir / "strong_descriptor_baselines.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
