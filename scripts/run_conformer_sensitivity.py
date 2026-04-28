from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix
from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, load_task_frame, make_scaffold_split


def conformer_descriptors(
    smiles: str,
    num_conformers: int,
    seed: int,
    max_heavy_atoms: int,
    embed_timeout_seconds: int,
) -> tuple[np.ndarray, bool]:
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None or base_mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return np.zeros(12, dtype=np.float32), False
    mol = Chem.AddHs(base_mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
    if hasattr(params, "timeout"):
        params.timeout = int(embed_timeout_seconds)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if len(ids) == 0:
        return np.zeros(12, dtype=np.float32), False
    rows = []
    for cid in ids:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=int(cid), maxIters=80)
        except Exception:
            pass
        rows.append(
            [
                Descriptors3D.Asphericity(mol, confId=int(cid)),
                Descriptors3D.Eccentricity(mol, confId=int(cid)),
                Descriptors3D.InertialShapeFactor(mol, confId=int(cid)),
                Descriptors3D.NPR1(mol, confId=int(cid)),
                Descriptors3D.NPR2(mol, confId=int(cid)),
                Descriptors3D.RadiusOfGyration(mol, confId=int(cid)),
            ]
        )
    arr = np.asarray(rows, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0), True


def conformer_descriptor_worker(args: tuple[str, int, int, int, int]) -> tuple[np.ndarray, bool]:
    smiles, num_conformers, seed, max_heavy_atoms, embed_timeout_seconds = args
    return conformer_descriptors(
        smiles,
        num_conformers=num_conformers,
        seed=seed,
        max_heavy_atoms=max_heavy_atoms,
        embed_timeout_seconds=embed_timeout_seconds,
    )


def build_conformer_descriptor_matrix(
    smiles: list[str],
    num_conformers: int,
    seed: int,
    conformer_workers: int,
    max_heavy_atoms: int,
    embed_timeout_seconds: int,
) -> tuple[np.ndarray, list[bool]]:
    tasks = [(smi, num_conformers, seed + idx, max_heavy_atoms, embed_timeout_seconds) for idx, smi in enumerate(smiles)]
    if conformer_workers <= 1:
        results = [conformer_descriptor_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=conformer_workers) as executor:
            results = list(executor.map(conformer_descriptor_worker, tasks, chunksize=16))
    desc_rows = [item[0] for item in results]
    ok = [item[1] for item in results]
    return np.asarray(desc_rows, dtype=np.float32), ok


def run_task(
    task_cfg: dict,
    seed: int,
    n_jobs: int,
    conformer_counts: list[int],
    conformer_workers: int,
    max_heavy_atoms: int,
    embed_timeout_seconds: int,
) -> tuple[list[dict], list[dict]]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    fp = morgan_fingerprint_matrix(df["smiles"].tolist())
    rows, stats = [], []
    smiles = df["smiles"].tolist()
    for num_conformers in conformer_counts:
        desc, ok = build_conformer_descriptor_matrix(
            smiles,
            num_conformers=num_conformers,
            seed=seed,
            conformer_workers=conformer_workers,
            max_heavy_atoms=max_heavy_atoms,
            embed_timeout_seconds=embed_timeout_seconds,
        )
        x = np.concatenate([fp, desc], axis=1)
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=n_jobs, random_state=seed)
        model.fit(x[split["train"]], y[split["train"]])
        probs = model.predict_proba(x[split["test"]])[:, 1]
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "num_conformers": num_conformers,
                "model": "rf_2d_plus_3d_conformer_stats",
                "conformer_success_rate": float(np.mean(ok)),
                "max_heavy_atoms": int(max_heavy_atoms),
                "embed_timeout_seconds": int(embed_timeout_seconds),
                **evaluate_probs(y[split["test"]], probs),
            }
        )
        stats.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "num_conformers": num_conformers,
                "success_rate": float(np.mean(ok)),
                "max_heavy_atoms": int(max_heavy_atoms),
                "embed_timeout_seconds": int(embed_timeout_seconds),
            }
        )
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--conformer-workers", type=int, default=32)
    parser.add_argument("--num-conformers", default="1,3,5")
    parser.add_argument("--max-heavy-atoms", type=int, default=60)
    parser.add_argument("--embed-timeout-seconds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "conformer_sensitivity")
    parser.add_argument("--task", action="append", default=None)
    args = parser.parse_args()
    conformer_counts = [int(x) for x in args.num_conformers.split(",") if x]
    wanted = set(tuple(item.split(":", 1)) for item in args.task) if args.task else None
    tasks = [t for t in TASKS if wanted is None or (t["dataset"], t["label"]) in wanted]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = [], []
    for task in tasks:
        print(f"RUN {task['dataset']}::{task['label']}")
        r, s = run_task(
            task,
            args.seed,
            args.n_jobs,
            conformer_counts,
            args.conformer_workers,
            args.max_heavy_atoms,
            args.embed_timeout_seconds,
        )
        rows.extend(r)
        stats.extend(s)
    pd.DataFrame(rows).to_csv(args.output_dir / "conformer_sensitivity_results.csv", index=False)
    pd.DataFrame(stats).to_csv(args.output_dir / "conformer_generation_stats.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": len(rows), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()

