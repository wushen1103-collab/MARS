from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.chemprop_compat import (
    patch_pandas_rdkit_compat,
    patch_torch_load_weights_only_false,
)
from admet_shift_reliability.datasets import load_task_frame
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


def select_task(dataset: str, label: str) -> dict:
    for task in TASKS:
        if task["dataset"] == dataset and task["label"] == label:
            return task
    raise ValueError(f"Unknown task: {dataset}:{label}")


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    valid = [Chem.MolFromSmiles(str(s)) is not None for s in df["smiles"]]
    out = df.loc[valid].copy()
    out = out.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return out


def make_random_split(y: np.ndarray, seed: int) -> dict[str, list[int]]:
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
    return {"train": train_idx.tolist(), "valid": valid_idx.tolist(), "test": test_idx.tolist()}


def make_scaffold_split(smiles: list[str]) -> dict[str, list[int]]:
    return BemisMurckoScaffoldSplitter(valid_frac=0.1, test_frac=0.2).split(smiles)


def write_split_csvs(df: pd.DataFrame, split: dict[str, list[int]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part in ("train", "valid", "test"):
        path = out_dir / f"{part}.csv"
        df.iloc[split[part]][["smiles", "label"]].to_csv(path, index=False)
        paths[part] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--loss-function", default=None)
    parser.add_argument("--save-tag", default=None)
    args = parser.parse_args()

    task_cfg = select_task(args.dataset, args.label)
    df = load_task_frame(task_cfg)
    df = df.dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df)

    y = df["label"].to_numpy()
    split = make_random_split(y, args.seed) if args.split == "random" else make_scaffold_split(df["smiles"].tolist())

    split_dir = ROOT / "data" / "chemprop_splits" / args.dataset / args.label / f"{args.split}_seed{args.seed}"
    split_paths = write_split_csvs(df, split, split_dir)

    patch_pandas_rdkit_compat()
    patch_torch_load_weights_only_false()
    from chemprop.train import chemprop_train

    run_name = f"{args.dataset}__{args.label}__{args.split}"
    if args.save_tag:
        run_name = f"{run_name}__{args.save_tag}"
    save_dir = ROOT / "outputs" / "chemprop_baseline" / run_name
    cli_args = [
        "chemprop_train",
        "--data_path",
        str(split_paths["train"]),
        "--separate_val_path",
        str(split_paths["valid"]),
        "--separate_test_path",
        str(split_paths["test"]),
        "--dataset_type",
        "classification",
        "--metric",
        "prc-auc",
        "--extra_metrics",
        "auc",
        "binary_cross_entropy",
        "--save_dir",
        str(save_dir),
        "--save_preds",
        "--class_balance",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--ensemble_size",
        str(args.ensemble_size),
        "--gpu",
        str(args.gpu),
        "--smiles_columns",
        "smiles",
        "--target_columns",
        "label",
    ]
    if args.loss_function:
        cli_args.extend(["--loss_function", args.loss_function])
    sys.argv = cli_args
    chemprop_train()

    payload = {
        "dataset": args.dataset,
        "label": args.label,
        "split": args.split,
        "seed": args.seed,
        "split_dir": str(split_dir),
        "save_dir": str(save_dir),
    }
    print("CHEMPROP_BASELINE_DONE", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

