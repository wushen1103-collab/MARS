from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.chemprop_compat import (  # noqa: E402
    patch_pandas_rdkit_compat,
    patch_torch_load_weights_only_false,
)
from admet_shift_reliability.datasets import load_task_frame  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_chemprop_baseline import TASKS, select_task  # noqa: E402
from run_realistic_ood_splits import (  # noqa: E402
    make_fingerprint_density_split,
    make_molecular_weight_reverse_split,
    make_pca_cluster_split,
    split_usable,
)

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    valid = [Chem.MolFromSmiles(str(s)) is not None for s in df["smiles"]]
    return df.loc[valid].drop_duplicates(subset=["smiles"]).reset_index(drop=True).copy()


def make_split(split_name: str, smiles: list[str], y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    x = morgan_fingerprint_matrix(smiles)
    if split_name in {"pca_cluster", "umap"}:
        return make_pca_cluster_split(x, y, seed)
    if split_name in {"fingerprint_density", "lohi"}:
        return make_fingerprint_density_split(x, y, seed)
    if split_name == "molecular_weight_reverse":
        return make_molecular_weight_reverse_split(smiles, y, seed)
    raise ValueError(f"Unsupported strict split: {split_name}")


def write_split_csvs(df: pd.DataFrame, split: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
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
    parser.add_argument(
        "--strict-split",
        choices=["fingerprint_density", "molecular_weight_reverse", "pca_cluster", "lohi", "umap"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--save-tag", default="strict_single")
    args = parser.parse_args()

    task_cfg = select_task(args.dataset, args.label)
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df)
    y = df["label"].to_numpy()
    split = make_split(args.strict_split, df["smiles"].tolist(), y, args.seed)
    if not split_usable(split, y):
        raise ValueError(f"{args.dataset}:{args.label}:{args.strict_split} has a single-class split")

    split_dir = ROOT / "data" / "chemprop_strict_splits" / args.dataset / args.label / f"{args.strict_split}_seed{args.seed}"
    split_paths = write_split_csvs(df, split, split_dir)

    patch_pandas_rdkit_compat()
    patch_torch_load_weights_only_false()
    from chemprop.train import chemprop_train

    run_name = f"{args.dataset}__{args.label}__{args.strict_split}__seed{args.seed}__{args.save_tag}"
    save_dir = ROOT / "outputs" / "chemprop_strict_ood" / run_name
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
    sys.argv = cli_args
    chemprop_train()
    payload = {"dataset": args.dataset, "label": args.label, "split": args.strict_split, "seed": args.seed, "save_dir": str(save_dir)}
    print("CHEMPROP_STRICT_OOD_DONE", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
