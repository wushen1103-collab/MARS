from __future__ import annotations

import argparse
import os
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.conformers import generate_conformer
from admet_shift_reliability.models_3d import SchNetBinaryClassifier
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


DATASET_CONFIGS = {
    "bbbp": {"path": ROOT / "data" / "raw" / "BBBP.csv", "smiles_col": "smiles", "default_label": "p_np"},
    "clintox": {"path": ROOT / "data" / "raw" / "clintox.csv.gz", "smiles_col": "smiles", "default_label": "CT_TOX"},
    "tox21": {"path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "default_label": "NR-AhR"},
    "ames": {
        "path": ROOT / "data" / "raw" / "AMES_tdc.csv.gz",
        "smiles_col": "Drug",
        "raw_label_col": "Y",
        "default_label": "AMES",
    },
    "herg": {
        "path": ROOT / "data" / "raw" / "hERG_tdc.csv.gz",
        "smiles_col": "Drug",
        "raw_label_col": "Y",
        "default_label": "hERG",
    },
    "dili": {
        "path": ROOT / "data" / "raw" / "DILI_tdc.csv.gz",
        "smiles_col": "Drug",
        "raw_label_col": "Y",
        "default_label": "DILI",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    valid = []
    for smi in df[smiles_col].astype(str):
        valid.append(Chem.MolFromSmiles(smi) is not None)
    out = df.loc[valid].copy()
    out = out.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
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


def cache_path_for(dataset: str, seed: int) -> Path:
    cache_dir = ROOT / "data" / "cache_3d"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{dataset}__seed{seed}.pt"


def build_cached_graph_item(task: tuple[str, int]) -> tuple[str, np.ndarray, np.ndarray] | None:
    smiles, seed = task
    try:
        atomic_numbers, positions = generate_conformer(smiles, seed=seed)
    except ValueError:
        return None
    return smiles, atomic_numbers, positions


def load_or_build_base_graph_cache(dataset: str, seed: int, cache_workers: int) -> dict[str, object]:
    cache_path = cache_path_for(dataset=dataset, seed=seed)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    cfg = DATASET_CONFIGS[dataset]
    df = pd.read_csv(cfg["path"])
    df = df[[cfg["smiles_col"]]].dropna().copy()
    df = df.rename(columns={cfg["smiles_col"]: "smiles"})
    df = filter_valid_smiles(df, "smiles")

    smiles = []
    tasks = [(row.smiles, seed + idx) for idx, row in enumerate(df.itertuples(index=False))]
    if cache_workers > 1:
        with ProcessPoolExecutor(max_workers=cache_workers) as executor:
            results = list(
                tqdm(
                    executor.map(build_cached_graph_item, tasks, chunksize=32),
                    total=len(tasks),
                    desc=f"cache_{dataset}_3d",
                    leave=False,
                    disable=not sys.stderr.isatty(),
                )
            )
    else:
        results = [
            build_cached_graph_item(task)
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc=f"cache_{dataset}_3d",
                leave=False,
                disable=not sys.stderr.isatty(),
            )
        ]

    graphs_by_smiles = {}
    for item in results:
        if item is None:
            continue
        smiles_value, atomic_numbers, positions = item
        graphs_by_smiles[smiles_value] = Data(
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(positions, dtype=torch.float32),
        )
        smiles.append(smiles_value)

    payload = {"graphs_by_smiles": graphs_by_smiles, "smiles": smiles}
    torch.save(payload, cache_path)
    return payload


def load_graph_dataset(dataset: str, label: str, seed: int, cache_workers: int) -> tuple[list, list[str], np.ndarray]:
    base_payload = load_or_build_base_graph_cache(dataset=dataset, seed=seed, cache_workers=cache_workers)
    graphs_by_smiles = base_payload["graphs_by_smiles"]

    cfg = DATASET_CONFIGS[dataset]
    raw_label = cfg.get("raw_label_col", label)
    df = pd.read_csv(cfg["path"])
    df = df[[cfg["smiles_col"], raw_label]].dropna().copy()
    df = df.rename(columns={cfg["smiles_col"]: "smiles", raw_label: "label"})
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")

    graphs = []
    smiles = []
    labels = []
    for row in df.itertuples(index=False):
        base_graph = graphs_by_smiles.get(row.smiles)
        if base_graph is None:
            continue
        graph = deepcopy(base_graph)
        graph.y = torch.tensor([int(row.label)], dtype=torch.float32)
        graphs.append(graph)
        smiles.append(row.smiles)
        labels.append(int(row.label))

    return graphs, smiles, np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    probs_all = []
    y_all = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch.y.view(-1).detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y)

    probs = np.concatenate(probs_all)
    y_true = np.concatenate(y_all).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
        "positive_rate": float(np.mean(y_true)),
    }


@torch.no_grad()
def collect_logits_probs(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    probs_all = []
    y_all = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(batch.y.view(-1).detach().cpu().numpy())
    return (
        np.concatenate(logits_all).astype(np.float64),
        np.concatenate(probs_all).astype(np.float64),
        np.concatenate(y_all).astype(int),
    )


def build_prediction_frame(
    *,
    dataset: str,
    label: str,
    split_name: str,
    seed: int,
    part: str,
    indices: list[int],
    smiles: list[str],
    logits: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    row_index = np.asarray(indices, dtype=np.int64)
    if not (len(row_index) == len(logits) == len(probs) == len(y_true)):
        raise RuntimeError(f"Prediction alignment failed for {part}: indices and outputs have different lengths.")
    return pd.DataFrame(
        {
            "dataset": dataset,
            "label": label,
            "model": "schnet",
            "split": split_name,
            "seed": int(seed),
            "part": part,
            "row_index": row_index,
            "smiles": [smiles[idx] for idx in row_index],
            "y_true": y_true.astype(int),
            "logit": logits,
            "prob": probs,
        }
    )


def write_prediction_dump(
    *,
    model,
    valid_loader,
    test_loader,
    split: dict[str, list[int]],
    smiles: list[str],
    device: torch.device,
    dataset: str,
    label: str,
    split_name: str,
    seed: int,
    prediction_dir: Path,
) -> Path:
    rows = []
    for part, loader in (("valid", valid_loader), ("test", test_loader)):
        logits, probs, y_true = collect_logits_probs(model, loader, device=device)
        rows.append(
            build_prediction_frame(
                dataset=dataset,
                label=label,
                split_name=split_name,
                seed=seed,
                part=part,
                indices=split[part],
                smiles=smiles,
                logits=logits,
                probs=probs,
                y_true=y_true,
            )
        )

    prediction_dir.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "_")
    out_path = prediction_dir / f"schnet__{dataset}__{safe}__{split_name}__seed{seed}.predictions.csv"
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=min(64, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    label = args.label or DATASET_CONFIGS[args.dataset]["default_label"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    graphs, smiles, y = load_graph_dataset(args.dataset, label, seed=args.seed, cache_workers=args.cache_workers)
    split = make_random_split(y, args.seed) if args.split == "random" else make_scaffold_split(smiles)
    usable, reason = split_is_usable(split, y)
    if not usable:
        raise RuntimeError(f"Split unusable: {reason}")

    train_graphs = [graphs[i] for i in split["train"]]
    valid_graphs = [graphs[i] for i in split["valid"]]
    test_graphs = [graphs[i] for i in split["test"]]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SchNetBinaryClassifier(
        hidden_channels=args.hidden_channels,
        num_filters=args.num_filters,
        num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
    ).to(device)
    pos = float(y[split["train"]].sum())
    neg = float(len(split["train"]) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_valid = -float("inf")
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_metrics = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **valid_metrics})
        print(
            f"EPOCH {epoch:03d} train_loss={train_loss:.4f} valid_auprc={valid_metrics['auprc']:.4f} "
            f"valid_auroc={valid_metrics['auroc']:.4f} valid_ece={valid_metrics['ece']:.4f}"
        )

        if valid_metrics["auprc"] > best_valid:
            best_valid = valid_metrics["auprc"]
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"EARLY_STOP at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    result = {
        "dataset": args.dataset,
        "label": label,
        "split": args.split,
        "seed": args.seed,
        "epochs_ran": len(history),
        "train_size": len(train_graphs),
        "valid_size": len(valid_graphs),
        "test_size": len(test_graphs),
        "n_positive_total": int(y.sum()),
        "n_positive_test": int(y[split["test"]].sum()),
        **test_metrics,
    }
    print("TEST_METRICS", json.dumps(result, indent=2))

    out_dir = args.output_dir or (ROOT / "outputs" / "schnet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{label.replace('/', '_')}__{args.split}"
    (out_dir / f"{stem}.history.json").write_text(json.dumps(history, indent=2))
    (out_dir / f"{stem}.result.json").write_text(json.dumps(result, indent=2))
    if args.prediction_dir is not None:
        prediction_path = write_prediction_dump(
            model=model,
            valid_loader=valid_loader,
            test_loader=test_loader,
            split=split,
            smiles=smiles,
            device=device,
            dataset=args.dataset,
            label=label,
            split_name=args.split,
            seed=args.seed,
            prediction_dir=args.prediction_dir,
        )
        print(f"PREDICTIONS {prediction_path}")


if __name__ == "__main__":
    main()
