from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.gnn_models import GINBinaryClassifier
from admet_shift_reliability.graphs import smiles_to_pyg_graph
from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


DATASET_CONFIGS = {
    "tox21": {"path": ROOT / "data" / "raw" / "tox21.csv.gz", "smiles_col": "smiles", "default_label": "NR-AhR"},
    "bbbp": {"path": ROOT / "data" / "raw" / "BBBP.csv", "smiles_col": "smiles", "default_label": "p_np"},
    "clintox": {"path": ROOT / "data" / "raw" / "clintox.csv.gz", "smiles_col": "smiles", "default_label": "CT_TOX"},
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


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
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


def load_graph_dataset(dataset: str, label: str) -> tuple[list, list[str], np.ndarray]:
    cfg = DATASET_CONFIGS[dataset]
    raw_label = cfg.get("raw_label_col", label)
    df = pd.read_csv(cfg["path"])
    df = df[[cfg["smiles_col"], raw_label]].dropna().copy()
    df = df.rename(columns={cfg["smiles_col"]: "smiles", raw_label: "label"})
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    smiles = df["smiles"].tolist()
    graphs = [smiles_to_pyg_graph(smi, y=target) for smi, target in zip(smiles, df["label"])]
    return graphs, smiles, df["label"].to_numpy()


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


@torch.no_grad()
def collect_embeddings_and_probs(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    probs = []
    labels = []
    for batch in loader:
        batch = batch.to(device)
        graph_repr = model.encode_batch(batch)
        logits = model.head(graph_repr).squeeze(-1)
        embeddings.append(graph_repr.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(batch.y.view(-1).detach().cpu().numpy())
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0).astype(np.int64),
    )


def compute_embedding_anchor_prob(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    query_emb: np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    k = max(1, min(int(n_neighbors), int(train_emb.shape[0])))
    nn_index = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    nn_index.fit(train_emb)
    distances, indices = nn_index.kneighbors(query_emb, return_distance=True)
    similarities = np.clip(1.0 - distances, a_min=1e-6, a_max=None)
    weights = similarities / similarities.sum(axis=1, keepdims=True)
    return np.sum(weights * train_y[indices], axis=1)


def maybe_fit_meta(valid_features: np.ndarray, y_valid: np.ndarray, seed: int) -> LogisticRegression | None:
    if np.unique(y_valid).size < 2:
        return None
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(valid_features, y_valid)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="tox21")
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    graphs, smiles, y = load_graph_dataset(args.dataset, args.label)
    split = make_scaffold_split(smiles)
    usable, reason = split_is_usable(split, y)
    if not usable:
        raise RuntimeError(f"Split unusable: {reason}")

    train_graphs = [graphs[i] for i in split["train"]]
    valid_graphs = [graphs[i] for i in split["valid"]]
    test_graphs = [graphs[i] for i in split["test"]]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = GINBinaryClassifier(hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
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
        _, valid_probs, valid_labels = collect_embeddings_and_probs(model, valid_loader, device)
        valid_auprc = float(average_precision_score(valid_labels, valid_probs))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_auprc": valid_auprc})
        print(f"EPOCH {epoch:03d} train_loss={train_loss:.4f} valid_auprc={valid_auprc:.4f}")

        if valid_auprc > best_valid:
            best_valid = valid_auprc
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

    train_emb, train_probs, train_labels = collect_embeddings_and_probs(model, train_loader, device)
    valid_emb, valid_probs, valid_labels = collect_embeddings_and_probs(model, valid_loader, device)
    test_emb, test_probs, test_labels = collect_embeddings_and_probs(model, test_loader, device)

    valid_anchor = compute_embedding_anchor_prob(train_emb, train_labels, valid_emb, n_neighbors=15)
    test_anchor = compute_embedding_anchor_prob(train_emb, train_labels, test_emb, n_neighbors=15)

    valid_meta_x = np.column_stack([valid_probs, valid_anchor, np.abs(valid_probs - valid_anchor)]).astype(np.float32)
    test_meta_x = np.column_stack([test_probs, test_anchor, np.abs(test_probs - test_anchor)]).astype(np.float32)
    meta_model = maybe_fit_meta(valid_meta_x, valid_labels, seed=args.seed)
    if meta_model is None:
        meta_test = 0.5 * test_probs + 0.5 * test_anchor
    else:
        meta_test = meta_model.predict_proba(test_meta_x)[:, 1]

    result = {
        "dataset": args.dataset,
        "label": args.label,
        "split": "scaffold",
        "epochs_ran": len(history),
        "train_size": len(train_graphs),
        "valid_size": len(valid_graphs),
        "test_size": len(test_graphs),
        "n_positive_total": int(y.sum()),
        "n_positive_test": int(test_labels.sum()),
    }
    for prefix, probs in (
        ("gin", test_probs),
        ("embed_anchor", test_anchor),
        ("meta", meta_test),
    ):
        for key, value in evaluate_probs(test_labels, probs).items():
            result[f"{prefix}_{key}"] = value

    print("TEST_METRICS", json.dumps(result, indent=2))

    out_dir = args.output_dir or (ROOT / "outputs" / "gin_embedding_anchor_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.label.replace('/', '_')}__scaffold"
    (out_dir / f"{stem}.history.json").write_text(json.dumps(history, indent=2))
    (out_dir / f"{stem}.result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
