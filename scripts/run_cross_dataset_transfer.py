from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.datasets import load_task_frame  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, fit_rf  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    group: str
    name: str
    label: str = "Y"
    source: str = "tdc"
    main_dataset: str | None = None
    main_label: str | None = None


TRANSFER_PAIRS = [
    ("bbbp", "bbb_martins", "bbb_endpoint"),
    ("bbb_martins", "bbbp", "bbb_endpoint"),
    ("bbbp", "b3db_classification", "bbb_endpoint"),
    ("b3db_classification", "bbbp", "bbb_endpoint"),
    ("bbb_martins", "b3db_classification", "bbb_endpoint"),
    ("herg", "herg_karim", "herg_endpoint"),
    ("herg_karim", "herg", "herg_endpoint"),
    ("herg", "herg_central", "herg_endpoint"),
    ("herg_central", "herg", "herg_endpoint"),
    ("ames", "carcinogens_lagunin", "genotoxicity_safety_family"),
    ("carcinogens_lagunin", "ames", "genotoxicity_safety_family"),
    ("clintox", "dili", "clinical_toxicity_family"),
    ("dili", "clintox", "clinical_toxicity_family"),
]


DATASETS = {
    "bbbp": DatasetSpec("bbbp", group="main", name="bbbp", label="p_np", source="main", main_dataset="bbbp", main_label="p_np"),
    "clintox": DatasetSpec("clintox", group="main", name="clintox", label="CT_TOX", source="main", main_dataset="clintox", main_label="CT_TOX"),
    "ames": DatasetSpec("ames", group="main", name="ames", label="AMES", source="main", main_dataset="ames", main_label="AMES"),
    "herg": DatasetSpec("herg", group="main", name="herg", label="hERG", source="main", main_dataset="herg", main_label="hERG"),
    "dili": DatasetSpec("dili", group="main", name="dili", label="DILI", source="main", main_dataset="dili", main_label="DILI"),
    "bbb_martins": DatasetSpec("bbb_martins", group="ADME", name="BBB_Martins"),
    "b3db_classification": DatasetSpec("b3db_classification", group="ADME", name="B3DB_classification"),
    "herg_karim": DatasetSpec("herg_karim", group="Tox", name="hERG_Karim"),
    "herg_central": DatasetSpec("herg_central", group="Tox", name="hERG_Central"),
    "carcinogens_lagunin": DatasetSpec("carcinogens_lagunin", group="Tox", name="Carcinogens_Lagunin"),
}


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main_task_cfg(dataset: str, label: str) -> dict:
    for task in TASKS:
        if task["dataset"] == dataset and task["label"] == label:
            return task
    raise ValueError(f"Unknown main task {dataset}:{label}")


def normalize_binary_frame(df: pd.DataFrame, smiles_col: str, label_col: str) -> pd.DataFrame:
    out = df[[smiles_col, label_col]].rename(columns={smiles_col: "smiles", label_col: "label"}).dropna().copy()
    unique = sorted(pd.Series(out["label"]).dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"Expected binary labels, got {unique[:10]}")
    mapping = {unique[0]: 0, unique[1]: 1}
    out["label"] = out["label"].map(mapping).astype(int)
    return out


def load_tdc_frame(spec: DatasetSpec) -> pd.DataFrame:
    if spec.group == "ADME":
        from tdc.single_pred import ADME

        data = ADME(name=spec.name, path=str(ROOT / "data" / "tdc_external"))
    elif spec.group == "Tox":
        from tdc.single_pred import Tox

        data = Tox(name=spec.name, path=str(ROOT / "data" / "tdc_external"))
    else:
        raise ValueError(f"Unsupported TDC group: {spec.group}")
    df = data.get_data(format="df")
    smiles_col = "Drug" if "Drug" in df.columns else "smiles" if "smiles" in df.columns else df.columns[0]
    label_col = "Y" if "Y" in df.columns else spec.label if spec.label in df.columns else df.columns[-1]
    return normalize_binary_frame(df, smiles_col=smiles_col, label_col=label_col)


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if spec.source == "main":
        task = main_task_cfg(spec.main_dataset or spec.dataset_id, spec.main_label or spec.label)
        df = load_task_frame(task).dropna().copy()
        df["label"] = df["label"].astype(int)
    else:
        df = load_tdc_frame(spec)
    df = filter_valid_smiles(df, "smiles")
    df["canonical_smiles"] = [canonical_smiles(smi) for smi in df["smiles"]]
    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError(f"{spec.dataset_id} has a single class after filtering")
    return df


def make_train_valid_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, valid_idx = train_test_split(indices, test_size=0.15, random_state=seed, stratify=y)
    return np.asarray(train_idx), np.asarray(valid_idx)


def fit_reasoner(valid_prob: np.ndarray, valid_anchor: dict[str, np.ndarray], valid_y: np.ndarray, seed: int) -> LogisticRegression | None:
    if np.unique(valid_y).size < 2:
        return None
    x_valid = np.column_stack([valid_prob, valid_anchor["anchor_prob"], np.abs(valid_prob - valid_anchor["anchor_prob"]), valid_anchor["anchor_disagreement"], valid_anchor["anchor_distance_mean"]])
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def apply_reasoner(model: LogisticRegression | None, prob: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    if model is None:
        return np.clip(0.5 * prob + 0.5 * anchor["anchor_prob"], 0.0, 1.0)
    x = np.column_stack([prob, anchor["anchor_prob"], np.abs(prob - anchor["anchor_prob"]), anchor["anchor_disagreement"], anchor["anchor_distance_mean"]])
    return model.predict_proba(x)[:, 1]


def run_pair(source_id: str, target_id: str, family: str, seed: int, rf_n_jobs: int) -> list[dict]:
    source = load_dataset(DATASETS[source_id])
    target = load_dataset(DATASETS[target_id])
    source_smiles = set(source["canonical_smiles"])
    target_no_overlap = target[~target["canonical_smiles"].isin(source_smiles)].copy().reset_index(drop=True)
    overlap = len(target) - len(target_no_overlap)
    if len(target_no_overlap) < 20 or target_no_overlap["label"].nunique() < 2:
        raise ValueError(f"{source_id}->{target_id} target too small or single-class after overlap removal")

    source_y = source["label"].to_numpy()
    target_y = target_no_overlap["label"].to_numpy()
    train_idx, valid_idx = make_train_valid_indices(source_y, seed=seed)
    source_x = morgan_fingerprint_matrix(source["smiles"].tolist())
    target_x = morgan_fingerprint_matrix(target_no_overlap["smiles"].tolist())
    source_xb = source_x.astype(bool)
    target_xb = target_x.astype(bool)

    rf = fit_rf(source_x[train_idx], source_y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    valid_prob = rf.predict_proba(source_x[valid_idx])[:, 1]
    target_prob = rf.predict_proba(target_x)[:, 1]
    valid_anchor = compute_anchor_features(source_xb[train_idx], source_y[train_idx], source_xb[valid_idx], n_neighbors=15)
    target_anchor = compute_anchor_features(source_xb[train_idx], source_y[train_idx], target_xb, n_neighbors=15)
    reasoner = fit_reasoner(valid_prob, valid_anchor, source_y[valid_idx], seed=seed)
    reasoning_prob = apply_reasoner(reasoner, target_prob, target_anchor)

    rows = []
    for model_name, probs in {"rf_morgan": target_prob, "retrieval_only": target_anchor["anchor_prob"], "anchor_reasoning": reasoning_prob}.items():
        rows.append(
            {
                "source_dataset": source_id,
                "target_dataset": target_id,
                "transfer_family": family,
                "model": model_name,
                "seed": seed,
                "source_n": int(len(source)),
                "source_train_size": int(len(train_idx)),
                "source_valid_size": int(len(valid_idx)),
                "target_original_n": int(len(target)),
                "target_test_size": int(len(target_no_overlap)),
                "target_overlap_removed": int(overlap),
                "target_positive_rate": float(np.mean(target_y)),
                **evaluate_probs(target_y, probs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cross_dataset_transfer_20260422")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    for source_id, target_id, family in TRANSFER_PAIRS:
        print(f"RUN {source_id}->{target_id}", flush=True)
        try:
            rows.extend(run_pair(source_id, target_id, family, seed=args.seed, rf_n_jobs=args.rf_n_jobs))
        except Exception as exc:
            failures.append({"source_dataset": source_id, "target_dataset": target_id, "transfer_family": family, "error": repr(exc)})
            print(f"SKIP {source_id}->{target_id}: {exc}", flush=True)

    pd.DataFrame(rows).to_csv(args.output_dir / "cross_dataset_transfer_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "cross_dataset_transfer_failures.csv", index=False)
    summary = {
        "candidate_pairs": len(TRANSFER_PAIRS),
        "successful_pairs": int(pd.DataFrame(rows)[["source_dataset", "target_dataset"]].drop_duplicates().shape[0]) if rows else 0,
        "rows": len(rows),
        "failures": len(failures),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
