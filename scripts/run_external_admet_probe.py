from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import evaluate_probs, filter_valid_smiles, fit_rf, make_scaffold_split  # noqa: E402


ADME_CLASSIFICATION_CANDIDATES = [
    "HIA_Hou",
    "Pgp_Broccatelli",
    "Bioavailability_Ma",
    "BBB_Martins",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
]


def load_tdc_adme(name: str) -> pd.DataFrame:
    from tdc.single_pred import ADME

    data = ADME(name=name, path=str(ROOT / "data" / "tdc_external"))
    df = data.get_data().rename(columns={"Drug": "smiles", "Y": "label"})
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Unexpected TDC schema for {name}: {df.columns.tolist()}")
    df = df[["smiles", "label"]].dropna().copy()
    unique = sorted(pd.Series(df["label"]).dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"{name} is not binary classification: labels={unique[:8]}")
    mapping = {unique[0]: 0, unique[1]: 1}
    df["label"] = df["label"].map(mapping).astype(int)
    return df


def fit_reasoner(valid_prob: np.ndarray, valid_anchor: dict[str, np.ndarray], valid_y: np.ndarray, seed: int) -> LogisticRegression:
    x_valid = np.column_stack(
        [
            valid_prob,
            valid_anchor["anchor_prob"],
            np.abs(valid_prob - valid_anchor["anchor_prob"]),
            valid_anchor["anchor_disagreement"],
            valid_anchor["anchor_distance_mean"],
        ]
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed)
    model.fit(x_valid, valid_y)
    return model


def apply_reasoner(model: LogisticRegression, prob: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    x = np.column_stack(
        [
            prob,
            anchor["anchor_prob"],
            np.abs(prob - anchor["anchor_prob"]),
            anchor["anchor_disagreement"],
            anchor["anchor_distance_mean"],
        ]
    )
    return model.predict_proba(x)[:, 1]


def run_dataset(name: str, seed: int, rf_n_jobs: int) -> list[dict]:
    df = load_tdc_adme(name)
    df = filter_valid_smiles(df, "smiles")
    if len(df) < 100 or df["label"].nunique() < 2:
        raise ValueError(f"{name} too small or single class after filtering")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    if any(np.unique(y[split[part]]).size < 2 for part in ["train", "valid", "test"]):
        raise ValueError(f"{name} scaffold split has a single-class partition")
    x = morgan_fingerprint_matrix(df["smiles"].tolist())
    xb = x.astype(bool)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    valid_prob = rf.predict_proba(x[valid_idx])[:, 1]
    test_prob = rf.predict_proba(x[test_idx])[:, 1]
    valid_anchor = compute_anchor_features(xb[train_idx], y[train_idx], xb[valid_idx], n_neighbors=15)
    test_anchor = compute_anchor_features(xb[train_idx], y[train_idx], xb[test_idx], n_neighbors=15)
    reasoner = fit_reasoner(valid_prob, valid_anchor, y[valid_idx], seed=seed)
    reasoning_prob = apply_reasoner(reasoner, test_prob, test_anchor)

    rows = []
    for model, probs in {"rf_morgan": test_prob, "anchor_reasoning": reasoning_prob}.items():
        rows.append(
            {
                "dataset": name,
                "label": "Y",
                "split": "scaffold",
                "model": model,
                "n_samples": int(len(df)),
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "test_size": int(len(test_idx)),
                **evaluate_probs(y[test_idx], probs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--datasets", default=",".join(ADME_CLASSIFICATION_CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "external_admet_probe")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        print(f"RUN {name}", flush=True)
        try:
            rows.extend(run_dataset(name, seed=args.seed, rf_n_jobs=args.rf_n_jobs))
        except Exception as exc:
            failures.append({"dataset": name, "error": repr(exc)})
            print(f"SKIP {name}: {exc}", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "external_admet_probe_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "external_admet_probe_failures.csv", index=False)
    summary = {
        "candidate_datasets": len([item for item in args.datasets.split(",") if item.strip()]),
        "successful_datasets": int(pd.DataFrame(rows)["dataset"].nunique()) if rows else 0,
        "rows": len(rows),
        "failures": len(failures),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

