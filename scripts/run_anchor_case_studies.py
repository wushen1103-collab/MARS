from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import TASKS, filter_valid_smiles, fit_rf, load_task_frame, make_scaffold_split  # noqa: E402


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


def compact_anchor_text(smiles: list[str], labels: np.ndarray, similarities: np.ndarray) -> str:
    return " ; ".join(f"{smi}|y={int(label)}|sim={float(sim):.3f}" for smi, label, sim in zip(smiles, labels, similarities))


def pick_cases(frame: pd.DataFrame, category: str, mask: np.ndarray, max_cases: int) -> pd.DataFrame:
    if not np.any(mask):
        return pd.DataFrame()
    sub = frame.loc[mask].copy()
    sub["case_category"] = category
    return sub.sort_values(["confidence", "ood_score"], ascending=[False, False]).head(max_cases)


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int, top_k: int, max_cases: int) -> list[dict]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    x = morgan_fingerprint_matrix(df["smiles"].tolist())
    xb = x.astype(bool)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    if any(np.unique(y[idx]).size < 2 for idx in (train_idx, valid_idx, test_idx)):
        return []

    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    rf_valid = rf.predict_proba(x[valid_idx])[:, 1]
    rf_test = rf.predict_proba(x[test_idx])[:, 1]
    anchor_valid = compute_anchor_features(xb[train_idx], y[train_idx], xb[valid_idx], n_neighbors=top_k)
    anchor_test = compute_anchor_features(xb[train_idx], y[train_idx], xb[test_idx], n_neighbors=top_k)
    reasoner = fit_reasoner(rf_valid, anchor_valid, y[valid_idx], seed=seed)
    reasoning_test = apply_reasoner(reasoner, rf_test, anchor_test)

    nn = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=min(top_k, len(train_idx)))
    nn.fit(xb[train_idx])
    distances, indices = nn.kneighbors(xb[test_idx], return_distance=True)
    train_smiles = df.iloc[train_idx]["smiles"].tolist()
    train_y = y[train_idx]

    rows = []
    for row_pos, test_row_idx in enumerate(test_idx):
        neighbor_global = train_idx[indices[row_pos]]
        neighbor_smiles = [train_smiles[int(local_idx)] for local_idx in indices[row_pos]]
        neighbor_labels = train_y[indices[row_pos]]
        similarities = 1.0 - distances[row_pos]
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "test_index": int(test_row_idx),
                "smiles": df.iloc[test_row_idx]["smiles"],
                "y_true": int(y[test_row_idx]),
                "rf_prob": float(rf_test[row_pos]),
                "anchor_prob": float(anchor_test["anchor_prob"][row_pos]),
                "anchor_disagreement": float(anchor_test["anchor_disagreement"][row_pos]),
                "reasoning_prob": float(reasoning_test[row_pos]),
                "rf_pred": int(rf_test[row_pos] >= 0.5),
                "reasoning_pred": int(reasoning_test[row_pos] >= 0.5),
                "rf_correct": bool((rf_test[row_pos] >= 0.5) == y[test_row_idx]),
                "reasoning_correct": bool((reasoning_test[row_pos] >= 0.5) == y[test_row_idx]),
                "confidence": float(abs(reasoning_test[row_pos] - 0.5) * 2.0),
                "max_train_similarity": float(np.max(similarities)),
                "ood_score": float(1.0 - np.max(similarities)),
                "top_anchor_indices": ",".join(str(int(idx)) for idx in neighbor_global),
                "top_anchors": compact_anchor_text(neighbor_smiles, neighbor_labels, similarities),
            }
        )
    case_df = pd.DataFrame(rows)
    selected = pd.concat(
        [
            pick_cases(case_df, "high_conf_correct_ood", (case_df["max_train_similarity"] < 0.5) & case_df["reasoning_correct"], max_cases),
            pick_cases(case_df, "high_conf_error_ood", (case_df["max_train_similarity"] < 0.5) & (~case_df["reasoning_correct"]), max_cases),
            pick_cases(case_df, "anchor_rescue", (~case_df["rf_correct"]) & case_df["reasoning_correct"], max_cases),
            pick_cases(case_df, "anchor_failure", case_df["rf_correct"] & (~case_df["reasoning_correct"]), max_cases),
        ],
        ignore_index=True,
    )
    return selected.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "anchor_case_studies")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}", flush=True)
        rows.extend(run_task(task, args.seed, args.rf_n_jobs, args.top_k, args.max_cases))
        pd.DataFrame(rows).to_csv(args.output_dir / "anchor_case_studies.csv", index=False)
    summary = {"rows": len(rows), "tasks": len(TASKS), "top_k": args.top_k, "max_cases_per_category": args.max_cases}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
