from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_leakage_shift_uq_bootstrap import nearest_train_similarity  # noqa: E402
from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    fit_rf,
    load_task_frame,
    make_scaffold_split,
    split_is_usable,
)
from run_strict_ood_model_matrix import apply_reasoner, fit_reasoner  # noqa: E402


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def safe_eval(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan}
    return evaluate_probs(y_true, probs)


def make_strata(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    return [
        ("nearest_similarity", "<0.30", frame["nearest_similarity"] < 0.30),
        ("nearest_similarity", "0.30-0.50", (frame["nearest_similarity"] >= 0.30) & (frame["nearest_similarity"] < 0.50)),
        ("nearest_similarity", ">=0.50", frame["nearest_similarity"] >= 0.50),
        ("anchor_disagreement", "<0.10", frame["anchor_disagreement"] < 0.10),
        ("anchor_disagreement", "0.10-0.20", (frame["anchor_disagreement"] >= 0.10) & (frame["anchor_disagreement"] < 0.20)),
        ("anchor_disagreement", ">=0.20", frame["anchor_disagreement"] >= 0.20),
        ("anchor_novelty", "<0.50", frame["anchor_novelty"] < 0.50),
        ("anchor_novelty", "0.50-0.70", (frame["anchor_novelty"] >= 0.50) & (frame["anchor_novelty"] < 0.70)),
        ("anchor_novelty", ">=0.70", frame["anchor_novelty"] >= 0.70),
    ]


def run_task(task: dict, seed: int, rf_n_jobs: int) -> tuple[list[dict], list[dict]]:
    df = load_task_frame(task).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        raise ValueError(f"{task['dataset']}::{task['label']} unusable split: {reason}")

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
    nearest_similarity, _ = nearest_train_similarity(xb[train_idx], xb[test_idx])

    samples = pd.DataFrame(
        {
            "dataset": task["dataset"],
            "label": task["label"],
            "seed": seed,
            "y_true": y[test_idx],
            "rf_prob": test_prob,
            "anchor_reasoning_prob": reasoning_prob,
            "nearest_similarity": nearest_similarity,
            "anchor_disagreement": test_anchor["anchor_disagreement"],
            "anchor_novelty": test_anchor["anchor_distance_mean"],
        }
    )
    metric_rows = []
    for dimension, stratum, mask in make_strata(samples):
        subset = samples.loc[mask]
        for model, column in [("rf_morgan", "rf_prob"), ("anchor_reasoning", "anchor_reasoning_prob")]:
            metrics = safe_eval(subset["y_true"].to_numpy(), subset[column].to_numpy())
            metric_rows.append(
                {
                    "dataset": task["dataset"],
                    "label": task["label"],
                    "seed": seed,
                    "dimension": dimension,
                    "stratum": stratum,
                    "model": model,
                    "n": int(len(subset)),
                    "positive_rate": float(subset["y_true"].mean()) if len(subset) else np.nan,
                    **metrics,
                }
            )
    return samples.to_dict("records"), metric_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "anchor_stratified_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    metric_rows = []
    failures = []
    for seed in parse_ints(args.seeds):
        for task in TASKS:
            print(f"RUN {task['dataset']}::{task['label']} seed={seed}", flush=True)
            try:
                samples, metrics = run_task(task, seed=seed, rf_n_jobs=args.rf_n_jobs)
                sample_rows.extend(samples)
                metric_rows.extend(metrics)
            except Exception as exc:
                failures.append({"dataset": task["dataset"], "label": task["label"], "seed": seed, "error": repr(exc)})
    pd.DataFrame(sample_rows).to_csv(args.output_dir / "anchor_stratified_samples.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "anchor_stratified_metrics.csv", index=False)
    if not metrics.empty:
        wide = metrics.pivot_table(
            index=["dataset", "label", "seed", "dimension", "stratum", "n", "positive_rate"],
            columns="model",
            values=["auroc", "auprc", "brier", "ece"],
        )
        wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
        wide = wide.reset_index()
        for metric in ["auroc", "auprc"]:
            wide[f"{metric}_delta_anchor_minus_rf"] = wide[f"{metric}_anchor_reasoning"] - wide[f"{metric}_rf_morgan"]
        for metric in ["brier", "ece"]:
            wide[f"{metric}_delta_anchor_minus_rf"] = wide[f"{metric}_rf_morgan"] - wide[f"{metric}_anchor_reasoning"]
        wide.to_csv(args.output_dir / "anchor_stratified_deltas.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "anchor_stratified_failures.csv", index=False)
    summary = {
        "sample_rows": len(sample_rows),
        "metric_rows": len(metric_rows),
        "failures": len(failures),
        "seeds": parse_ints(args.seeds),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
