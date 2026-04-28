from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.features import morgan_fingerprint_matrix

from run_reliability_benchmark import TASKS, evaluate_probs, filter_valid_smiles, fit_rf, load_task_frame, make_scaffold_split


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def temperature_scale(valid_y: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray) -> tuple[np.ndarray, float]:
    valid_logits = logit(valid_prob)
    test_logits = logit(test_prob)

    def objective(temp: float) -> float:
        return float(log_loss(valid_y, sigmoid(valid_logits / temp), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    temp = float(result.x)
    return sigmoid(test_logits / temp), temp


def platt_scale(valid_y: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    model.fit(logit(valid_prob).reshape(-1, 1), valid_y)
    return model.predict_proba(logit(test_prob).reshape(-1, 1))[:, 1]


def isotonic_scale(valid_y: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(valid_prob, valid_y)
    return model.predict(test_prob)


def run_task(task_cfg: dict, seed: int, rf_n_jobs: int | None) -> list[dict]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    x = morgan_fingerprint_matrix(df["smiles"].tolist())
    train_idx = np.asarray(split["train"], dtype=np.int64)
    valid_idx = np.asarray(split["valid"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    rf = fit_rf(x[train_idx], y[train_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    valid_prob = rf.predict_proba(x[valid_idx])[:, 1]
    test_prob = rf.predict_proba(x[test_idx])[:, 1]
    temp_prob, temp = temperature_scale(y[valid_idx], valid_prob, test_prob)
    calibrated = {
        "uncalibrated": test_prob,
        "temperature": temp_prob,
        "platt": platt_scale(y[valid_idx], valid_prob, test_prob),
        "isotonic": isotonic_scale(y[valid_idx], valid_prob, test_prob),
    }
    rows = []
    for method, probs in calibrated.items():
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "base_model": "rf_morgan",
                "calibration": method,
                "temperature": temp if method == "temperature" else np.nan,
                "n_valid": int(len(valid_idx)),
                "n_test": int(len(test_idx)),
                **evaluate_probs(y[test_idx], np.clip(probs, 0.0, 1.0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "calibration_benchmark")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        print(f"RUN {task['dataset']}::{task['label']}")
        rows.extend(run_task(task, seed=args.seed, rf_n_jobs=args.rf_n_jobs))
    pd.DataFrame(rows).to_csv(args.output_dir / "calibration_results.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
