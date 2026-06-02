from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.anchor_reliability import error_detection_auroc, risk_coverage_auc


DEFAULT_PREDICTION_DIR = ROOT / "outputs" / "neural_prediction_dump" / "predictions"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "neural_calibration_true"
REQUIRED_COLUMNS = {
    "dataset",
    "label",
    "model",
    "split",
    "seed",
    "part",
    "row_index",
    "smiles",
    "y_true",
    "logit",
    "prob",
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def logit_from_prob(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
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


def probability_confidence(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    return np.abs(probs - 0.5) * 2.0


def safe_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


def safe_auprc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, probs))


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    confidence = probability_confidence(probs) if confidence is None else np.asarray(confidence, dtype=np.float64)
    return {
        "auroc": safe_auroc(y_true, probs),
        "auprc": safe_auprc(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
        "nll": float(log_loss(y_true, probs, labels=[0, 1])),
        "risk_coverage_auc": risk_coverage_auc(y_true, probs, confidence),
        "error_detection_auroc": error_detection_auroc(y_true, probs, confidence),
        "positive_rate": float(np.mean(y_true)),
    }


def temperature_scale(
    valid_y: np.ndarray,
    valid_logits: np.ndarray,
    test_logits: np.ndarray,
) -> tuple[np.ndarray, float]:
    valid_y = np.asarray(valid_y).astype(int)
    valid_logits = np.asarray(valid_logits, dtype=np.float64)
    test_logits = np.asarray(test_logits, dtype=np.float64)
    if np.unique(valid_y).size < 2:
        return sigmoid(test_logits), float("nan")

    def objective(temp: float) -> float:
        return float(log_loss(valid_y, sigmoid(valid_logits / temp), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    temp = float(result.x) if result.success else 1.0
    return sigmoid(test_logits / temp), temp


def platt_scale(valid_y: np.ndarray, valid_logits: np.ndarray, test_logits: np.ndarray) -> np.ndarray:
    if np.unique(valid_y).size < 2:
        return sigmoid(test_logits)
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    model.fit(np.asarray(valid_logits).reshape(-1, 1), valid_y)
    return model.predict_proba(np.asarray(test_logits).reshape(-1, 1))[:, 1]


def isotonic_scale(valid_y: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    if np.unique(valid_y).size < 2:
        return test_prob
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(valid_prob, valid_y)
    return model.predict(test_prob)


def load_predictions(prediction_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(prediction_dir.glob("**/*.predictions.csv")) if prediction_dir.exists() else []
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), files
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    out["y_true"] = out["y_true"].astype(int)
    out["row_index"] = out["row_index"].astype(int)
    return out, files


def calibrate_single_runs(predictions: pd.DataFrame) -> list[dict]:
    if predictions.empty:
        return []
    rows = []
    group_cols = ["dataset", "label", "model", "split", "seed"]
    for (dataset, label, model_name, split_name, seed), run_df in predictions.groupby(group_cols, dropna=False):
        valid_df = run_df.loc[run_df["part"] == "valid"].sort_values(["row_index", "smiles"])
        test_df = run_df.loc[run_df["part"] == "test"].sort_values(["row_index", "smiles"])
        if valid_df.empty or test_df.empty:
            continue

        valid_y = valid_df["y_true"].to_numpy(dtype=int)
        test_y = test_df["y_true"].to_numpy(dtype=int)
        valid_prob = valid_df["prob"].to_numpy(dtype=np.float64)
        test_prob = test_df["prob"].to_numpy(dtype=np.float64)
        valid_logits = valid_df["logit"].to_numpy(dtype=np.float64)
        test_logits = test_df["logit"].to_numpy(dtype=np.float64)

        temp_prob, temp = temperature_scale(valid_y, valid_logits, test_logits)
        calibrated = [
            ("uncalibrated", test_prob, float("nan"), None),
            ("temperature", temp_prob, temp, None),
            ("platt", platt_scale(valid_y, valid_logits, test_logits), float("nan"), None),
            ("isotonic", isotonic_scale(valid_y, valid_prob, test_prob), float("nan"), None),
        ]
        if {"mc_prob_mean", "mc_prob_std"}.issubset(test_df.columns) and test_df["mc_prob_mean"].notna().any():
            mc_prob = test_df["mc_prob_mean"].to_numpy(dtype=np.float64)
            if "mc_confidence" in test_df.columns:
                mc_confidence = test_df["mc_confidence"].fillna(0.0).to_numpy(dtype=np.float64)
            else:
                mc_confidence = 1.0 - np.clip(test_df["mc_prob_std"].fillna(0.5).to_numpy(dtype=np.float64) / 0.5, 0.0, 1.0)
            calibrated.append(("mc_dropout", mc_prob, float("nan"), mc_confidence))

        for calibration, probs, temperature, confidence in calibrated:
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "model": model_name,
                    "split": split_name,
                    "seed": int(seed),
                    "calibration": calibration,
                    "temperature": temperature,
                    "n_valid": int(len(valid_df)),
                    "n_test": int(len(test_df)),
                    "source_files": ";".join(sorted(set(test_df["source_file"].astype(str)))),
                    **evaluate_probs(test_y, probs, confidence=confidence),
                }
            )
    return rows


def aligned_ensemble_arrays(part_df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    if value_col not in part_df.columns:
        return None
    index_cols = ["row_index", "smiles", "y_true"]
    pivot = part_df.pivot_table(index=index_cols, columns="seed", values=value_col, aggfunc="mean")
    if pivot.shape[1] < 2:
        return None
    pivot = pivot.dropna(axis=0, thresh=2)
    if pivot.empty:
        return None
    member_counts = pivot.notna().sum(axis=1)
    y_true = pivot.index.get_level_values("y_true").to_numpy(dtype=int)
    probs = pivot.mean(axis=1).to_numpy(dtype=np.float64)
    return y_true, probs, int(member_counts.min()), int(member_counts.max())


def calibrate_deep_ensembles(predictions: pd.DataFrame) -> list[dict]:
    if predictions.empty:
        return []
    rows = []
    group_cols = ["dataset", "label", "model", "split"]
    for (dataset, label, model_name, split_name), model_df in predictions.groupby(group_cols, dropna=False):
        valid = aligned_ensemble_arrays(model_df.loc[model_df["part"] == "valid"], "prob")
        test = aligned_ensemble_arrays(model_df.loc[model_df["part"] == "test"], "prob")
        if valid is None or test is None:
            continue

        valid_y, valid_prob, min_valid_members, max_valid_members = valid
        test_y, test_prob, min_test_members, max_test_members = test
        valid_logits = logit_from_prob(valid_prob)
        test_logits = logit_from_prob(test_prob)
        temp_prob, temp = temperature_scale(valid_y, valid_logits, test_logits)
        calibrated = [
            ("uncalibrated", test_prob, float("nan")),
            ("temperature", temp_prob, temp),
            ("platt", platt_scale(valid_y, valid_logits, test_logits), float("nan")),
            ("isotonic", isotonic_scale(valid_y, valid_prob, test_prob), float("nan")),
        ]
        for calibration, probs, temperature in calibrated:
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "model": f"{model_name}_deep_ensemble",
                    "split": split_name,
                    "seed": "ensemble",
                    "calibration": calibration,
                    "temperature": temperature,
                    "n_valid": int(len(valid_y)),
                    "n_test": int(len(test_y)),
                    "min_valid_members": min_valid_members,
                    "max_valid_members": max_valid_members,
                    "min_test_members": min_test_members,
                    "max_test_members": max_test_members,
                    "source_files": ";".join(sorted(set(model_df["source_file"].astype(str)))),
                    **evaluate_probs(test_y, probs),
                }
            )
    return rows


def write_outputs(
    *,
    output_dir: Path,
    prediction_files: list[Path],
    single_rows: list[dict],
    ensemble_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_df = pd.DataFrame(single_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    calibration_df.to_csv(output_dir / "calibration_results.csv", index=False)
    ensemble_df.to_csv(output_dir / "deep_ensemble_results.csv", index=False)
    summary = {
        "prediction_files": int(len(prediction_files)),
        "calibration_rows": int(len(calibration_df)),
        "deep_ensemble_rows": int(len(ensemble_df)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--expected-files", type=int, default=112)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    while True:
        predictions, files = load_predictions(args.prediction_dir)
        single_rows = calibrate_single_runs(predictions)
        ensemble_rows = calibrate_deep_ensembles(predictions)
        write_outputs(
            output_dir=args.output_dir,
            prediction_files=files,
            single_rows=single_rows,
            ensemble_rows=ensemble_rows,
        )
        status = {
            "prediction_files": len(files),
            "expected_files": args.expected_files,
            "calibration_rows": len(single_rows),
            "deep_ensemble_rows": len(ensemble_rows),
            "output_dir": str(args.output_dir),
        }
        if not args.watch or len(files) >= args.expected_files:
            print(json.dumps(status, indent=2))
            return
        print(f"WAIT {json.dumps(status)} next_poll_seconds={args.poll_seconds}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
