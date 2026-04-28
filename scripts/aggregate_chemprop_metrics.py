from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_shift_reliability.chemprop_compat import (  # noqa: E402
    patch_pandas_rdkit_compat,
    patch_torch_load_weights_only_false,
)


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
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * (int(np.sum(mask)) / len(y_true))
    return float(ece)


def risk_coverage_auc(y_true: np.ndarray, y_prob: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-confidence)
    y_sorted = y_true[order]
    pred_sorted = (y_prob[order] >= 0.5).astype(int)
    errors = (pred_sorted != y_sorted).astype(float)
    coverages = np.arange(1, len(errors) + 1, dtype=float) / len(errors)
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1, dtype=float)
    return float(np.trapz(risks, coverages))


def error_detection_auroc(y_true: np.ndarray, y_prob: np.ndarray, confidence: np.ndarray) -> float:
    pred = (y_prob >= 0.5).astype(int)
    errors = (pred != y_true).astype(int)
    if np.unique(errors).size < 2:
        return float("nan")
    return float(roc_auc_score(errors, 1.0 - confidence))


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "nll": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "risk_coverage_auc": risk_coverage_auc(y_true, y_prob, confidence),
        "error_detection_auroc": error_detection_auroc(y_true, y_prob, confidence),
        "positive_rate": float(np.mean(y_true)),
    }


def parse_run_name(run_dir: Path) -> tuple[str, str, str, str]:
    parts = run_dir.name.split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse Chemprop run directory name: {run_dir.name}")
    dataset, label, split = parts[:3]
    tag = "__".join(parts[3:]) if len(parts) > 3 else "single"
    return dataset, label, split, tag


def read_args(run_dir: Path) -> dict:
    path = run_dir / "args.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def normalize_smiles_cell(value: object) -> str:
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            return text
    return text


def read_prediction_probs(pred_path: Path, label: str) -> tuple[list[str], np.ndarray]:
    pred_df = pd.read_csv(pred_path)
    if "smiles" not in pred_df.columns:
        raise ValueError(f"Missing smiles column in {pred_path}")
    prob_col = label if label in pred_df.columns else pred_df.columns[-1]
    smiles = [normalize_smiles_cell(item) for item in pred_df["smiles"]]
    probs = pred_df[prob_col].astype(float).to_numpy()
    return smiles, probs


def read_truth(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected smiles,label columns in {path}")
    return df["smiles"].astype(str).tolist(), df["label"].astype(int).to_numpy()


def aligned_truth(split_path: Path, pred_path: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    truth_smiles, y_true = read_truth(split_path)
    pred_smiles, probs = read_prediction_probs(pred_path, label)
    if len(truth_smiles) != len(pred_smiles):
        raise ValueError(f"Length mismatch truth={len(truth_smiles)} preds={len(pred_smiles)} for {pred_path}")
    if truth_smiles != pred_smiles:
        truth = pd.DataFrame({"smiles": truth_smiles, "y_true": y_true})
        pred = pd.DataFrame({"smiles": pred_smiles, "prob": probs})
        merged = pred.merge(truth, on="smiles", how="left")
        if merged["y_true"].isna().any():
            raise ValueError(f"Could not align predictions to truth for {pred_path}")
        y_true = merged["y_true"].astype(int).to_numpy()
        probs = merged["prob"].astype(float).to_numpy()
    return y_true, probs


def maybe_generate_predictions(run_dir: Path, data_path: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    patch_pandas_rdkit_compat()
    patch_torch_load_weights_only_false()
    from chemprop.train import chemprop_predict

    argv = [
        "chemprop_predict",
        "--test_path",
        str(data_path),
        "--checkpoint_dir",
        str(run_dir),
        "--preds_path",
        str(out_path),
        "--smiles_columns",
        "smiles",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        chemprop_predict()
    finally:
        sys.argv = old_argv


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(valid_y: np.ndarray, valid_prob: np.ndarray) -> float:
    logits = logit(valid_prob)
    grid = np.exp(np.linspace(math.log(0.05), math.log(10.0), 121))
    losses = []
    for temp in grid:
        losses.append(log_loss(valid_y, sigmoid(logits / temp), labels=[0, 1]))
    return float(grid[int(np.argmin(losses))])


def calibrated_variants(valid_y: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    variants = {"uncalibrated": test_prob}
    if np.unique(valid_y).size < 2:
        return variants

    platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    platt.fit(logit(valid_prob).reshape(-1, 1), valid_y)
    variants["platt"] = platt.predict_proba(logit(test_prob).reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(valid_prob, valid_y)
    variants["isotonic"] = np.clip(iso.predict(test_prob), 0.0, 1.0)

    temp = fit_temperature(valid_y, valid_prob)
    variants["temperature"] = sigmoid(logit(test_prob) / temp)
    return variants


def run_dir_metrics(run_dir: Path, generate_valid_preds: bool) -> list[dict]:
    dataset, label, split, tag = parse_run_name(run_dir)
    args = read_args(run_dir)
    test_path = Path(args.get("separate_test_path", ""))
    valid_path = Path(args.get("separate_val_path", ""))
    test_pred_path = run_dir / "test_preds.csv"
    if not test_pred_path.exists():
        test_pred_path = run_dir / "fold_0" / "test_preds.csv"
    if not test_path.exists() or not test_pred_path.exists():
        return []

    valid_pred_path = run_dir / "valid_preds.csv"
    has_valid = valid_path.exists()
    if generate_valid_preds and has_valid:
        maybe_generate_predictions(run_dir, valid_path, valid_pred_path)

    test_y, test_prob = aligned_truth(test_path, test_pred_path, label)
    if has_valid and valid_pred_path.exists():
        valid_y, valid_prob = aligned_truth(valid_path, valid_pred_path, label)
    else:
        valid_y, valid_prob = np.array([], dtype=int), np.array([], dtype=float)

    seed_text = valid_path.parent.name if has_valid else ""
    seed = int(seed_text.rsplit("seed", 1)[-1]) if "seed" in seed_text else 42
    variants = calibrated_variants(valid_y, valid_prob, test_prob, seed=seed) if len(valid_y) else {"uncalibrated": test_prob}
    rows = []
    for calibration, probs in variants.items():
        rows.append(
            {
                "pool": "baseline",
                "source": "chemprop_real",
                "method": f"chemprop_{tag}_{calibration}",
                "dataset": dataset,
                "label": label,
                "split": split,
                "tag": tag,
                "calibration": calibration,
                "ensemble_size": args.get("ensemble_size"),
                "loss_function": args.get("loss_function"),
                "n_test": int(len(test_y)),
                "n_valid": int(len(valid_y)),
                **evaluate_probs(test_y, probs),
            }
        )
    return rows


def run(root: Path, output_dir: Path, generate_valid_preds: bool, chemprop_root: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    chemprop_root = chemprop_root or (root / "outputs" / "chemprop_baseline")
    for run_dir in sorted(item for item in chemprop_root.glob("*") if item.is_dir()):
        rows.extend(run_dir_metrics(run_dir, generate_valid_preds=generate_valid_preds))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "chemprop_calibration_metrics.csv", index=False)
    summary = {
        "rows": int(len(metrics)),
        "runs": int(metrics[["dataset", "label", "split", "tag"]].drop_duplicates().shape[0]) if not metrics.empty else 0,
        "tasks": int(metrics[["dataset", "label"]].drop_duplicates().shape[0]) if not metrics.empty else 0,
        "generated_valid_predictions": bool(generate_valid_preds),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--chemprop-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "chemprop_metrics_20260422")
    parser.add_argument("--generate-valid-preds", action="store_true")
    args = parser.parse_args()
    chemprop_root = args.chemprop_root
    if chemprop_root is not None and not chemprop_root.is_absolute():
        chemprop_root = args.root / chemprop_root
    summary = run(args.root, args.output_dir, generate_valid_preds=args.generate_valid_preds, chemprop_root=chemprop_root)
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
