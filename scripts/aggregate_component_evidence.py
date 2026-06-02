from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LOWER_IS_BETTER = {"brier", "ece", "risk_coverage_auc", "selective_error_50", "selective_error_80"}


def bootstrap_effect(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(10000)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_stats(
    frame: pd.DataFrame,
    *,
    block: str,
    first: str,
    second: str,
    metric: str,
    pair_cols: list[str],
    model_col: str = "model",
) -> dict:
    sub = frame[frame[model_col].isin([first, second])].copy()
    wide = sub.pivot_table(index=pair_cols, columns=model_col, values=metric, aggfunc="first").dropna()
    raw = wide[first] - wide[second]
    if metric in LOWER_IS_BETTER:
        raw = -raw
    values = raw.to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_effect(values)
    return {
        "block": block,
        "comparison": f"{first} vs {second}",
        "metric": metric,
        "n_pairs": int(len(values)),
        "first_mean": float(wide[first].mean()),
        "second_mean": float(wide[second].mean()),
        "sign_adjusted_effect": float(np.mean(values)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wilcoxon_p": 1.0 if np.allclose(values, 0.0) else float(wilcoxon(values, alternative="two-sided").pvalue),
    }


def mean_std(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    out = frame.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns.to_flat_index()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "revision" / "aggregate")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "revision" / "component_evidence")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strict = pd.read_csv(args.input_dir / "strict_ood_all_seeds.csv")
    transfer = pd.read_csv(args.input_dir / "transfer_all_seeds.csv")
    strict_conf = pd.read_csv(args.input_dir / "strict_ood_confidence_all_seeds.csv")
    metrics = ["auroc", "auprc", "brier", "ece"]

    rows = []
    for block, frame, pair_cols in [
        ("strict_ood", strict[strict["status"] == "ok"], ["dataset", "label", "split", "seed"]),
        ("transfer", transfer, ["source_dataset", "target_dataset", "seed"]),
    ]:
        for first, second in [
            ("retrieval_only", "rf_morgan"),
            ("anchor_reasoning", "retrieval_only"),
            ("anchor_reasoning", "rf_morgan"),
        ]:
            for metric in metrics:
                rows.append(
                    paired_stats(
                        frame,
                        block=block,
                        first=first,
                        second=second,
                        metric=metric,
                        pair_cols=pair_cols,
                    )
                )

    anchor_conf = strict_conf[strict_conf["model"] == "anchor_reasoning"].copy()
    for metric in ["error_detection_auroc", "risk_coverage_auc", "selective_error_50", "selective_error_80"]:
        rows.append(
            paired_stats(
                anchor_conf,
                block="strict_ood_confidence",
                first="learned_shift_error_model",
                second="prob_margin",
                metric=metric,
                pair_cols=["dataset", "label", "split", "seed"],
                model_col="confidence",
            )
        )

    stats = pd.DataFrame(rows)
    stats.to_csv(args.output_dir / "component_paired_stats.csv", index=False)
    mean_std(strict[strict["status"] == "ok"], ["split", "model"], metrics).to_csv(args.output_dir / "strict_component_mean_std.csv", index=False)
    mean_std(transfer, ["model"], metrics).to_csv(args.output_dir / "transfer_component_mean_std.csv", index=False)
    mean_std(anchor_conf, ["confidence"], ["error_detection_auroc", "risk_coverage_auc", "selective_error_50", "selective_error_80"]).to_csv(args.output_dir / "confidence_component_mean_std.csv", index=False)

    summary = {
        "strict_rows": int(len(strict)),
        "transfer_rows": int(len(transfer)),
        "strict_confidence_rows": int(len(anchor_conf)),
        "paired_comparisons": int(len(stats)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
