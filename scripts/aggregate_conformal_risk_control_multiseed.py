from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LOWER_IS_BETTER = {"brier", "ece", "adaptive_ece", "classwise_ece", "avg_set_size", "singleton_error", "test_error"}


def read_shards(input_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob(f"seed*/{filename}")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = int(path.parent.name.removeprefix("seed"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_std(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = [col for col in value_cols if col in frame.columns]
    out = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns.to_flat_index()]
    return out


def paired_effect(
    frame: pd.DataFrame,
    *,
    first: str,
    second: str,
    metric: str,
    pair_cols: list[str],
    model_col: str = "model",
    seed: int = 42,
) -> dict:
    sub = frame[frame[model_col].isin([first, second])].copy()
    wide = sub.pivot_table(index=pair_cols, columns=model_col, values=metric, aggfunc="first").dropna()
    raw = wide[first] - wide[second]
    if metric in LOWER_IS_BETTER:
        raw = -raw
    rng = np.random.default_rng(seed)
    values = raw.to_numpy(dtype=float)
    boot = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(10000)])
    p_value = 1.0 if np.allclose(values, 0.0) else float(wilcoxon(values, alternative="two-sided").pvalue)
    return {
        "comparison": f"{first} vs {second}",
        "metric": metric,
        "n_pairs": int(len(values)),
        "first_mean": float(wide[first].mean()),
        "second_mean": float(wide[second].mean()),
        "sign_adjusted_effect": float(np.mean(values)),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "revision" / "conformal_risk_control_shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "revision" / "conformal_risk_control_aggregate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = read_shards(args.input_dir, "base_model_metrics.csv")
    classwise = read_shards(args.input_dir, "classwise_calibration_metrics.csv")
    conformal = read_shards(args.input_dir, "conformal_set_metrics.csv")
    risk = read_shards(args.input_dir, "risk_control_metrics.csv")

    base.to_csv(args.output_dir / "base_model_metrics_all_seeds.csv", index=False)
    classwise.to_csv(args.output_dir / "classwise_calibration_all_seeds.csv", index=False)
    conformal.to_csv(args.output_dir / "conformal_set_metrics_all_seeds.csv", index=False)
    risk.to_csv(args.output_dir / "risk_control_metrics_all_seeds.csv", index=False)

    mean_std(base, ["model"], ["auroc", "auprc", "brier", "ece"]).to_csv(args.output_dir / "base_model_mean_std.csv", index=False)
    mean_std(classwise, ["model"], ["adaptive_ece", "class_0_ece", "class_1_ece", "classwise_ece"]).to_csv(args.output_dir / "classwise_calibration_mean_std.csv", index=False)
    mean_std(conformal, ["model", "alpha"], ["coverage", "avg_set_size", "singleton_rate", "ambiguous_rate", "empty_rate", "singleton_error"]).to_csv(args.output_dir / "conformal_set_mean_std.csv", index=False)
    mean_std(risk, ["model", "target_risk"], ["test_coverage", "test_error", "test_kept", "valid_selected_coverage", "valid_selected_empirical_risk", "valid_selected_wilson_upper"]).to_csv(args.output_dir / "risk_control_mean_std.csv", index=False)

    rows = []
    for metric in ["coverage", "avg_set_size", "singleton_rate", "singleton_error"]:
        rows.append(
            {
                "block": "conformal_alpha_0.10",
                **paired_effect(
                    conformal[conformal["alpha"].round(6) == 0.1],
                    first="anchor_reasoning",
                    second="rf_morgan",
                    metric=metric,
                    pair_cols=["dataset", "label", "seed", "alpha"],
                ),
            }
        )
    for metric in ["test_coverage", "test_error"]:
        rows.append(
            {
                "block": "risk_control_target_0.10",
                **paired_effect(
                    risk[risk["target_risk"].round(6) == 0.1],
                    first="anchor_reasoning",
                    second="rf_morgan",
                    metric=metric,
                    pair_cols=["dataset", "label", "seed", "target_risk"],
                ),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(args.output_dir / "conformal_risk_control_paired_stats.csv", index=False)

    summary = {
        "base_rows": int(len(base)),
        "classwise_rows": int(len(classwise)),
        "conformal_rows": int(len(conformal)),
        "risk_rows": int(len(risk)),
        "seeds": sorted(base["seed"].dropna().astype(int).unique().tolist()) if not base.empty else [],
        "paired_comparisons": int(len(stats)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
