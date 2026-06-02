from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "neural_calibration_true"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "analysis"

LOWER_IS_BETTER = ["brier", "ece", "nll", "risk_coverage_auc"]
HIGHER_IS_BETTER = ["auroc", "auprc", "error_detection_auroc"]
METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER
RELIABILITY_METRICS = ["brier", "ece", "nll", "risk_coverage_auc", "error_detection_auroc"]
TASK_KEYS = ["dataset", "label"]
RUN_KEYS = ["dataset", "label", "model", "split", "seed"]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in frame.columns.to_flat_index()
    ]
    return frame


def signed_improvement(metric: str, value: pd.Series, baseline: pd.Series) -> pd.Series:
    if metric in LOWER_IS_BETTER:
        return baseline - value
    if metric in HIGHER_IS_BETTER:
        return value - baseline
    raise KeyError(metric)


def aggregate_single_runs(calibration: pd.DataFrame) -> pd.DataFrame:
    grouped = calibration.groupby(["dataset", "label", "model", "split", "calibration"], dropna=False)
    out = grouped[METRICS].agg(["mean", "std", "count"]).reset_index()
    out = flatten_columns(out)
    for metric in METRICS[1:]:
        count_col = f"{metric}_count"
        if count_col in out.columns:
            out = out.drop(columns=[count_col])
    out = out.rename(columns={f"{METRICS[0]}_count": "n"})
    return out


def single_run_delta_vs_uncalibrated(calibration: pd.DataFrame) -> pd.DataFrame:
    base = calibration.loc[calibration["calibration"] == "uncalibrated", RUN_KEYS + METRICS].copy()
    base = base.rename(columns={metric: f"base_{metric}" for metric in METRICS})
    compare = calibration.loc[calibration["calibration"] != "uncalibrated"].copy()
    merged = compare.merge(base, on=RUN_KEYS, how="left", validate="many_to_one")
    for metric in METRICS:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"base_{metric}"]
        merged[f"{metric}_improvement"] = signed_improvement(metric, merged[metric], merged[f"base_{metric}"])
        merged[f"{metric}_win"] = merged[f"{metric}_improvement"] > 0
    merged["reliability_win_fraction"] = merged[[f"{m}_win" for m in RELIABILITY_METRICS]].mean(axis=1)
    return merged


def summarize_method_level(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, calibration), group in delta.groupby(["model", "calibration"], dropna=False):
        row = {
            "source": "single_run",
            "model": model,
            "calibration": calibration,
            "n_comparisons": int(len(group)),
            "reliability_win_fraction_mean": float(group["reliability_win_fraction"].mean()),
        }
        for metric in METRICS:
            row[f"{metric}_improvement_mean"] = float(group[f"{metric}_improvement"].mean())
            row[f"{metric}_improvement_median"] = float(group[f"{metric}_improvement"].median())
            row[f"{metric}_win_rate"] = float(group[f"{metric}_win"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_delta_vs_single(calibration: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    single_uncal = calibration.loc[calibration["calibration"] == "uncalibrated"].copy()
    single_mean = (
        single_uncal.groupby(["dataset", "label", "model", "split"], dropna=False)[METRICS]
        .mean()
        .reset_index()
        .rename(columns={metric: f"single_uncal_{metric}" for metric in METRICS})
    )
    ens = ensemble.copy()
    ens["base_model"] = ens["model"].str.replace("_deep_ensemble", "", regex=False)
    merged = ens.merge(
        single_mean,
        left_on=["dataset", "label", "base_model", "split"],
        right_on=["dataset", "label", "model", "split"],
        how="left",
        suffixes=("", "_single"),
        validate="many_to_one",
    )
    if "model_single" in merged.columns:
        merged = merged.drop(columns=["model_single"])
    for metric in METRICS:
        baseline = merged[f"single_uncal_{metric}"]
        merged[f"{metric}_delta"] = merged[metric] - baseline
        merged[f"{metric}_improvement"] = signed_improvement(metric, merged[metric], baseline)
        merged[f"{metric}_win"] = merged[f"{metric}_improvement"] > 0
    merged["reliability_win_fraction"] = merged[[f"{m}_win" for m in RELIABILITY_METRICS]].mean(axis=1)
    return merged


def build_candidate_table(single_mean: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    single_rows = []
    for row in single_mean.itertuples(index=False):
        payload = {
            "source": "single_seed_mean",
            "candidate": f"{row.model}+{row.calibration}",
            "dataset": row.dataset,
            "label": row.label,
            "model": row.model,
            "calibration": row.calibration,
            "split": row.split,
            "n": int(row.n),
        }
        for metric in METRICS:
            payload[metric] = getattr(row, f"{metric}_mean")
        single_rows.append(payload)

    ensemble_rows = []
    for row in ensemble.itertuples(index=False):
        payload = {
            "source": "deep_ensemble",
            "candidate": f"{row.model}+{row.calibration}",
            "dataset": row.dataset,
            "label": row.label,
            "model": row.model,
            "calibration": row.calibration,
            "split": row.split,
            "n": int(row.min_test_members),
        }
        for metric in METRICS:
            payload[metric] = getattr(row, metric)
        ensemble_rows.append(payload)

    candidates = pd.DataFrame(single_rows + ensemble_rows)
    score_parts = []
    for _, group in candidates.groupby(TASK_KEYS, dropna=False):
        ranked = group.copy()
        rank_cols = []
        for metric in ["brier", "ece", "nll", "risk_coverage_auc"]:
            col = f"rank_{metric}"
            ranked[col] = ranked[metric].rank(method="average", ascending=True, pct=True)
            rank_cols.append(col)
        ranked["rank_error_detection_auroc"] = ranked["error_detection_auroc"].rank(
            method="average", ascending=False, pct=True
        )
        rank_cols.append("rank_error_detection_auroc")
        ranked["reliability_rank_score"] = ranked[rank_cols].mean(axis=1)
        score_parts.append(ranked)
    return pd.concat(score_parts, ignore_index=True)


def best_rows(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_reliability = (
        candidates.sort_values(["dataset", "label", "reliability_rank_score", "ece", "brier"])
        .groupby(TASK_KEYS, as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_performance = (
        candidates.sort_values(["dataset", "label", "auprc", "auroc"], ascending=[True, True, False, False])
        .groupby(TASK_KEYS, as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_reliability, best_performance


def build_headline_summary(
    calibration: pd.DataFrame,
    ensemble: pd.DataFrame,
    method_summary: pd.DataFrame,
    ensemble_delta: pd.DataFrame,
    best_reliability: pd.DataFrame,
    best_performance: pd.DataFrame,
) -> dict:
    def best_method(metric: str) -> dict:
        col = f"{metric}_improvement_mean"
        rows = method_summary.dropna(subset=[col])
        if rows.empty:
            return {}
        row = rows.sort_values(col, ascending=False).iloc[0]
        return {
            "model": row["model"],
            "calibration": row["calibration"],
            "mean_improvement": float(row[col]),
            "win_rate": float(row[f"{metric}_win_rate"]),
        }

    ens_uncal = ensemble_delta.loc[ensemble_delta["calibration"] == "uncalibrated"]
    ens_summary = {}
    for metric in ["auprc", "auroc", "ece", "brier", "nll", "risk_coverage_auc"]:
        col = f"{metric}_improvement"
        ens_summary[f"{metric}_mean_improvement_vs_single_uncalibrated"] = float(ens_uncal[col].mean())
        ens_summary[f"{metric}_win_rate_vs_single_uncalibrated"] = float((ens_uncal[col] > 0).mean())

    return {
        "single_calibration_rows": int(len(calibration)),
        "deep_ensemble_rows": int(len(ensemble)),
        "best_single_method_by_ece": best_method("ece"),
        "best_single_method_by_brier": best_method("brier"),
        "best_single_method_by_nll": best_method("nll"),
        "best_single_method_by_risk_coverage_auc": best_method("risk_coverage_auc"),
        "best_single_method_by_error_detection_auroc": best_method("error_detection_auroc"),
        "deep_ensemble_uncalibrated_summary": ens_summary,
        "best_reliability_candidate_counts": best_reliability["candidate"].value_counts().to_dict(),
        "best_performance_candidate_counts": best_performance["candidate"].value_counts().to_dict(),
    }


def write_report(
    output_dir: Path,
    headline: dict,
    method_summary: pd.DataFrame,
    ensemble_delta: pd.DataFrame,
    best_reliability: pd.DataFrame,
    best_performance: pd.DataFrame,
) -> None:
    lines = [
        "# True Neural Calibration Analysis",
        "",
        "## Coverage",
        "",
        f"- Single-run calibration rows: {headline['single_calibration_rows']}",
        f"- Deep ensemble rows: {headline['deep_ensemble_rows']}",
        f"- Best reliability tasks summarized: {len(best_reliability)}",
        f"- Best performance tasks summarized: {len(best_performance)}",
        "",
        "## Headline Findings",
        "",
    ]
    for key in [
        "best_single_method_by_ece",
        "best_single_method_by_brier",
        "best_single_method_by_nll",
        "best_single_method_by_risk_coverage_auc",
        "best_single_method_by_error_detection_auroc",
    ]:
        payload = headline[key]
        if payload:
            lines.append(
                f"- {key}: {payload['model']} + {payload['calibration']} "
                f"mean_improvement={payload['mean_improvement']:.6f}, win_rate={payload['win_rate']:.3f}"
            )
    lines.extend(["", "## Deep Ensemble Versus Single Uncalibrated", ""])
    ens_uncal = ensemble_delta.loc[ensemble_delta["calibration"] == "uncalibrated"]
    for metric in ["auprc", "auroc", "ece", "brier", "nll", "risk_coverage_auc"]:
        col = f"{metric}_improvement"
        lines.append(
            f"- {metric}: mean_improvement={ens_uncal[col].mean():.6f}, "
            f"win_rate={(ens_uncal[col] > 0).mean():.3f}"
        )
    lines.extend(["", "## Best Reliability Candidates By Task", ""])
    for row in best_reliability.itertuples(index=False):
        lines.append(
            f"- {row.dataset}/{row.label}: {row.candidate}, "
            f"source={row.source}, auroc={row.auroc:.4f}, auprc={row.auprc:.4f}, "
            f"ece={row.ece:.4f}, brier={row.brier:.4f}, risk_coverage_auc={row.risk_coverage_auc:.4f}"
        )
    lines.extend(["", "## Best Performance Candidates By Task", ""])
    for row in best_performance.itertuples(index=False):
        lines.append(
            f"- {row.dataset}/{row.label}: {row.candidate}, "
            f"source={row.source}, auroc={row.auroc:.4f}, auprc={row.auprc:.4f}, "
            f"ece={row.ece:.4f}, brier={row.brier:.4f}"
        )
    lines.extend(["", "## Files", ""])
    for name in [
        "single_run_mean_std.csv",
        "single_run_delta_vs_uncalibrated.csv",
        "deep_ensemble_delta_vs_single.csv",
        "best_reliability_by_task.csv",
        "best_performance_by_task.csv",
        "method_level_summary.csv",
        "headline_summary.json",
        "calibration_delta_summary.png",
        "deep_ensemble_tradeoff.png",
    ]:
        lines.append(f"- {name}")
    (output_dir / "neural_calibration_report.md").write_text("\n".join(lines) + "\n")


def make_figures(output_dir: Path, method_summary: pd.DataFrame, ensemble_delta: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_df = method_summary.copy()
    plot_df["method"] = plot_df["model"] + "+" + plot_df["calibration"]
    plot_df = plot_df.sort_values("ece_improvement_mean", ascending=False)
    top = plot_df.head(14)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(top))
    width = 0.38
    ax.bar(x - width / 2, top["ece_improvement_mean"], width=width, label="ECE improvement")
    ax.bar(x + width / 2, top["brier_improvement_mean"], width=width, label="Brier improvement")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top["method"], rotation=45, ha="right")
    ax.set_ylabel("Mean signed improvement")
    ax.set_title("Calibration Reliability Delta vs Uncalibrated")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_delta_summary.png", dpi=300)
    fig.savefig(output_dir / "calibration_delta_summary.pdf")
    plt.close(fig)

    ens = ensemble_delta.loc[ensemble_delta["calibration"] == "uncalibrated"].copy()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for model, group in ens.groupby("base_model", dropna=False):
        ax.scatter(group["auprc_improvement"], group["ece_improvement"], label=model, s=42, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("AUPRC improvement vs single uncalibrated")
    ax.set_ylabel("ECE improvement vs single uncalibrated")
    ax.set_title("Deep Ensemble Tradeoff")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "deep_ensemble_tradeoff.png", dpi=300)
    fig.savefig(output_dir / "deep_ensemble_tradeoff.pdf")
    plt.close(fig)


def run(input_dir: Path, output_dir: Path) -> dict:
    calibration_path = input_dir / "calibration_results.csv"
    ensemble_path = input_dir / "deep_ensemble_results.csv"
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    if not ensemble_path.exists():
        raise FileNotFoundError(ensemble_path)

    calibration = pd.read_csv(calibration_path)
    ensemble = pd.read_csv(ensemble_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_mean = aggregate_single_runs(calibration)
    single_delta = single_run_delta_vs_uncalibrated(calibration)
    method_summary = summarize_method_level(single_delta)
    ensemble_delta = ensemble_delta_vs_single(calibration, ensemble)
    candidates = build_candidate_table(single_mean, ensemble)
    best_reliability, best_performance = best_rows(candidates)
    headline = build_headline_summary(
        calibration=calibration,
        ensemble=ensemble,
        method_summary=method_summary,
        ensemble_delta=ensemble_delta,
        best_reliability=best_reliability,
        best_performance=best_performance,
    )

    single_mean.to_csv(output_dir / "single_run_mean_std.csv", index=False)
    single_delta.to_csv(output_dir / "single_run_delta_vs_uncalibrated.csv", index=False)
    ensemble_delta.to_csv(output_dir / "deep_ensemble_delta_vs_single.csv", index=False)
    candidates.to_csv(output_dir / "candidate_reliability_table.csv", index=False)
    best_reliability.to_csv(output_dir / "best_reliability_by_task.csv", index=False)
    best_performance.to_csv(output_dir / "best_performance_by_task.csv", index=False)
    method_summary.to_csv(output_dir / "method_level_summary.csv", index=False)
    (output_dir / "headline_summary.json").write_text(json.dumps(headline, indent=2, sort_keys=True))
    write_report(output_dir, headline, method_summary, ensemble_delta, best_reliability, best_performance)
    make_figures(output_dir, method_summary, ensemble_delta)
    return headline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    headline = run(args.input_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **headline}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
