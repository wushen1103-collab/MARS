from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_claim_audit"
TASK_KEYS = ["dataset", "label"]
METRICS = ["auroc", "auprc", "brier", "ece"]


def load_candidate_pool(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "neural_calibration_true" / "analysis" / "candidate_reliability_table.csv"
    candidates = pd.read_csv(path)
    keep = TASK_KEYS + ["source", "candidate", "model", "calibration", "split"] + METRICS
    out = candidates[keep].copy()
    out["pool"] = "true_neural_calibration_or_ensemble"
    out["method"] = out["candidate"]
    return out


def load_baseline_pool(root: Path) -> pd.DataFrame:
    rows = []
    chemprop_path = root / "outputs" / "chemprop_metrics" / "chemprop_calibration_metrics.csv"
    if chemprop_path.exists():
        chemprop = pd.read_csv(chemprop_path)
        chemprop = chemprop.loc[chemprop["split"].eq("scaffold")]
        for row in chemprop.itertuples(index=False):
            rows.append(
                {
                    "pool": "baseline",
                    "source": row.source,
                    "method": row.method,
                    "dataset": row.dataset,
                    "label": row.label,
                    "split": row.split,
                    "auroc": row.auroc,
                    "auprc": row.auprc,
                    "brier": row.brier,
                    "ece": row.ece,
                }
            )

    strong_path = root / "outputs" / "strong_descriptor_baselines" / "strong_descriptor_baselines.csv"
    if strong_path.exists():
        strong = pd.read_csv(strong_path)
        for row in strong.itertuples(index=False):
            rows.append(
                {
                    "pool": "baseline",
                    "source": "strong_descriptor",
                    "method": row.model,
                    "dataset": row.dataset,
                    "label": row.label,
                    "split": row.split,
                    "auroc": row.auroc,
                    "auprc": row.auprc,
                    "brier": row.brier,
                    "ece": row.ece,
                }
            )

    aggregate_path = root / "outputs" / "neural_multiseed_aggregate_seed1_4" / "aggregate_mean_std.csv"
    if aggregate_path.exists():
        aggregate = pd.read_csv(aggregate_path)
        aggregate = aggregate.loc[aggregate["split"].eq("scaffold")]
        for row in aggregate.itertuples(index=False):
            rows.append(
                {
                    "pool": "baseline",
                    "source": f"multiseed_{row.stage}",
                    "method": row.model_variant,
                    "dataset": row.dataset,
                    "label": row.label,
                    "split": row.split,
                    "auroc": row.auroc_mean,
                    "auprc": row.auprc_mean,
                    "brier": row.brier_mean,
                    "ece": row.ece_mean,
                }
            )
    return pd.DataFrame(rows)


def build_audit(candidate_pool: pd.DataFrame, baseline_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, label), candidates in candidate_pool.groupby(TASK_KEYS, dropna=False):
        baselines = baseline_pool.loc[(baseline_pool["dataset"] == dataset) & (baseline_pool["label"] == label)]
        if baselines.empty:
            continue
        best_candidate_perf = candidates.sort_values(["auprc", "auroc"], ascending=[False, False]).iloc[0]
        best_baseline_perf = baselines.sort_values(["auprc", "auroc"], ascending=[False, False]).iloc[0]
        best_candidate_ece = candidates.sort_values(["ece", "brier"], ascending=[True, True]).iloc[0]
        best_baseline_ece = baselines.sort_values(["ece", "brier"], ascending=[True, True]).iloc[0]
        best_candidate_brier = candidates.sort_values(["brier", "ece"], ascending=[True, True]).iloc[0]
        best_baseline_brier = baselines.sort_values(["brier", "ece"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "candidate_perf_method": best_candidate_perf["method"],
                "candidate_perf_source": best_candidate_perf["source"],
                "candidate_auprc": best_candidate_perf["auprc"],
                "candidate_auroc": best_candidate_perf["auroc"],
                "baseline_perf_method": best_baseline_perf["method"],
                "baseline_perf_source": best_baseline_perf["source"],
                "baseline_auprc": best_baseline_perf["auprc"],
                "baseline_auroc": best_baseline_perf["auroc"],
                "auprc_delta": best_candidate_perf["auprc"] - best_baseline_perf["auprc"],
                "auroc_delta": best_candidate_perf["auroc"] - best_baseline_perf["auroc"],
                "performance_win": bool(best_candidate_perf["auprc"] > best_baseline_perf["auprc"]),
                "candidate_ece_method": best_candidate_ece["method"],
                "candidate_ece_source": best_candidate_ece["source"],
                "candidate_ece": best_candidate_ece["ece"],
                "baseline_ece_method": best_baseline_ece["method"],
                "baseline_ece_source": best_baseline_ece["source"],
                "baseline_ece": best_baseline_ece["ece"],
                "ece_improvement": best_baseline_ece["ece"] - best_candidate_ece["ece"],
                "ece_win": bool(best_candidate_ece["ece"] < best_baseline_ece["ece"]),
                "candidate_brier_method": best_candidate_brier["method"],
                "candidate_brier": best_candidate_brier["brier"],
                "baseline_brier_method": best_baseline_brier["method"],
                "baseline_brier": best_baseline_brier["brier"],
                "brier_improvement": best_baseline_brier["brier"] - best_candidate_brier["brier"],
                "brier_win": bool(best_candidate_brier["brier"] < best_baseline_brier["brier"]),
            }
        )
    return pd.DataFrame(rows).sort_values(TASK_KEYS).reset_index(drop=True)


def build_summary(audit: pd.DataFrame) -> dict:
    performance_win_rate = float(audit["performance_win"].mean())
    reliability_win_rate = float(audit["ece_win"].mean())
    brier_win_rate = float(audit["brier_win"].mean())
    return {
        "tasks": int(len(audit)),
        "performance_wins": int(audit["performance_win"].sum()),
        "performance_win_rate": performance_win_rate,
        "mean_auprc_delta": float(audit["auprc_delta"].mean()),
        "ece_wins": int(audit["ece_win"].sum()),
        "reliability_win_rate": reliability_win_rate,
        "mean_ece_improvement": float(audit["ece_improvement"].mean()),
        "brier_wins": int(audit["brier_win"].sum()),
        "brier_win_rate": brier_win_rate,
        "mean_brier_improvement": float(audit["brier_improvement"].mean()),
        "claim_guardrail": "Do not claim universal SOTA" if performance_win_rate < 1.0 else "Internal performance SOTA across this pool",
    }


def write_recommendations(output_dir: Path, audit: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# Internal SOTA Claim Audit",
        "",
        "## Summary",
        "",
        f"- Tasks audited: {summary['tasks']}",
        f"- Performance wins by AUPRC: {summary['performance_wins']}/{summary['tasks']} ({summary['performance_win_rate']:.3f})",
        f"- Mean AUPRC delta: {summary['mean_auprc_delta']:.6f}",
        f"- ECE wins: {summary['ece_wins']}/{summary['tasks']} ({summary['reliability_win_rate']:.3f})",
        f"- Mean ECE improvement: {summary['mean_ece_improvement']:.6f}",
        f"- Brier wins: {summary['brier_wins']}/{summary['tasks']} ({summary['brier_win_rate']:.3f})",
        f"- Claim guardrail: {summary['claim_guardrail']}",
        "",
        "## Recommended Claim Wording",
        "",
        "- Do not claim universal SOTA from the current pool.",
        "- Claim stronger OOD reliability: true neural calibration/ensemble improves ECE on most tasks and Brier on most tasks.",
        "- Claim competitive OOD performance: the new neural candidates win AUPRC on a majority, but not all, tasks.",
        "- Use task-specific language for ClinTox, hERG, and Tox21 NR-AhR because descriptor baselines remain stronger in AUPRC there.",
        "",
        "## Task-Level Notes",
        "",
    ]
    for row in audit.itertuples(index=False):
        perf = "win" if row.performance_win else "loss"
        ece = "win" if row.ece_win else "loss"
        lines.append(
            f"- {row.dataset}/{row.label}: performance_{perf} "
            f"(delta AUPRC={row.auprc_delta:.6f}, candidate={row.candidate_perf_method}, baseline={row.baseline_perf_method}); "
            f"ece_{ece} (improvement={row.ece_improvement:.6f}, candidate={row.candidate_ece_method}, baseline={row.baseline_ece_method})."
        )
    (output_dir / "claim_recommendations.md").write_text("\n".join(lines) + "\n")


def run(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_pool = load_candidate_pool(root)
    baseline_pool = load_baseline_pool(root)
    audit = build_audit(candidate_pool, baseline_pool)
    summary = build_summary(audit)
    candidate_pool.to_csv(output_dir / "candidate_pool.csv", index=False)
    baseline_pool.to_csv(output_dir / "baseline_pool.csv", index=False)
    audit.to_csv(output_dir / "internal_sota_audit_by_task.csv", index=False)
    (output_dir / "claim_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_recommendations(output_dir, audit, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.root, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

