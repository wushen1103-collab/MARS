from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auroc", "auprc", "brier", "ece"]
LOWER_IS_BETTER = {"brier", "ece", "risk_coverage_auc"}


def read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_seed_shards(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = int(path.parent.name.removeprefix("seed"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    valid = np.where(np.isfinite(values))[0]
    if not len(valid):
        return adjusted.tolist()
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def paired_cluster_stats(
    frame: pd.DataFrame,
    *,
    block: str,
    first: str,
    second: str,
    pair_cols: list[str],
    cluster_cols: list[str],
    metric: str,
    value_col: str,
    model_col: str = "model",
    n_boot: int = 10000,
    seed: int = 20260531,
) -> dict:
    sub = frame[frame[model_col].isin([first, second])].copy()
    wide = sub.pivot_table(index=pair_cols, columns=model_col, values=value_col, aggfunc="first").dropna()
    if wide.empty:
        raise ValueError(f"No paired rows for {block}: {first} vs {second}, metric={metric}")
    raw = wide[first] - wide[second]
    if metric in LOWER_IS_BETTER:
        raw = -raw
    pair = raw.rename("effect").reset_index()
    cluster = pair.groupby(cluster_cols, as_index=False)["effect"].mean()
    cluster_values = cluster["effect"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [
            np.mean(rng.choice(cluster_values, size=len(cluster_values), replace=True))
            for _ in range(n_boot)
        ]
    )
    if np.allclose(cluster_values, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(cluster_values, alternative="two-sided").pvalue)
    return {
        "block": block,
        "comparison": f"{first} vs {second}",
        "metric": metric,
        "n_pairs": int(len(pair)),
        "n_clusters": int(len(cluster)),
        "first_mean": float(wide[first].mean()),
        "first_sd": float(wide[first].std(ddof=1)),
        "second_mean": float(wide[second].mean()),
        "second_sd": float(wide[second].std(ddof=1)),
        "sign_adjusted_effect": float(pair["effect"].mean()),
        "cluster_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "cluster_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_cluster_p": p_value,
    }


def strict_paths(root: Path) -> list[Path]:
    original = sorted((root / "outputs" / "strict_ood_model_matrix_shards_20260422").glob("seed4[2-4]_*/*metrics.csv"))
    revision = sorted((root / "outputs" / "revision_20260531" / "strict_ood_shards").glob("seed4[5-6]_*/*metrics.csv"))
    return original + revision


def recursive_metric_paths(input_dirs: list[Path], filename: str) -> list[Path]:
    return sorted(path for input_dir in input_dirs for path in input_dir.glob(f"**/{filename}"))


def add_mean_std(frame: pd.DataFrame, group_cols: list[str], output: Path) -> None:
    if frame.empty:
        return
    numeric = [column for column in frame.columns if column in METRICS or column in {"error_detection_auroc", "risk_coverage_auc"}]
    summary = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns.to_flat_index()]
    summary.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "revision_20260531" / "aggregate")
    parser.add_argument(
        "--strict-input-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing strict OOD metrics, recursively searched; may be repeated.",
    )
    parser.add_argument(
        "--transfer-input-dir",
        type=Path,
        default=ROOT / "outputs" / "revision_20260531" / "transfer_shards",
    )
    parser.add_argument(
        "--external-input-dir",
        type=Path,
        default=ROOT / "outputs" / "revision_20260531" / "external_admet_shards",
    )
    parser.add_argument("--reliability-input", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.strict_input_dir:
        strict_metric_files = recursive_metric_paths(args.strict_input_dir, "strict_ood_model_metrics.csv")
        strict_conf_files = recursive_metric_paths(args.strict_input_dir, "strict_ood_confidence_metrics.csv")
    else:
        legacy_strict_files = strict_paths(ROOT)
        strict_metric_files = [path for path in legacy_strict_files if path.name == "strict_ood_model_metrics.csv"]
        strict_conf_files = [path for path in legacy_strict_files if path.name == "strict_ood_confidence_metrics.csv"]
    transfer_files = sorted(args.transfer_input_dir.glob("seed*/cross_dataset_transfer_metrics.csv"))
    external_files = sorted(args.external_input_dir.glob("seed*/external_admet_probe_metrics.csv"))

    strict = read_many(strict_metric_files)
    strict_conf = read_many(strict_conf_files)
    transfer = read_many(transfer_files)
    external = read_seed_shards(external_files)
    reliability_input = args.reliability_input
    if reliability_input is None:
        revision_reliability = ROOT / "outputs" / "revision_20260531" / "reliability_benchmark_aggregate" / "all_results.csv"
        legacy_reliability = ROOT / "outputs" / "reliability_benchmark_expanded_multiseed" / "all_results.csv"
        reliability_input = revision_reliability if revision_reliability.exists() else legacy_reliability
    reliability = pd.read_csv(reliability_input)

    strict.to_csv(args.output_dir / "strict_ood_all_seeds.csv", index=False)
    strict_conf.to_csv(args.output_dir / "strict_ood_confidence_all_seeds.csv", index=False)
    transfer.to_csv(args.output_dir / "transfer_all_seeds.csv", index=False)
    external.to_csv(args.output_dir / "external_admet_all_seeds.csv", index=False)
    add_mean_std(strict[strict.get("status", "ok") == "ok"], ["split", "model"], args.output_dir / "strict_ood_mean_std.csv")
    add_mean_std(transfer, ["model"], args.output_dir / "transfer_mean_std.csv")
    add_mean_std(external, ["dataset", "model"], args.output_dir / "external_admet_mean_std.csv")
    add_mean_std(reliability, ["method"], args.output_dir / "reliability_mean_std.csv")

    rows = []
    for metric in METRICS:
        rows.append(
            paired_cluster_stats(
                strict[strict["status"] == "ok"],
                block="strict_ood",
                first="anchor_reasoning",
                second="rf_morgan",
                pair_cols=["dataset", "label", "split", "seed"],
                cluster_cols=["dataset", "label"],
                metric=metric,
                value_col=metric,
            )
        )
        for split in sorted(strict["split"].dropna().unique()):
            rows.append(
                paired_cluster_stats(
                    strict[(strict["status"] == "ok") & (strict["split"] == split)],
                    block=f"strict_ood_{split}",
                    first="anchor_reasoning",
                    second="rf_morgan",
                    pair_cols=["dataset", "label", "seed"],
                    cluster_cols=["dataset", "label"],
                    metric=metric,
                    value_col=metric,
                )
            )
        rows.append(
            paired_cluster_stats(
                transfer,
                block="transfer",
                first="anchor_reasoning",
                second="rf_morgan",
                pair_cols=["source_dataset", "target_dataset", "seed"],
                cluster_cols=["source_dataset", "target_dataset"],
                metric=metric,
                value_col=metric,
            )
        )
        rows.append(
            paired_cluster_stats(
                external,
                block="external_admet",
                first="anchor_reasoning",
                second="rf_morgan",
                pair_cols=["dataset", "seed"],
                cluster_cols=["dataset"],
                metric=metric,
                value_col=metric,
            )
        )
    for first in ["anchor_heuristic", "learned", "tree_conf"]:
        for metric in ["error_detection_auroc", "risk_coverage_auc"]:
            rows.append(
                paired_cluster_stats(
                    reliability,
                    block="reliability",
                    first=first,
                    second="margin",
                    pair_cols=["dataset", "label", "seed"],
                    cluster_cols=["dataset", "label"],
                    metric=metric,
                    value_col=metric,
                    model_col="method",
                )
            )
    stats = pd.DataFrame(rows)
    stats["wilcoxon_cluster_p_bh"] = benjamini_hochberg(stats["wilcoxon_cluster_p"].tolist())
    stats.to_csv(args.output_dir / "paired_cluster_bootstrap_stats.csv", index=False)
    summary = {
        "strict_rows": int(len(strict)),
        "strict_confidence_rows": int(len(strict_conf)),
        "transfer_rows": int(len(transfer)),
        "external_admet_rows": int(len(external)),
        "reliability_rows": int(len(reliability)),
        "statistical_comparisons": int(len(stats)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
