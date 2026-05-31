from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIGHER_IS_BETTER = ["auroc", "auprc"]
LOWER_IS_BETTER = ["brier", "ece"]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "revision_20260531" / "anchor_stratified_v2",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.input_dir / "anchor_stratified_metrics.csv")
    wide = metrics.pivot_table(
        index=["dataset", "label", "seed", "dimension", "stratum", "n", "positive_rate"],
        columns="model",
        values=HIGHER_IS_BETTER + LOWER_IS_BETTER,
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in HIGHER_IS_BETTER:
        wide[f"{metric}_benefit_anchor_over_rf"] = wide[f"{metric}_anchor_reasoning"] - wide[f"{metric}_rf_morgan"]
    for metric in LOWER_IS_BETTER:
        wide[f"{metric}_benefit_anchor_over_rf"] = wide[f"{metric}_rf_morgan"] - wide[f"{metric}_anchor_reasoning"]
    wide.to_csv(output_dir / "anchor_stratified_benefits.csv", index=False)

    benefit_cols = [f"{metric}_benefit_anchor_over_rf" for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER]
    summary_rows = []
    for (dimension, stratum), group in wide.groupby(["dimension", "stratum"], dropna=False):
        row = {
            "dimension": dimension,
            "stratum": stratum,
            "n_groups": int(len(group)),
            "total_samples": int(group["n"].sum()),
            "mean_group_size": float(group["n"].mean()),
        }
        for column in benefit_cols:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_weighted_mean"] = weighted_mean(group[column], group["n"])
            row[f"{column}_positive_fraction"] = float((group[column] > 0).mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "anchor_stratified_benefit_summary.csv", index=False)
    payload = {
        "benefit_rows": int(len(wide)),
        "summary_rows": int(len(summary)),
        "benefit_definition": "positive values indicate that anchor reasoning improves over RF Morgan",
    }
    (output_dir / "benefit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
