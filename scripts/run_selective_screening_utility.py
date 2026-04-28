from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def enrichment_at_fraction(y_true: np.ndarray, score: np.ndarray, fraction: float) -> tuple[float, int]:
    keep = max(1, int(round(len(y_true) * fraction)))
    order = np.argsort(-score, kind="mergesort")[:keep]
    baseline = float(np.mean(y_true))
    precision = float(np.mean(y_true[order]))
    return precision / max(baseline, 1e-12), keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "reliability_benchmark_expanded_multiseed" / "aggregate_mean_std.csv")
    parser.add_argument("--anchor-input", type=Path, default=ROOT / "outputs" / "anchor_reliability_probe" / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "selective_screening_utility")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rel = pd.read_csv(args.input)
    selective_cols = [c for c in rel.columns if c.startswith("selective_error_cov") and c.endswith("_mean")]
    rows = rel[["dataset", "label", "method", "base_auroc_mean", "base_auprc_mean", "error_detection_auroc_mean", "risk_coverage_auc_mean", *selective_cols]].copy()
    rows.to_csv(args.output_dir / "selective_coverage_metrics.csv", index=False)

    anchor = pd.read_csv(args.anchor_input)
    ef_rows = []
    for _, row in anchor.iterrows():
        for prefix in ["rf", "anchor", "meta"]:
            # We do not have per-sample probabilities here; use high-level AUPRC as a screening utility proxy.
            ef_rows.append(
                {
                    "dataset": row["dataset"],
                    "label": row["label"],
                    "model": prefix,
                    "screening_proxy": f"{prefix}_auprc",
                    "auprc": row[f"{prefix}_auprc"],
                    "brier": row[f"{prefix}_brier"],
                    "ece": row[f"{prefix}_ece"],
                }
            )
    pd.DataFrame(ef_rows).to_csv(args.output_dir / "enrichment_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"coverage_rows": len(rows), "enrichment_rows": len(ef_rows)}, indent=2))


if __name__ == "__main__":
    main()
