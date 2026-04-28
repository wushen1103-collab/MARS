from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


VARIANT_MAP = {
    "schnet": "Ours-Base",
    "gin_embedding_anchor_meta": "Ours+Anchor",
    "learned": "Ours+Reliability",
    "anchor_heuristic": "Ours-Full",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-aggregate", type=Path, default=ROOT / "outputs" / "neural_multiseed_20260421_aggregate_seed1_4" / "aggregate_mean_std.csv")
    parser.add_argument("--reliability-aggregate", type=Path, default=ROOT / "outputs" / "reliability_benchmark_expanded_multiseed" / "aggregate_mean_std.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ours_variant_tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    neural = pd.read_csv(args.neural_aggregate)
    rows = []
    for source_variant, ours_name in VARIANT_MAP.items():
        if source_variant in set(neural["model_variant"]):
            sub = neural[(neural["model_variant"] == source_variant) & (neural["split"] == "scaffold")].copy()
            sub["ours_variant"] = ours_name
            rows.append(sub)
    if args.reliability_aggregate.exists():
        rel = pd.read_csv(args.reliability_aggregate)
        for source_variant, ours_name in [("learned", "Ours+Reliability"), ("anchor_heuristic", "Ours-Full")]:
            sub = rel[rel["method"] == source_variant].copy()
            if not sub.empty:
                sub["ours_variant"] = ours_name
                sub["model_variant"] = source_variant
                sub["split"] = "scaffold"
                sub = sub.rename(columns={"base_auroc_mean": "auroc_mean", "base_auprc_mean": "auprc_mean", "base_brier_mean": "brier_mean", "base_ece_mean": "ece_mean"})
                rows.append(sub)
    table = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    table.to_csv(args.output_dir / "ours_variant_main_table.csv", index=False)
    pivot = table.pivot_table(index=["dataset", "label"], columns="ours_variant", values="auroc_mean", aggfunc="first").reset_index() if not table.empty else pd.DataFrame()
    pivot.to_csv(args.output_dir / "ours_variant_auroc_pivot.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": len(table), "variants": sorted(table["ours_variant"].dropna().unique().tolist()) if not table.empty else []}, indent=2))


if __name__ == "__main__":
    main()
