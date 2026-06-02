from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "error_detection_auroc",
    "risk_coverage_auc",
    "selective_error_cov02",
    "selective_error_cov05",
    "selective_error_cov08",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "revision" / "reliability_benchmark_shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "revision" / "reliability_benchmark_aggregate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("seed*/results_seed*.csv"))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No reliability benchmark shards found under {args.input_dir}")
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(args.output_dir / "all_results.csv", index=False)

    numeric = [metric for metric in METRICS if metric in results.columns]
    summary = results.groupby("method", dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns.to_flat_index()]
    summary.to_csv(args.output_dir / "aggregate_mean_std.csv", index=False)

    payload = {
        "input_shards": len(paths),
        "rows": int(len(results)),
        "seeds": sorted(results["seed"].dropna().astype(int).unique().tolist()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
