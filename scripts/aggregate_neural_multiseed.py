from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_OUTPUT = ROOT / "outputs" / "neural_multiseed_20260421"
DEFAULT_AGGREGATE_DIR = ROOT / "outputs" / "neural_multiseed_20260421_aggregate"

TASKS = [
    ("bbbp", "p_np"),
    ("clintox", "CT_TOX"),
    ("tox21", "NR-AhR"),
    ("tox21", "SR-MMP"),
    ("ames", "AMES"),
    ("herg", "hERG"),
    ("dili", "DILI"),
]
CORE_MODELS = ["gin", "gat", "mpnn", "schnet"]
CORE_SPLITS = ["scaffold", "random"]
METRICS = ["auroc", "auprc", "brier", "ece"]


def safe_label(label: str) -> str:
    return label.replace("/", "_")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_block(payload: dict, prefix: str | None = None) -> dict:
    row = {}
    for metric in METRICS:
        key = f"{prefix}_{metric}" if prefix else metric
        row[metric] = payload.get(key)
    return row


def collect_rows(
    base_output: Path,
    seeds: list[int],
    tasks: list[tuple[str, str]],
    core_models: list[str],
    core_splits: list[str],
    include_anchor: bool = True,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    missing: list[dict] = []

    for seed in seeds:
        for model in core_models:
            for dataset, label in tasks:
                for split in core_splits:
                    result_path = base_output / f"{model}_seed{seed}" / f"{dataset}__{safe_label(label)}__{split}.result.json"
                    if not (result_path.exists() and result_path.stat().st_size > 0):
                        missing.append(
                            {
                                "stage": "core",
                                "model_variant": model,
                                "seed": seed,
                                "dataset": dataset,
                                "label": label,
                                "split": split,
                                "path": str(result_path),
                            }
                        )
                        continue
                    payload = load_json(result_path)
                    rows.append(
                        {
                            "stage": "core",
                            "model_variant": model,
                            "seed": int(payload.get("seed", seed)),
                            "dataset": payload.get("dataset", dataset),
                            "label": payload.get("label", label),
                            "split": payload.get("split", split),
                            "path": str(result_path),
                            **metric_block(payload),
                        }
                    )

    if include_anchor:
        for seed in seeds:
            for dataset, label in tasks:
                result_path = base_output / f"gin_embedding_anchor_seed{seed}" / f"{dataset}__{safe_label(label)}__scaffold.result.json"
                if not (result_path.exists() and result_path.stat().st_size > 0):
                    missing.append(
                        {
                            "stage": "anchor",
                            "model_variant": "gin_embedding_anchor",
                            "seed": seed,
                            "dataset": dataset,
                            "label": label,
                            "split": "scaffold",
                            "path": str(result_path),
                        }
                    )
                    continue
                payload = load_json(result_path)
                for prefix in ("gin", "embed_anchor", "meta"):
                    rows.append(
                        {
                            "stage": "anchor",
                            "model_variant": f"gin_embedding_anchor_{prefix}",
                            "seed": seed,
                            "dataset": payload.get("dataset", dataset),
                            "label": payload.get("label", label),
                            "split": payload.get("split", "scaffold"),
                            "path": str(result_path),
                            **metric_block(payload, prefix=prefix),
                        }
                    )

    return rows, missing


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["stage", "model_variant", "dataset", "label", "split"], dropna=False)
    agg = grouped[METRICS].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns.to_flat_index()
    ]
    agg = agg.rename(columns={f"{METRICS[0]}_count": "n"})
    for metric in METRICS[1:]:
        count_col = f"{metric}_count"
        if count_col in agg.columns:
            agg = agg.drop(columns=[count_col])
    return agg


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_outputs(rows: list[dict], missing: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing)
    aggregate_df = aggregate_rows(rows)
    rows_df.to_csv(output_dir / "all_results.csv", index=False)
    missing_df.to_csv(output_dir / "missing.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_mean_std.csv", index=False)
    summary = {
        "num_rows": int(len(rows_df)),
        "num_missing_files": int(len(missing_df)),
        "num_aggregate_rows": int(len(aggregate_df)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--expected-files", type=int, default=189)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    while True:
        rows, missing = collect_rows(
            base_output=args.base_output,
            seeds=seeds,
            tasks=TASKS,
            core_models=CORE_MODELS,
            core_splits=CORE_SPLITS,
            include_anchor=True,
        )
        result_files = len(list(args.base_output.glob("**/*.result.json"))) if args.base_output.exists() else 0
        if not args.wait or (result_files >= args.expected_files and not missing):
            write_outputs(rows, missing, args.output_dir)
            print(
                json.dumps(
                    {
                        "rows": len(rows),
                        "missing": len(missing),
                        "result_files": result_files,
                        "output_dir": str(args.output_dir),
                    },
                    indent=2,
                )
            )
            return
        print(
            f"WAIT result_files={result_files}/{args.expected_files} missing={len(missing)} "
            f"next_poll_seconds={args.poll_seconds}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
