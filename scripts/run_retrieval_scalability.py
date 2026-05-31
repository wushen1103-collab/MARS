from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from admet_shift_reliability.anchor_reliability import compute_anchor_features  # noqa: E402
from admet_shift_reliability.features import morgan_fingerprint_matrix  # noqa: E402
from run_reliability_benchmark import TASKS, filter_valid_smiles, load_task_frame  # noqa: E402


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except ImportError:
        return float("nan")


def load_fingerprint_pool() -> np.ndarray:
    frames = []
    for task in TASKS:
        df = load_task_frame(task).dropna().copy()
        df = filter_valid_smiles(df, "smiles")
        frames.append(df[["smiles"]])
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["smiles"])
    return morgan_fingerprint_matrix(pool["smiles"].tolist()).astype(bool)


def sample_rows(pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return pool[rng.choice(len(pool), size=size, replace=size > len(pool))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-sizes", default="1000,5000,10000,25000,50000,100000")
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "retrieval_scalability_20260531")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pool = load_fingerprint_pool()
    query_x = sample_rows(pool, args.query_size, rng)
    rows = []
    for bank_size in parse_ints(args.bank_sizes):
        for repeat in range(args.repeats):
            bank_x = sample_rows(pool, bank_size, rng)
            bank_y = rng.integers(0, 2, size=bank_size, dtype=np.int64)
            rss_before = current_rss_mb()
            started = time.perf_counter()
            compute_anchor_features(bank_x, bank_y, query_x, n_neighbors=args.neighbors)
            elapsed = time.perf_counter() - started
            rss_after = current_rss_mb()
            row = {
                "bank_size": bank_size,
                "query_size": args.query_size,
                "neighbors": args.neighbors,
                "repeat": repeat + 1,
                "elapsed_seconds": elapsed,
                "milliseconds_per_query": 1000.0 * elapsed / args.query_size,
                "queries_per_second": args.query_size / elapsed,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "retrieval_scalability_runs.csv", index=False)
    summary = (
        df.groupby(["bank_size", "query_size", "neighbors"], as_index=False)
        .agg(
            milliseconds_per_query_mean=("milliseconds_per_query", "mean"),
            milliseconds_per_query_sd=("milliseconds_per_query", "std"),
            queries_per_second_mean=("queries_per_second", "mean"),
            rss_after_mb_mean=("rss_after_mb", "mean"),
        )
        .sort_values("bank_size")
    )
    summary.to_csv(args.output_dir / "retrieval_scalability_summary.csv", index=False)
    metadata = {
        "benchmark": "exact brute-force Jaccard top-K retrieval",
        "fingerprint_bits": int(pool.shape[1]),
        "fingerprint_pool_size": int(pool.shape[0]),
        "query_size": args.query_size,
        "neighbors": args.neighbors,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
