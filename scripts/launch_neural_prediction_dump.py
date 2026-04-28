from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = "outputs/neural_prediction_dump_20260422"
PREDICTION_DIR = f"{BASE_OUTPUT}/predictions"
CALIBRATION_OUTPUT = "outputs/neural_calibration_true_20260422"
LOG_DIR = "logs/neural_prediction_dump_20260422"

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


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    label: str
    split: str
    seed: int
    output_dir: str
    prediction_path: str
    command: str
    log_name: str


def safe_label(label: str) -> str:
    return label.replace("/", "_")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str, allowed: set[str] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if allowed is not None:
        invalid = sorted(set(items).difference(allowed))
        if invalid:
            raise ValueError(f"Invalid values {invalid}; allowed={sorted(allowed)}")
    return items


def build_job(
    *,
    model: str,
    dataset: str,
    label: str,
    split: str,
    seed: int,
    epochs: int,
    patience: int,
    mc_dropout_passes: int,
) -> Job:
    output_dir = f"{BASE_OUTPUT}/{model}_seed{seed}"
    prediction_path = f"{PREDICTION_DIR}/{model}__{dataset}__{safe_label(label)}__{split}__seed{seed}.predictions.csv"
    if model == "schnet":
        command = (
            "scripts/train_schnet_baseline.py "
            f"--dataset {dataset} --label {shlex.quote(label)} --split {split} --seed {seed} "
            f"--gpu 0 --epochs {epochs} --patience {patience} --batch-size 96 "
            "--num-workers 4 --cache-workers 16 "
            f"--output-dir {output_dir} --prediction-dir {PREDICTION_DIR}"
        )
    else:
        command = (
            "scripts/train_gin_baseline.py "
            f"--model {model} --dataset {dataset} --label {shlex.quote(label)} --split {split} --seed {seed} "
            f"--gpu 0 --epochs {epochs} --patience {patience} --batch-size 128 --num-workers 4 "
            f"--output-dir {output_dir} --prediction-dir {PREDICTION_DIR} "
            f"--mc-dropout-passes {mc_dropout_passes}"
        )
    return Job(
        model=model,
        dataset=dataset,
        label=label,
        split=split,
        seed=seed,
        output_dir=output_dir,
        prediction_path=prediction_path,
        command=command,
        log_name=f"{split}__seed{seed}__{model}__{dataset}__{safe_label(label)}.log",
    )


def build_jobs(
    *,
    models: list[str],
    splits: list[str],
    seeds: list[int],
    epochs: int,
    patience: int,
    mc_dropout_passes: int,
) -> list[Job]:
    jobs = []
    for seed in seeds:
        for model in models:
            for dataset, label in TASKS:
                for split in splits:
                    jobs.append(
                        build_job(
                            model=model,
                            dataset=dataset,
                            label=label,
                            split=split,
                            seed=seed,
                            epochs=epochs,
                            patience=patience,
                            mc_dropout_passes=mc_dropout_passes,
                        )
                    )
    return jobs


def render_worker_script(gpu: int, jobs: list[Job], python_bin: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"mkdir -p {shlex.quote(LOG_DIR)} {shlex.quote(PREDICTION_DIR)}",
        f"PYTHON_BIN={shlex.quote(str(python_bin))}",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}",
        "export PYTHONUNBUFFERED=1",
        f"echo START gpu={gpu} $(date '+%F %T %z')",
    ]
    for job in jobs:
        log_path = f"{LOG_DIR}/{job.log_name}"
        lines.extend(
            [
                f"echo JOB seed={job.seed} model={shlex.quote(job.model)} "
                f"task={shlex.quote(job.dataset + ':' + job.label)} split={shlex.quote(job.split)} $(date '+%F %T %z')",
                f"if [ -s {shlex.quote(job.prediction_path)} ]; then",
                f"  echo SKIP existing {shlex.quote(job.prediction_path)}",
                "else",
                f"  mkdir -p {shlex.quote(job.output_dir)} {shlex.quote(PREDICTION_DIR)}",
                f"  \"$PYTHON_BIN\" {job.command} > {shlex.quote(log_path)} 2>&1",
                "fi",
            ]
        )
    lines.append(f"echo DONE gpu={gpu} $(date '+%F %T %z')")
    return "\n".join(lines) + "\n"


def write_worker_scripts(jobs: list[Job], gpus: list[int], python_bin: Path) -> list[Path]:
    queue_dir = ROOT / LOG_DIR
    queue_dir.mkdir(parents=True, exist_ok=True)
    queues = {gpu: [] for gpu in gpus}
    for idx, job in enumerate(jobs):
        queues[gpus[idx % len(gpus)]].append(job)

    paths = []
    for gpu, gpu_jobs in queues.items():
        path = queue_dir / f"gpu{gpu}_queue.sh"
        path.write_text(render_worker_script(gpu, gpu_jobs, python_bin))
        path.chmod(0o755)
        paths.append(path)
    return paths


def start_workers(worker_scripts: list[Path]) -> list[tuple[Path, str]]:
    started = []
    for script in worker_scripts:
        launcher_log = script.with_suffix(".launcher.log")
        cmd = f"setsid bash {shlex.quote(str(script))} > {shlex.quote(str(launcher_log))} 2>&1 < /dev/null & echo $!"
        proc = subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)
        started.append((script, proc.stdout.strip()))
    return started


def start_aggregator(python_bin: Path, expected_files: int, poll_seconds: int) -> str:
    log_path = ROOT / LOG_DIR / "aggregate.watch.log"
    cmd = (
        f"setsid {shlex.quote(str(python_bin))} scripts/aggregate_neural_calibration.py "
        f"--prediction-dir {shlex.quote(PREDICTION_DIR)} --output-dir {shlex.quote(CALIBRATION_OUTPUT)} "
        f"--watch --expected-files {expected_files} --poll-seconds {poll_seconds} "
        f"> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!"
    )
    proc = subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--models", default="gin,gat,mpnn,schnet")
    parser.add_argument("--splits", default="scaffold")
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mc-dropout-passes", type=int, default=20)
    parser.add_argument("--aggregate-watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    gpus = parse_csv_ints(args.gpus)
    models = parse_csv_strings(args.models, allowed=set(CORE_MODELS))
    splits = parse_csv_strings(args.splits, allowed={"random", "scaffold"})
    jobs = build_jobs(
        models=models,
        splits=splits,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.patience,
        mc_dropout_passes=args.mc_dropout_passes,
    )
    worker_scripts = write_worker_scripts(jobs, gpus=gpus, python_bin=args.python_bin)

    print(f"planned_jobs={len(jobs)}")
    print(f"expected_prediction_files={len(jobs)}")
    print(f"gpus={','.join(str(gpu) for gpu in gpus)}")
    print(f"prediction_dir={PREDICTION_DIR}")
    for script in worker_scripts:
        print(f"worker_script={script}")

    if args.start:
        started = start_workers(worker_scripts)
        aggregate_pid = start_aggregator(
            python_bin=args.python_bin,
            expected_files=len(jobs),
            poll_seconds=args.poll_seconds,
        ) if args.aggregate_watch else ""
        pid_path = ROOT / LOG_DIR / "launcher_pids.txt"
        lines = [f"{pid}\t{script}" for script, pid in started]
        if aggregate_pid:
            lines.append(f"{aggregate_pid}\taggregate_neural_calibration.py")
        pid_path.write_text("\n".join(lines) + "\n")
        for script, pid in started:
            print(f"started pid={pid} script={script}")
        if aggregate_pid:
            print(f"started aggregate_pid={aggregate_pid}")
        print(f"pid_file={pid_path}")
    else:
        print("dry_run=true; pass --start to launch workers")


if __name__ == "__main__":
    main()
