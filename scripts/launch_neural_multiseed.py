from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = "outputs/neural_multiseed_20260421"
LOG_DIR = "logs/neural_multiseed_20260421"

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
    stage: str
    seed: int
    model: str
    dataset: str
    label: str
    split: str
    output_dir: str
    result_path: str
    command: str
    log_name: str


def safe_label(label: str) -> str:
    return label.replace("/", "_")


def build_core_job(model: str, dataset: str, label: str, split: str, seed: int) -> Job:
    output_dir = f"{BASE_OUTPUT}/{model}_seed{seed}"
    result_path = f"{output_dir}/{dataset}__{safe_label(label)}__{split}.result.json"
    if model == "schnet":
        command = (
            "scripts/train_schnet_baseline.py "
            f"--dataset {dataset} --label {shlex.quote(label)} --split {split} --seed {seed} "
            "--gpu 0 --epochs 80 --patience 15 --batch-size 96 --num-workers 4 --cache-workers 16 "
            f"--output-dir {output_dir}"
        )
    else:
        command = (
            "scripts/train_gin_baseline.py "
            f"--model {model} --dataset {dataset} --label {shlex.quote(label)} --split {split} --seed {seed} "
            "--gpu 0 --epochs 80 --patience 15 --batch-size 128 --num-workers 4 "
            f"--output-dir {output_dir}"
        )
    return Job(
        stage=f"core_{split}",
        seed=seed,
        model=model,
        dataset=dataset,
        label=label,
        split=split,
        output_dir=output_dir,
        result_path=result_path,
        command=command,
        log_name=f"core_{split}__seed{seed}__{model}__{dataset}__{safe_label(label)}.log",
    )


def build_anchor_job(dataset: str, label: str, seed: int) -> Job:
    output_dir = f"{BASE_OUTPUT}/gin_embedding_anchor_seed{seed}"
    result_path = f"{output_dir}/{dataset}__{safe_label(label)}__scaffold.result.json"
    command = (
        "scripts/run_gin_embedding_anchor_probe.py "
        f"--dataset {dataset} --label {shlex.quote(label)} --seed {seed} "
        "--gpu 0 --epochs 60 --patience 12 --batch-size 128 --num-workers 4 "
        f"--output-dir {output_dir}"
    )
    return Job(
        stage="anchor_scaffold",
        seed=seed,
        model="gin_embedding_anchor",
        dataset=dataset,
        label=label,
        split="scaffold",
        output_dir=output_dir,
        result_path=result_path,
        command=command,
        log_name=f"anchor_scaffold__seed{seed}__gin_embedding_anchor__{dataset}__{safe_label(label)}.log",
    )


def build_jobs(
    seeds: list[int],
    include_random: bool = True,
    include_anchor: bool = True,
) -> list[Job]:
    jobs: list[Job] = []
    for seed in seeds:
        for model in CORE_MODELS:
            for dataset, label in TASKS:
                jobs.append(build_core_job(model, dataset, label, "scaffold", seed))
    if include_anchor:
        for seed in seeds:
            for dataset, label in TASKS:
                jobs.append(build_anchor_job(dataset, label, seed))
    if include_random:
        for seed in seeds:
            for model in CORE_MODELS:
                for dataset, label in TASKS:
                    jobs.append(build_core_job(model, dataset, label, "random", seed))
    return jobs


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def render_worker_script(gpu: int, jobs: list[Job], python_bin: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"mkdir -p {shlex.quote(LOG_DIR)}",
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
                f"echo JOB {shlex.quote(job.stage)} seed={job.seed} model={shlex.quote(job.model)} "
                f"task={shlex.quote(job.dataset + ':' + job.label)} split={shlex.quote(job.split)} $(date '+%F %T %z')",
                f"if [ -s {shlex.quote(job.result_path)} ]; then",
                f"  echo SKIP existing {shlex.quote(job.result_path)}",
                "else",
                f"  mkdir -p {shlex.quote(job.output_dir)}",
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
        cmd = f"nohup bash {shlex.quote(str(script))} > {shlex.quote(str(launcher_log))} 2>&1 & echo $!"
        proc = subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)
        started.append((script, proc.stdout.strip()))
    return started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--gpus", default="2,3,5,6,7")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    gpus = parse_csv_ints(args.gpus)
    jobs = build_jobs(seeds=seeds, include_random=not args.no_random, include_anchor=not args.no_anchor)
    worker_scripts = write_worker_scripts(jobs, gpus=gpus, python_bin=args.python_bin)

    print(f"planned_jobs={len(jobs)}")
    print(f"gpus={','.join(str(gpu) for gpu in gpus)}")
    for script in worker_scripts:
        print(f"worker_script={script}")

    if args.start:
        started = start_workers(worker_scripts)
        pid_path = ROOT / LOG_DIR / "launcher_pids.txt"
        pid_path.write_text("\n".join(f"{pid}\t{script}" for script, pid in started) + "\n")
        for script, pid in started:
            print(f"started pid={pid} script={script}")
        print(f"pid_file={pid_path}")
    else:
        print("dry_run=true; pass --start to launch workers")


if __name__ == "__main__":
    main()
