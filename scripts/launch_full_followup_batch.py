from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "full_followup"
TASKS = [
    ("bbbp", "p_np"),
    ("clintox", "CT_TOX"),
    ("tox21", "NR-AhR"),
    ("tox21", "SR-MMP"),
    ("ames", "AMES"),
    ("herg", "hERG"),
    ("dili", "DILI"),
]


@dataclass(frozen=True)
class Job:
    name: str
    command: str
    expected_file: str
    log_name: str


def safe_label(label: str) -> str:
    return label.replace("/", "_")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def chemprop_jobs() -> list[Job]:
    jobs: list[Job] = []
    for dataset, label in TASKS:
        for tag, ensemble_size, loss_function, epochs in [
            ("ens5", 5, None, 30),
            ("dirichlet", 1, "dirichlet", 30),
            ("random_single", 1, None, 30),
        ]:
            split = "random" if tag == "random_single" else "scaffold"
            run_name = f"{dataset}__{label}__{split}__{tag}"
            expected = f"outputs/chemprop_baseline/{run_name}/test_preds.csv"
            cmd = (
                "scripts/run_chemprop_baseline.py "
                f"--dataset {shlex.quote(dataset)} --label {shlex.quote(label)} --split {split} "
                "--seed 42 --gpu 0 "
                f"--epochs {epochs} --batch-size 64 --num-workers 8 --ensemble-size {ensemble_size} "
                f"--save-tag {shlex.quote(tag)}"
            )
            if loss_function:
                cmd += f" --loss-function {shlex.quote(loss_function)}"
            jobs.append(
                Job(
                    name=f"chemprop_{tag}_{dataset}_{safe_label(label)}",
                    command=cmd,
                    expected_file=expected,
                    log_name=f"chemprop__{tag}__{dataset}__{safe_label(label)}.log",
                )
            )
    return jobs


def render_gpu_worker(gpu: int, jobs: list[Job], python_bin: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"mkdir -p {shlex.quote(str(LOG_DIR))}",
        f"PYTHON_BIN={shlex.quote(str(python_bin))}",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}",
        "export PYTHONUNBUFFERED=1",
        f"echo START gpu={gpu} $(date '+%F %T %z')",
    ]
    for job in jobs:
        log_path = LOG_DIR / job.log_name
        lines.extend(
            [
                f"echo JOB {shlex.quote(job.name)} $(date '+%F %T %z')",
                f"if [ -s {shlex.quote(job.expected_file)} ]; then",
                f"  echo SKIP existing {shlex.quote(job.expected_file)}",
                "else",
                f"  \"$PYTHON_BIN\" {job.command} > {shlex.quote(str(log_path))} 2>&1",
                "  status=$?",
                "  if [ $status -ne 0 ]; then",
                f"    echo FAIL {shlex.quote(job.name)} status=$status log={shlex.quote(str(log_path))}",
                "  fi",
                "fi",
            ]
        )
    lines.append(f"echo DONE gpu={gpu} $(date '+%F %T %z')")
    return "\n".join(lines) + "\n"


def render_cpu_worker(python_bin: Path, gpu_pid_file: Path, cpu_workers: int) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            f"cd {shlex.quote(str(ROOT))}",
            f"mkdir -p {shlex.quote(str(LOG_DIR))}",
            f"PYTHON_BIN={shlex.quote(str(python_bin))}",
            "export CUDA_VISIBLE_DEVICES=",
            "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}",
            "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}",
            "export PYTHONUNBUFFERED=1",
            "echo CPU_FOLLOWUPS_START $(date '+%F %T %z')",
            f"\"$PYTHON_BIN\" scripts/run_leakage_shift_uq_bootstrap.py --rf-n-jobs {cpu_workers} --n-boot 1000 > {shlex.quote(str(LOG_DIR / 'leakage_shift_uq_bootstrap.log'))} 2>&1 &",
            "p1=$!",
            f"\"$PYTHON_BIN\" scripts/run_activity_cliff_targeted.py --n-jobs {cpu_workers} > {shlex.quote(str(LOG_DIR / 'activity_cliff_targeted.log'))} 2>&1 &",
            "p2=$!",
            f"\"$PYTHON_BIN\" scripts/run_conformer_sensitivity.py --n-jobs 24 --conformer-workers {cpu_workers} --num-conformers 1,3,5 --output-dir outputs/conformer_sensitivity_full > {shlex.quote(str(LOG_DIR / 'conformer_sensitivity_full.log'))} 2>&1 &",
            "p3=$!",
            f"\"$PYTHON_BIN\" scripts/run_external_admet_probe.py --rf-n-jobs {cpu_workers} > {shlex.quote(str(LOG_DIR / 'external_admet_probe.log'))} 2>&1 &",
            "p4=$!",
            "wait $p1; echo DONE leakage_shift_uq status=$? $(date '+%F %T %z')",
            "wait $p2; echo DONE activity_cliff_targeted status=$? $(date '+%F %T %z')",
            "wait $p3; echo DONE conformer_sensitivity_full status=$? $(date '+%F %T %z')",
            "wait $p4; echo DONE external_admet_probe status=$? $(date '+%F %T %z')",
            "echo WAIT_GPU_CHEMPROP $(date '+%F %T %z')",
            f"while read -r pid script; do if [ -n \"$pid\" ]; then wait \"$pid\" 2>/dev/null || true; fi; done < {shlex.quote(str(gpu_pid_file))}",
            "export CUDA_VISIBLE_DEVICES=0",
            f"\"$PYTHON_BIN\" scripts/aggregate_chemprop_metrics.py --generate-valid-preds > {shlex.quote(str(LOG_DIR / 'aggregate_chemprop_metrics.log'))} 2>&1",
            f"\"$PYTHON_BIN\" scripts/build_internal_sota_claim_audit.py --output-dir outputs/paper_claim_audit_with_chemprop > {shlex.quote(str(LOG_DIR / 'paper_claim_audit_with_chemprop.log'))} 2>&1",
            "echo CPU_FOLLOWUPS_DONE $(date '+%F %T %z')",
            "",
        ]
    )


def write_gpu_workers(jobs: list[Job], gpus: list[int], python_bin: Path) -> list[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queues = {gpu: [] for gpu in gpus}
    for idx, job in enumerate(jobs):
        queues[gpus[idx % len(gpus)]].append(job)
    scripts = []
    for gpu, gpu_jobs in queues.items():
        path = LOG_DIR / f"gpu{gpu}_chemprop_queue.sh"
        path.write_text(render_gpu_worker(gpu, gpu_jobs, python_bin))
        path.chmod(0o755)
        scripts.append(path)
    return scripts


def start_script(script: Path) -> str:
    launcher_log = script.with_suffix(".launcher.log")
    cmd = f"setsid bash {shlex.quote(str(script))} > {shlex.quote(str(launcher_log))} 2>&1 < /dev/null & echo $!"
    proc = subprocess.run(["bash", "-lc", cmd], check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,7")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-workers", type=int, default=48)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    gpus = parse_csv_ints(args.gpus)
    jobs = chemprop_jobs()
    gpu_scripts = write_gpu_workers(jobs, gpus=gpus, python_bin=args.python_bin)
    manifest = {
        "chemprop_jobs": [job.__dict__ for job in jobs],
        "gpus": gpus,
        "cpu_workers": args.cpu_workers,
        "gpu_scripts": [str(path) for path in gpu_scripts],
    }
    (LOG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"planned_chemprop_jobs": len(jobs), "gpus": gpus, "log_dir": str(LOG_DIR)}, indent=2))

    if not args.start:
        print("dry_run=true; pass --start to launch")
        return

    gpu_pid_file = LOG_DIR / "gpu_worker_pids.txt"
    gpu_pids = [(start_script(script), script) for script in gpu_scripts]
    gpu_pid_file.write_text("\n".join(f"{pid} {script}" for pid, script in gpu_pids) + "\n")

    cpu_script = LOG_DIR / "cpu_followups_and_postprocess.sh"
    cpu_script.write_text(render_cpu_worker(args.python_bin, gpu_pid_file, args.cpu_workers))
    cpu_script.chmod(0o755)
    cpu_pid = start_script(cpu_script)
    (LOG_DIR / "all_launcher_pids.txt").write_text(
        "\n".join([f"{pid} {script}" for pid, script in gpu_pids] + [f"{cpu_pid} {cpu_script}"]) + "\n"
    )
    for pid, script in gpu_pids:
        print(f"started gpu_worker pid={pid} script={script}")
    print(f"started cpu_followups pid={cpu_pid} script={cpu_script}")
    print(f"pid_file={LOG_DIR / 'all_launcher_pids.txt'}")


if __name__ == "__main__":
    main()

