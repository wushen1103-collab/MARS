from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TASKS = [
    ("bbbp", "p_np"),
    ("clintox", "CT_TOX"),
    ("tox21", "NR-AhR"),
    ("tox21", "SR-MMP"),
    ("ames", "AMES"),
    ("herg", "hERG"),
    ("dili", "DILI"),
]
STRICT_SPLITS = ["fingerprint_density", "molecular_weight_reverse", "pca_cluster"]


def shell_quote(path: Path | str) -> str:
    text = str(path)
    return "'" + text.replace("'", "'\\''") + "'"


def distribute(items: list[tuple[str, str, str]], buckets: int) -> list[list[tuple[str, str, str]]]:
    out: list[list[tuple[str, str, str]]] = [[] for _ in range(max(1, buckets))]
    for idx, item in enumerate(items):
        out[idx % len(out)].append(item)
    return out


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | 0o111)


def build_run_script(gpus: list[int], log_dir: Path, rf_n_jobs: int, chemprop_epochs: int) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_for_pretrained = gpus[0]
    chemprop_gpus = gpus[1:] if len(gpus) > 1 else gpus
    chemprop_jobs = [(dataset, label, split) for dataset, label in TASKS for split in STRICT_SPLITS]
    chemprop_groups = distribute(chemprop_jobs, len(chemprop_gpus))

    lines = [
        "#!/usr/bin/env bash",
        "set +e",
        f"cd {shell_quote(ROOT)}",
        "export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}",
        "export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}",
        f"PY={shell_quote(PYTHON)}",
        f"LOG_DIR={shell_quote(log_dir)}",
        "mkdir -p \"$LOG_DIR\"",
        "echo \"TOPJOURNAL_BATCH_START $(date -Is)\" | tee \"$LOG_DIR/manifest.log\"",
        "",
        "$PY scripts/run_strict_ood_model_matrix.py --seeds 42,43,44 --rf-n-jobs "
        f"{rf_n_jobs} --ensemble-size 5 --output-dir outputs/strict_ood_model_matrix "
        "> \"$LOG_DIR/strict_ood_model_matrix.log\" 2>&1 &",
        "PID_STRICT=$!",
        "$PY scripts/run_conformal_risk_control.py --seed 42 --rf-n-jobs "
        f"{rf_n_jobs} --output-dir outputs/conformal_risk_control "
        "> \"$LOG_DIR/conformal_risk_control.log\" 2>&1 &",
        "PID_CONFORMAL=$!",
        "$PY scripts/run_cross_dataset_transfer.py --seed 42 --rf-n-jobs "
        f"{rf_n_jobs} --output-dir outputs/cross_dataset_transfer "
        "> \"$LOG_DIR/cross_dataset_transfer.log\" 2>&1 &",
        "PID_CROSS=$!",
        "$PY scripts/run_anchor_case_studies.py --seed 42 --rf-n-jobs "
        f"{rf_n_jobs} --output-dir outputs/anchor_case_studies "
        "> \"$LOG_DIR/anchor_case_studies.log\" 2>&1 &",
        "PID_CASES=$!",
        f"CUDA_VISIBLE_DEVICES={gpu_for_pretrained} $PY scripts/run_pretrained_smiles_baselines.py --gpu 0 --batch-size 128 --rf-n-jobs 24 "
        "--output-dir outputs/pretrained_smiles_baselines "
        "> \"$LOG_DIR/pretrained_smiles_baselines.log\" 2>&1 &",
        "PID_PRETRAIN=$!",
        "echo strict_ood_model_matrix=$PID_STRICT >> \"$LOG_DIR/manifest.log\"",
        "echo conformal_risk_control=$PID_CONFORMAL >> \"$LOG_DIR/manifest.log\"",
        "echo cross_dataset_transfer=$PID_CROSS >> \"$LOG_DIR/manifest.log\"",
        "echo anchor_case_studies=$PID_CASES >> \"$LOG_DIR/manifest.log\"",
        "echo pretrained_smiles_baselines=$PID_PRETRAIN >> \"$LOG_DIR/manifest.log\"",
    ]

    for group_idx, (gpu, jobs) in enumerate(zip(chemprop_gpus, chemprop_groups)):
        group_script = log_dir / f"chemprop_strict_gpu{gpu}.sh"
        group_lines = [
            "#!/usr/bin/env bash",
            "set +e",
            f"cd {shell_quote(ROOT)}",
            f"PY={shell_quote(PYTHON)}",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
            "export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}",
        ]
        for dataset, label, split in jobs:
            log_name = f"chemprop_strict__{dataset}__{label.replace('/', '_')}__{split}.log"
            group_lines.append(
                "$PY scripts/run_chemprop_strict_ood_baseline.py "
                f"--dataset {dataset} --label {label} --strict-split {split} --seed 42 --gpu 0 "
                f"--epochs {chemprop_epochs} --batch-size 64 --num-workers 8 --ensemble-size 1 "
                f"> \"$LOG_DIR/{log_name}\" 2>&1"
            )
        write_executable(group_script, "\n".join(group_lines) + "\n")
        lines.append(f"LOG_DIR={shell_quote(log_dir)} bash {shell_quote(group_script)} > \"$LOG_DIR/chemprop_strict_group{group_idx}_gpu{gpu}.log\" 2>&1 &")
        lines.append(f"PID_CHEMPROP_{group_idx}=$!")
        lines.append(f"echo chemprop_strict_group{group_idx}_gpu{gpu}=$PID_CHEMPROP_{group_idx} >> \"$LOG_DIR/manifest.log\"")

    wait_pids = ["$PID_STRICT", "$PID_CONFORMAL", "$PID_CROSS", "$PID_CASES", "$PID_PRETRAIN"] + [f"$PID_CHEMPROP_{idx}" for idx in range(len(chemprop_groups))]
    lines.extend(
        [
            "",
            "echo \"TOPJOURNAL_BATCH_WAIT $(date -Is)\" >> \"$LOG_DIR/manifest.log\"",
            "wait " + " ".join(wait_pids),
            "echo \"TOPJOURNAL_BATCH_AGGREGATE $(date -Is)\" >> \"$LOG_DIR/manifest.log\"",
            "$PY scripts/aggregate_chemprop_metrics.py --chemprop-root outputs/chemprop_strict_ood "
            "--output-dir outputs/chemprop_strict_ood_metrics --generate-valid-preds "
            "> \"$LOG_DIR/aggregate_chemprop_strict_ood.log\" 2>&1",
            "echo \"TOPJOURNAL_BATCH_DONE $(date -Is)\" >> \"$LOG_DIR/manifest.log\"",
        ]
    )
    run_script = log_dir / "run_all_topjournal_followup.sh"
    write_executable(run_script, "\n".join(lines) + "\n")
    return run_script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,7")
    parser.add_argument("--rf-n-jobs", type=int, default=32)
    parser.add_argument("--chemprop-epochs", type=int, default=20)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "topjournal_followup")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    run_script = build_run_script(gpus, args.log_dir, rf_n_jobs=args.rf_n_jobs, chemprop_epochs=args.chemprop_epochs)
    payload = {
        "run_script": str(run_script),
        "log_dir": str(args.log_dir),
        "gpus": gpus,
        "rf_n_jobs": args.rf_n_jobs,
        "chemprop_epochs": args.chemprop_epochs,
        "strict_chemprop_jobs": len(TASKS) * len(STRICT_SPLITS),
    }
    if args.start:
        log_path = args.log_dir / "run_all_topjournal_followup.nohup.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(["bash", str(run_script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        payload["pid"] = proc.pid
        payload["nohup_log"] = str(log_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
