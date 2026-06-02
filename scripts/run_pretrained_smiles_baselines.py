from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_reliability_benchmark import (  # noqa: E402
    TASKS,
    evaluate_probs,
    filter_valid_smiles,
    load_task_frame,
    make_scaffold_split,
    split_is_usable,
)


DEFAULT_MODELS = ["DeepChem/ChemBERTa-77M-MTR"]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def embed_smiles(
    smiles: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded)
            hidden = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
    emb = np.concatenate(rows, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb


def fit_predict_models(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, rf_n_jobs: int) -> dict[str, np.ndarray]:
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed),
    )
    logreg.fit(x_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        n_jobs=rf_n_jobs,
        class_weight="balanced",
        random_state=seed,
    )
    rf.fit(x_train, y_train)
    return {
        "pretrained_logreg": logreg.predict_proba(x_test)[:, 1],
        "pretrained_rf": rf.predict_proba(x_test)[:, 1],
    }


def run_task(task_cfg: dict, model_name: str, seed: int, device: str, batch_size: int, max_length: int, rf_n_jobs: int, cache_dir: Path) -> list[dict]:
    df = load_task_frame(task_cfg).dropna().copy()
    df["label"] = df["label"].astype(int)
    df = filter_valid_smiles(df, "smiles")
    y = df["label"].to_numpy()
    split = make_scaffold_split(df["smiles"].tolist())
    usable, reason = split_is_usable(split, y)
    if not usable:
        return [{"dataset": task_cfg["dataset"], "label": task_cfg["label"], "encoder": model_name, "model": "skipped", "status": f"skipped_{reason}"}]
    cache_path = cache_dir / f"{safe_name(model_name)}__{task_cfg['dataset']}__{safe_name(task_cfg['label'])}.npy"
    emb = embed_smiles(df["smiles"].tolist(), model_name=model_name, device=device, batch_size=batch_size, max_length=max_length, cache_path=cache_path)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    probs_by_model = fit_predict_models(emb[train_idx], y[train_idx], emb[test_idx], seed=seed, rf_n_jobs=rf_n_jobs)
    rows = []
    for model, probs in probs_by_model.items():
        rows.append(
            {
                "dataset": task_cfg["dataset"],
                "label": task_cfg["label"],
                "split": "scaffold",
                "encoder": model_name,
                "model": model,
                "seed": seed,
                "status": "ok",
                "n_samples": int(len(df)),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "embedding_dim": int(emb.shape[1]),
                **evaluate_probs(y[test_idx], probs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rf-n-jobs", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pretrained_smiles_baselines")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    device = f"cuda:{args.gpu}"
    rows: list[dict] = []
    failures: list[dict] = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        for task in TASKS:
            print(f"RUN encoder={model_name} task={task['dataset']}::{task['label']}", flush=True)
            try:
                rows.extend(run_task(task, model_name, args.seed, device, args.batch_size, args.max_length, args.rf_n_jobs, cache_dir))
                pd.DataFrame(rows).to_csv(args.output_dir / "pretrained_smiles_baseline_metrics.csv", index=False)
            except Exception as exc:
                failures.append({"encoder": model_name, "dataset": task["dataset"], "label": task["label"], "error": repr(exc)})
                pd.DataFrame(failures).to_csv(args.output_dir / "pretrained_smiles_baseline_failures.csv", index=False)
                print(f"SKIP encoder={model_name} task={task['dataset']}::{task['label']}: {exc}", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "pretrained_smiles_baseline_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "pretrained_smiles_baseline_failures.csv", index=False)
    summary = {
        "rows": len(rows),
        "failures": len(failures),
        "encoders": [item.strip() for item in args.models.split(",") if item.strip()],
        "tasks": int(pd.DataFrame(rows)[["dataset", "label"]].drop_duplicates().shape[0]) if rows else 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
