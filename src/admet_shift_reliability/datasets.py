from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def fetch_tdc_tox_frame(dataset_name: str) -> pd.DataFrame:
    from tdc.single_pred import Tox

    data = Tox(name=dataset_name)
    return data.get_data(format="df")


def _pick_existing_column(df: pd.DataFrame, candidates: list[str], field_name: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find {field_name} column in {list(df.columns)}")


def normalize_task_frame(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
) -> pd.DataFrame:
    out = df[[smiles_col, label_col]].copy()
    out = out.rename(columns={smiles_col: "smiles", label_col: "label"})
    return out


def load_task_frame(
    task_cfg: dict,
    fetch_tdc_tox_frame: Callable[[str], pd.DataFrame] = fetch_tdc_tox_frame,
) -> pd.DataFrame:
    source = task_cfg.get("source", "csv")

    if source == "csv":
        frame = pd.read_csv(task_cfg["path"])
        smiles_col = task_cfg["smiles_col"]
        label_col = task_cfg.get("raw_label_col", task_cfg["label"])
        return normalize_task_frame(frame, smiles_col=smiles_col, label_col=label_col)

    if source == "tdc_tox":
        cache_path = Path(task_cfg["cache_path"])
        if cache_path.exists():
            frame = pd.read_csv(cache_path)
        else:
            frame = fetch_tdc_tox_frame(task_cfg["tdc_name"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache_path, index=False)

        smiles_col = task_cfg.get("smiles_col") or _pick_existing_column(
            frame,
            candidates=["Drug", "SMILES", "smiles", "X"],
            field_name="smiles",
        )
        label_col = task_cfg.get("raw_label_col") or _pick_existing_column(
            frame,
            candidates=["Y", "y", "label"],
            field_name="label",
        )
        return normalize_task_frame(frame, smiles_col=smiles_col, label_col=label_col)

    raise ValueError(f"Unsupported task source: {source}")

