from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_script_module(script_name: str, module_name: str):
    spec = spec_from_file_location(module_name, SCRIPTS / script_name)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gin_embedding_probe_evaluate_probs_clips_numerical_overflow():
    module = _load_script_module("run_gin_embedding_anchor_probe.py", "gin_embed_probe_test")

    metrics = module.evaluate_probs(
        y_true=np.array([0, 1], dtype=np.int64),
        probs=np.array([0.0, 1.0000001], dtype=np.float32),
    )

    assert 0.0 <= metrics["brier"] <= 1.0


def test_anchor_reliability_probe_load_probe_frame_supports_tdc_tasks(monkeypatch: pytest.MonkeyPatch):
    module = _load_script_module("run_anchor_reliability_probe.py", "anchor_reliability_probe_test")

    task_cfg = {
        "dataset": "ames",
        "source": "tdc_tox",
        "tdc_name": "AMES",
        "cache_path": ROOT / "data" / "raw" / "AMES_tdc.csv.gz",
        "label": "AMES",
    }

    monkeypatch.setattr(
        module,
        "load_task_frame",
        lambda cfg: pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [1, 0]}),
        raising=False,
    )

    df = module.load_probe_frame(task_cfg)

    assert list(df.columns) == ["smiles", "label"]
    assert df["label"].tolist() == [1, 0]

