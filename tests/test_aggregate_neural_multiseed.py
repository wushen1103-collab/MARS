import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_neural_multiseed.py"


def load_aggregator_module():
    spec = importlib.util.spec_from_file_location("aggregate_neural_multiseed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_collects_core_and_anchor_variant_rows_and_aggregates(tmp_path):
    mod = load_aggregator_module()
    base = tmp_path / "outputs" / "neural_multiseed_20260421"

    write_json(
        base / "gin_seed1" / "bbbp__p_np__scaffold.result.json",
        {
            "dataset": "bbbp",
            "model": "gin",
            "label": "p_np",
            "split": "scaffold",
            "seed": 1,
            "auroc": 0.7,
            "auprc": 0.8,
            "brier": 0.2,
            "ece": 0.1,
        },
    )
    write_json(
        base / "gin_embedding_anchor_seed1" / "bbbp__p_np__scaffold.result.json",
        {
            "dataset": "bbbp",
            "label": "p_np",
            "split": "scaffold",
            "gin_auroc": 0.71,
            "gin_auprc": 0.81,
            "gin_brier": 0.19,
            "gin_ece": 0.09,
            "embed_anchor_auroc": 0.72,
            "embed_anchor_auprc": 0.82,
            "embed_anchor_brier": 0.18,
            "embed_anchor_ece": 0.08,
            "meta_auroc": 0.73,
            "meta_auprc": 0.83,
            "meta_brier": 0.17,
            "meta_ece": 0.07,
        },
    )

    rows, missing = mod.collect_rows(
        base_output=base,
        seeds=[1],
        tasks=[("bbbp", "p_np")],
        core_models=["gin"],
        core_splits=["scaffold"],
        include_anchor=True,
    )

    assert missing == []
    assert {row["model_variant"] for row in rows} == {
        "gin",
        "gin_embedding_anchor_gin",
        "gin_embedding_anchor_embed_anchor",
        "gin_embedding_anchor_meta",
    }

    aggregate = mod.aggregate_rows(rows)
    assert set(aggregate["model_variant"]) == {row["model_variant"] for row in rows}
    assert "auroc_mean" in aggregate.columns
    assert set(aggregate["n"]) == {1}
