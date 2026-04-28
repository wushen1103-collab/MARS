from pathlib import Path

import pandas as pd

from admet_shift_reliability.datasets import load_task_frame


def test_load_task_frame_from_csv_normalizes_columns(tmp_path: Path):
    csv_path = tmp_path / "bbbp.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "p_np": [1, 0],
        }
    ).to_csv(csv_path, index=False)

    task_cfg = {
        "source": "csv",
        "path": csv_path,
        "smiles_col": "smiles",
        "label": "p_np",
    }

    df = load_task_frame(task_cfg)

    assert list(df.columns) == ["smiles", "label"]
    assert df["smiles"].tolist() == ["CCO", "CCN"]
    assert df["label"].tolist() == [1, 0]


def test_load_task_frame_downloads_and_caches_tdc_tox_dataset(tmp_path: Path):
    cache_path = tmp_path / "ames.csv"
    calls = {"count": 0}

    def fake_fetch(dataset_name: str) -> pd.DataFrame:
        calls["count"] += 1
        assert dataset_name == "AMES"
        return pd.DataFrame(
            {
                "Drug": ["CCO", "CCN"],
                "Y": [1, 0],
            }
        )

    task_cfg = {
        "source": "tdc_tox",
        "tdc_name": "AMES",
        "cache_path": cache_path,
    }

    first = load_task_frame(task_cfg, fetch_tdc_tox_frame=fake_fetch)
    second = load_task_frame(task_cfg, fetch_tdc_tox_frame=lambda _: (_ for _ in ()).throw(RuntimeError("cache not used")))

    assert calls["count"] == 1
    assert cache_path.exists()
    assert list(first.columns) == ["smiles", "label"]
    assert first.equals(second)

