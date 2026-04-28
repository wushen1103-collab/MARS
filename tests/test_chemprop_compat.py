import pandas.io.formats.format as fmt
import torch

from admet_shift_reliability.chemprop_compat import (
    patch_pandas_rdkit_compat,
    patch_torch_load_weights_only_false,
)


def test_patch_pandas_rdkit_compat_adds_get_adjustment():
    original = getattr(fmt, "get_adjustment", None)
    if hasattr(fmt, "get_adjustment"):
        delattr(fmt, "get_adjustment")

    try:
        patch_pandas_rdkit_compat()
        assert hasattr(fmt, "get_adjustment")
    finally:
        if original is not None:
            fmt.get_adjustment = original


def test_patch_torch_load_weights_only_false_sets_default_false():
    original_load = torch.load
    calls = {}

    def fake_load(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "ok"

    torch.load = fake_load
    try:
        patch_torch_load_weights_only_false()
        result = torch.load("dummy.pt")
        assert result == "ok"
        assert calls["kwargs"]["weights_only"] is False
    finally:
        torch.load = original_load

