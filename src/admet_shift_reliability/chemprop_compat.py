from __future__ import annotations


def patch_pandas_rdkit_compat() -> None:
    import pandas.io.formats.format as fmt
    from pandas.io.formats.printing import get_adjustment

    if not hasattr(fmt, "get_adjustment"):
        fmt.get_adjustment = get_adjustment


def patch_torch_load_weights_only_false() -> None:
    import torch

    original_load = torch.load
    if getattr(original_load, "_admet_shift_patched", False):
        return

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    patched_load._admet_shift_patched = True
    torch.load = patched_load

