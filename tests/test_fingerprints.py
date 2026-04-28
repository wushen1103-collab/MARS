import numpy as np

from admet_shift_reliability.features import morgan_fingerprint_matrix


def test_morgan_fingerprint_matrix_has_expected_shape_and_dtype():
    smiles = ["CCO", "c1ccccc1"]

    fps = morgan_fingerprint_matrix(smiles, radius=2, n_bits=128)

    assert fps.shape == (2, 128)
    assert fps.dtype == np.float32


def test_morgan_fingerprint_matrix_rejects_invalid_smiles():
    try:
        morgan_fingerprint_matrix(["CCO", "bad_smiles"])
    except ValueError as exc:
        assert "Invalid SMILES" in str(exc)
    else:
        raise AssertionError("Expected invalid SMILES to raise ValueError")
