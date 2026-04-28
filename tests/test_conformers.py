import numpy as np

from admet_shift_reliability.conformers import generate_conformer


def test_generate_conformer_returns_atomic_numbers_and_positions():
    atomic_numbers, positions = generate_conformer("CCO", seed=7)

    assert atomic_numbers.tolist() == [6, 6, 8]
    assert positions.shape == (3, 3)
    assert np.isfinite(positions).all()


def test_generate_conformer_rejects_invalid_smiles():
    try:
        generate_conformer("not_a_smiles")
    except ValueError as exc:
        assert "Invalid SMILES" in str(exc)
    else:
        raise AssertionError("Expected invalid SMILES to raise ValueError")
