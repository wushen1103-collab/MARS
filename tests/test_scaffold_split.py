from admet_shift_reliability.splits import BemisMurckoScaffoldSplitter


def test_same_scaffold_is_not_split_across_partitions():
    smiles = [
        "Cc1ccccc1",
        "Oc1ccccc1",
        "Nc1ccccc1",
        "CCO",
        "CCCO",
        "CCCCO",
    ]

    splitter = BemisMurckoScaffoldSplitter(valid_frac=0.2, test_frac=0.3)
    split = splitter.split(smiles)

    partition_by_index = {}
    for name in ("train", "valid", "test"):
        for idx in split[name]:
            partition_by_index[idx] = name

    aromatic_partition = {partition_by_index[i] for i in (0, 1, 2)}
    alcohol_partition = {partition_by_index[i] for i in (3, 4, 5)}

    assert len(aromatic_partition) == 1
    assert len(alcohol_partition) == 1


def test_split_covers_all_valid_indices_without_overlap():
    smiles = [
        "CCO",
        "CCN",
        "c1ccccc1",
        "CC(=O)O",
        "CCCl",
        "CCBr",
    ]

    splitter = BemisMurckoScaffoldSplitter(valid_frac=0.2, test_frac=0.3)
    split = splitter.split(smiles)

    combined = set(split["train"]) | set(split["valid"]) | set(split["test"])
    assert combined == set(range(len(smiles)))
    assert set(split["train"]).isdisjoint(split["valid"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["valid"]).isdisjoint(split["test"])


def test_invalid_smiles_raise_clear_error():
    splitter = BemisMurckoScaffoldSplitter(valid_frac=0.2, test_frac=0.2)

    try:
        splitter.split(["CCO", "not_a_smiles"])
    except ValueError as exc:
        assert "Invalid SMILES" in str(exc)
    else:
        raise AssertionError("Expected invalid SMILES to raise ValueError")
