from admet_shift_reliability.graphs import smiles_to_pyg_graph


def test_smiles_to_pyg_graph_builds_basic_graph():
    graph = smiles_to_pyg_graph("CCO", y=1)

    assert graph.x.size(0) == 3
    assert graph.edge_index.size(0) == 2
    assert graph.y.item() == 1


def test_smiles_to_pyg_graph_rejects_invalid_smiles():
    try:
        smiles_to_pyg_graph("not_a_smiles", y=0)
    except ValueError as exc:
        assert "Invalid SMILES" in str(exc)
    else:
        raise AssertionError("Expected invalid SMILES to raise ValueError")
