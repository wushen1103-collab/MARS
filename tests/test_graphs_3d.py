import torch

from admet_shift_reliability.graphs_3d import smiles_to_3d_pyg_graph


def test_smiles_to_3d_pyg_graph_builds_schnet_inputs():
    graph = smiles_to_3d_pyg_graph("CCO", y=1)

    assert graph.z.dtype == torch.long
    assert graph.z.tolist() == [6, 6, 8]
    assert graph.pos.shape == (3, 3)
    assert torch.isfinite(graph.pos).all()
    assert graph.y.item() == 1
