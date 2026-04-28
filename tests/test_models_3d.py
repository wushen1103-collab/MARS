import torch
from torch_geometric.data import Batch

from admet_shift_reliability.graphs_3d import smiles_to_3d_pyg_graph
from admet_shift_reliability.models_3d import SchNetBinaryClassifier


def test_schnet_binary_classifier_forward_returns_graph_logits():
    graphs = [
        smiles_to_3d_pyg_graph("CCO", y=1),
        smiles_to_3d_pyg_graph("c1ccccc1", y=0),
    ]
    batch = Batch.from_data_list(graphs)
    model = SchNetBinaryClassifier(
        hidden_channels=32,
        num_filters=32,
        num_interactions=3,
        num_gaussians=25,
        cutoff=10.0,
    )

    logits = model(batch)

    assert logits.shape == (2,)
    assert torch.is_floating_point(logits)
