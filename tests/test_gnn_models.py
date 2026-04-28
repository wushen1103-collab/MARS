import torch
from torch_geometric.data import Batch

from admet_shift_reliability.graphs import smiles_to_pyg_graph
from admet_shift_reliability.gnn_models import (
    GATBinaryClassifier,
    GINBinaryClassifier,
    MPNNBinaryClassifier,
    build_graph_model,
)


def test_gin_binary_classifier_forward_returns_graph_logits():
    graphs = [
        smiles_to_pyg_graph("CCO", y=1),
        smiles_to_pyg_graph("c1ccccc1", y=0),
    ]
    batch = Batch.from_data_list(graphs)
    model = GINBinaryClassifier(hidden_dim=32, num_layers=3, dropout=0.1)

    logits = model(batch)

    assert logits.shape == (2,)
    assert torch.is_floating_point(logits)


def test_gin_binary_classifier_encode_batch_returns_graph_embeddings():
    graphs = [
        smiles_to_pyg_graph("CCO", y=1),
        smiles_to_pyg_graph("c1ccccc1", y=0),
    ]
    batch = Batch.from_data_list(graphs)
    model = GINBinaryClassifier(hidden_dim=32, num_layers=3, dropout=0.1)

    graph_embeddings = model.encode_batch(batch)

    assert graph_embeddings.shape == (2, 32)
    assert torch.is_floating_point(graph_embeddings)


def test_gat_binary_classifier_forward_returns_graph_logits():
    graphs = [
        smiles_to_pyg_graph("CCO", y=1),
        smiles_to_pyg_graph("c1ccccc1", y=0),
    ]
    batch = Batch.from_data_list(graphs)
    model = GATBinaryClassifier(hidden_dim=32, num_layers=3, dropout=0.1)

    logits = model(batch)

    assert logits.shape == (2,)
    assert torch.is_floating_point(logits)


def test_mpnn_binary_classifier_forward_returns_graph_logits():
    graphs = [
        smiles_to_pyg_graph("CCO", y=1),
        smiles_to_pyg_graph("c1ccccc1", y=0),
    ]
    batch = Batch.from_data_list(graphs)
    model = MPNNBinaryClassifier(hidden_dim=32, num_layers=3, dropout=0.1)

    logits = model(batch)

    assert logits.shape == (2,)
    assert torch.is_floating_point(logits)


def test_build_graph_model_constructs_requested_model():
    gin = build_graph_model("gin", hidden_dim=32, num_layers=3, dropout=0.1)
    gat = build_graph_model("gat", hidden_dim=32, num_layers=3, dropout=0.1)
    mpnn = build_graph_model("mpnn", hidden_dim=32, num_layers=3, dropout=0.1)

    assert isinstance(gin, GINBinaryClassifier)
    assert isinstance(gat, GATBinaryClassifier)
    assert isinstance(mpnn, MPNNBinaryClassifier)

