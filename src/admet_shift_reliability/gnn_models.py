from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GINConv, NNConv, global_add_pool


class AtomFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_embeddings: int = 256, num_feature_cols: int = 9) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings, hidden_dim) for _ in range(num_feature_cols)]
        )
        self.fallback = nn.LazyLinear(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.int32, torch.int64):
            out = 0
            num_cols = min(x.size(1), len(self.embeddings))
            for idx in range(num_cols):
                out = out + self.embeddings[idx](x[:, idx].clamp(min=0, max=self.embeddings[idx].num_embeddings - 1))
            return out

        return self.fallback(x.float())


class BondFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_embeddings: int = 64, num_feature_cols: int = 3) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings, hidden_dim) for _ in range(num_feature_cols)]
        )
        self.fallback = nn.LazyLinear(hidden_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.dtype in (torch.int32, torch.int64):
            out = 0
            num_cols = min(edge_attr.size(1), len(self.embeddings))
            for idx in range(num_cols):
                out = out + self.embeddings[idx](
                    edge_attr[:, idx].clamp(min=0, max=self.embeddings[idx].num_embeddings - 1)
                )
            return out

        return self.fallback(edge_attr.float())


def _gin_mlp(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINBinaryClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_dim=hidden_dim)
        self.convs = nn.ModuleList([GINConv(_gin_mlp(hidden_dim)) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_batch(self, batch) -> torch.Tensor:
        x = self.encoder(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return global_add_pool(x, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        graph_repr = self.encode_batch(batch)
        logits = self.head(graph_repr).squeeze(-1)
        return logits


class GATBinaryClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_dim=hidden_dim)
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_batch(self, batch) -> torch.Tensor:
        x = self.encoder(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return global_add_pool(x, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        graph_repr = self.encode_batch(batch)
        logits = self.head(graph_repr).squeeze(-1)
        return logits


class MPNNBinaryClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.node_encoder = AtomFeatureEncoder(hidden_dim=hidden_dim)
        self.edge_encoder = BondFeatureEncoder(hidden_dim=hidden_dim)
        self.convs = nn.ModuleList(
            [
                NNConv(
                    hidden_dim,
                    hidden_dim,
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                    ),
                    aggr="mean",
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_batch(self, batch) -> torch.Tensor:
        x = self.node_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index, edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return global_add_pool(x, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        graph_repr = self.encode_batch(batch)
        logits = self.head(graph_repr).squeeze(-1)
        return logits


def build_graph_model(
    model_name: str,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.2,
) -> nn.Module:
    name = model_name.lower()
    if name == "gin":
        return GINBinaryClassifier(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == "gat":
        return GATBinaryClassifier(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == "mpnn":
        return MPNNBinaryClassifier(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unsupported graph model: {model_name}")

