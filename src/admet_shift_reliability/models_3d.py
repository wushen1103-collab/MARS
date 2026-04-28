from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.models import SchNet


class DenseRadiusInteractionGraph(nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index_parts = []
        edge_weight_parts = []

        for graph_id in batch.unique(sorted=True):
            node_idx = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
            if node_idx.numel() <= 1:
                continue

            coords = pos.index_select(0, node_idx)
            distances = torch.cdist(coords, coords, p=2)

            for src_local in range(coords.size(0)):
                neighbor_local = torch.nonzero(
                    (distances[src_local] <= self.cutoff) & (distances[src_local] > 0),
                    as_tuple=False,
                ).view(-1)
                if neighbor_local.numel() == 0:
                    continue

                if neighbor_local.numel() > self.max_num_neighbors:
                    neighbor_dist = distances[src_local, neighbor_local]
                    keep = torch.topk(
                        neighbor_dist,
                        k=self.max_num_neighbors,
                        largest=False,
                    ).indices
                    neighbor_local = neighbor_local[keep]

                src_global = node_idx.new_full((neighbor_local.numel(),), node_idx[src_local].item())
                dst_global = node_idx.index_select(0, neighbor_local)
                edge_index_parts.append(torch.stack([src_global, dst_global], dim=0))
                edge_weight_parts.append(distances[src_local, neighbor_local])

        if not edge_index_parts:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            edge_weight = torch.empty((0,), dtype=pos.dtype, device=pos.device)
            return edge_index, edge_weight

        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_weight = torch.cat(edge_weight_parts, dim=0)
        return edge_index, edge_weight


class SchNetBinaryClassifier(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.interaction_graph = DenseRadiusInteractionGraph(
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
        self.backbone = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=self.interaction_graph,
            max_num_neighbors=max_num_neighbors,
            readout="add",
        )

    def forward(self, batch) -> torch.Tensor:
        logits = self.backbone(batch.z, batch.pos, batch.batch)
        return logits.squeeze(-1)
