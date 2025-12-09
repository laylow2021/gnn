"""Lightweight reference models for link prediction and anomaly detection."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv


class LinkPredictionModel(nn.Module):
    """Link predictor with selectable encoder (MLP, GraphSAGE, GCN, GAT)."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        encoder: str = "mlp",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        encoder = encoder.lower()
        if encoder == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self._use_edge_index = False
        elif encoder == "graphsage":
            self.conv1 = SAGEConv(in_channels, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self._use_edge_index = True
        elif encoder == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self._use_edge_index = True
        elif encoder == "gat":
            self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=gat_heads, concat=False)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
            self._use_edge_index = True
        else:
            raise ValueError(f"Unsupported encoder type: {encoder}")
        self.encoder_type = encoder
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_edge_index:
            z = self.conv1(x, edge_index)
            z = torch.relu(z)
            z = self.conv2(z, edge_index)
        else:
            z = self.encoder(x)
        src, dst = edge_index
        scores = self.scorer(z[src], z[dst]).squeeze(-1)
        return scores, z


class GraphAutoEncoder(nn.Module):
    """Dense autoencoder for anomaly detection on node features."""

    def __init__(self, in_channels: int, hidden_dim: int = 64, bottleneck: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def anomaly_score(reconstruction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error per node."""
    return torch.mean((reconstruction - x) ** 2, dim=-1)
