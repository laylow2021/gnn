"""Task-specific training loops for plug-and-play models."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

from .models import anomaly_score


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in requested and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def train_link_prediction(
    model: nn.Module,
    data: Data,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> Tuple[Dict[str, float], nn.Module]:
    """Train a link prediction model with negative sampling."""
    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    data = data.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_scores, _ = model(data.x, data.edge_index)
        neg_edge_index = negative_sampling(
            data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )
        neg_scores, _ = model(data.x, neg_edge_index)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
        )
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        pos_scores, _ = model(data.x, data.edge_index)
        neg_edge_index = negative_sampling(
            data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )
        neg_scores, _ = model(data.x, neg_edge_index)
        try:
            auc = float(
                roc_auc_score(
                    torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu(),
                    torch.cat([pos_scores, neg_scores]).cpu(),
                )
            )
        except ValueError:
            auc = float("nan")
        metrics = {
            "loss": float(loss.detach().cpu()),
            "pos_score_mean": float(torch.sigmoid(pos_scores).mean().cpu()),
            "neg_score_mean": float(torch.sigmoid(neg_scores).mean().cpu()),
            "auc": auc,
        }
    return metrics, model


def train_autoencoder(
    model: nn.Module,
    data: Data,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    threshold_quantile: float = 0.99,
) -> Tuple[Dict[str, float], nn.Module]:
    """Train an autoencoder for anomaly detection on node features."""
    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    data = data.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction, _ = model(data.x)
        loss = criterion(reconstruction, data.x)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        reconstruction, _ = model(data.x)
        final_loss = criterion(reconstruction, data.x)
        scores = anomaly_score(reconstruction, data.x)
        thresh = float(torch.quantile(scores, threshold_quantile).cpu())
        flagged = int((scores > thresh).sum().cpu())
        metrics = {
            "loss": float(final_loss.detach().cpu()),
            "threshold": thresh,
            "flagged": flagged,
        }
    return metrics, model
