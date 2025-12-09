"""Helper to extract node embeddings from trained models."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from .models import GraphAutoEncoder, LinkPredictionModel


def extract_embeddings(model: torch.nn.Module, data: Data) -> torch.Tensor:
    """Return node embeddings given a trained model and graph data."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, LinkPredictionModel):
            _, z = model(data.x, data.edge_index)
            return z
        if isinstance(model, GraphAutoEncoder):
            _, z = model(data.x)
            return z
        # Fallback: try calling model and check outputs
        try:
            output = model(data.x, data.edge_index)
            if isinstance(output, (list, tuple)) and len(output) == 2:
                return output[1]
            if hasattr(output, "shape"):
                return output
        except Exception:
            pass
        return data.x
