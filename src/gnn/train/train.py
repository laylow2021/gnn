"""Training loop scaffolding."""

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data


def train(
    model: torch.nn.Module,
    train_data: Data,
    val_data: Data | None,
    config: Dict,
) -> Tuple[Dict[str, float], torch.nn.Module]:
    """Minimal training stub; replace with full loop."""
    requested_device = config.get("device", "cpu")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in str(requested_device) and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = model.to(device)

    train_data = train_data.to(device)
    val_data = val_data.to(device) if val_data is not None else None

    model.train()
    loader = DataLoader([train_data], batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(config.get("epochs", 1)):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # TODO: use batch attributes for forward pass
            logits = model(batch.x, batch.edge_index) if hasattr(model, "__call__") else None
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
        # TODO: add validation and metrics
    metrics = {"loss": float(loss.detach().cpu())}
    return metrics, model
