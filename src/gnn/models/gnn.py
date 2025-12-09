"""GNN model factory."""

from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv


def build_model(config: dict) -> nn.Module:
    """Return a simple GNN based on config; extend as architectures grow."""
    model_type = config.get("type", "gcn").lower()
    hidden_dims = config.get("hidden_dims", [64, 64])
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]

    if model_type == "gcn":
        conv_cls = GCNConv
    elif model_type == "gat":
        conv_cls = lambda in_c, out_c: GATv2Conv(in_c, out_c, heads=config.get("heads", 4), concat=False)
    elif model_type == "sage":
        conv_cls = SAGEConv
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    layers = []
    last_dim = in_channels
    for dim in hidden_dims:
        layers.append(conv_cls(last_dim, dim))
        layers.append(nn.ReLU())
        last_dim = dim
    layers.append(conv_cls(last_dim, out_channels))
    return nn.Sequential(*layers)

