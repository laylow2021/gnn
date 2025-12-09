"""Plotting utilities for training and evaluation."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data


def plot_learning_curves(history: Dict[str, List[float]], output_path: str | None = None):
    """Plot loss/metric curves from training history."""
    fig, ax = plt.subplots()
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_transaction_graph(
    graph: Data,
    max_edges: int = 300,
    seed: int = 42,
    figsize: tuple[int, int] = (10, 8),
    color_by_bank: bool = True,
) -> tuple:
    """Visualize a transaction graph (accounts as nodes, payments as directed edges)."""
    if graph.edge_index.size(1) == 0:
        raise ValueError("Graph has no edges to visualize.")

    edge_index = graph.edge_index[:, :max_edges]
    G = nx.DiGraph()
    account_ids = getattr(graph, "account_ids", list(range(graph.num_nodes)))
    idx_lookup = getattr(graph, "metadata", {}).get("account_to_idx", {a: i for i, a in enumerate(account_ids)})

    for src, dst in edge_index.t().tolist():
        G.add_edge(account_ids[src], account_ids[dst])

    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=figsize)

    nodelist = list(G.nodes())

    if color_by_bank and hasattr(graph, "bank_ids") and graph.bank_ids:
        bank_lookup = graph.bank_ids
        bank_feature = None
        if graph.x is not None and graph.x.shape[1] >= len(bank_lookup):
            bank_feature = torch.argmax(graph.x[:, : len(bank_lookup)], dim=1).tolist()
        if bank_feature is None and graph.x is not None:
            # fallback to first column as scalar bank id (from loader scalar path)
            bank_feature = graph.x[:, 0].tolist()

        colors = []
        for node in nodelist:
            idx = idx_lookup.get(node)
            if idx is None or bank_feature is None:
                colors.append(0)
            else:
                colors.append(bank_feature[idx])
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=80, node_color=colors, cmap="tab20", ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=80, node_color="#4e79a7", ax=ax)

    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, width=0.7, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.set_title(getattr(graph, "metadata", {}).get("label", "transaction graph"))
    ax.axis("off")
    return fig, ax
