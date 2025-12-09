"""Subgraph extraction utilities for suspicious account seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass
class SubgraphResult:
    seed: str
    nodes: List[str]
    score: float
    edge_count: int
    node_count: int
    edge_type_proportions: dict | None = None


def build_networkx_from_data(data: Data) -> nx.Graph:
    """Convert torch_geometric Data to networkx Graph using account_ids if present."""
    account_ids = getattr(data, "account_ids", list(range(data.num_nodes)))
    G = nx.DiGraph() if data.is_directed() else nx.Graph()
    edges = data.edge_index.t().tolist()
    for src, dst in edges:
        G.add_edge(account_ids[src], account_ids[dst])
    return G


def extract_topk_subgraphs(
    data: Data,
    account_scores: pd.Series,
    k: int = 5,
    hops: int = 1,
    min_size: int = 2,
) -> List[SubgraphResult]:
    """Extract ego subgraphs around top-K scored accounts."""
    G = build_networkx_from_data(data)
    results: List[SubgraphResult] = []
    top_seeds = account_scores.sort_values(ascending=False).head(k)

    for seed, score in top_seeds.items():
        if seed not in G:
            continue
        nodes = list(nx.ego_graph(G, seed, radius=hops).nodes())
        if len(nodes) < min_size:
            continue
        sub = G.subgraph(nodes)
        edge_type_props = None
        if hasattr(data, "metadata") and "feature_schema" in data.metadata:
            # Attempt to compute edge type proportions if present on data.edge_attr
            edge_feature_names = data.metadata["feature_schema"].get("edge", [])
            if data.edge_attr is not None and edge_feature_names:
                type_cols = [i for i, name in enumerate(edge_feature_names) if name.startswith("type_")]
                if type_cols:
                    edge_type_props = {}
                    for name, idx in zip(edge_feature_names, range(len(edge_feature_names))):
                        if not name.startswith("type_"):
                            continue
                        vals = []
                        for src, dst in sub.edges():
                            try:
                                src_idx = data.metadata["account_to_idx"][src]
                                dst_idx = data.metadata["account_to_idx"][dst]
                                # find original edge indices matching this pair
                                mask = (
                                    (data.edge_index[0] == src_idx)
                                    & (data.edge_index[1] == dst_idx)
                                )
                                if mask.any():
                                    vals.append(float(data.edge_attr[mask, edge_feature_names.index(name)].mean()))
                            except KeyError:
                                continue
                        if vals:
                            edge_type_props[name] = float(sum(vals) / len(vals))

        results.append(
            SubgraphResult(
                seed=seed,
                nodes=nodes,
                score=float(score),
                edge_count=sub.number_of_edges(),
                node_count=sub.number_of_nodes(),
                edge_type_proportions=edge_type_props,
            )
        )
    return results
