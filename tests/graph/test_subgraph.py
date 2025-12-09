import pandas as pd
import torch
from torch_geometric.data import Data

from gnn.graph.subgraph import extract_topk_subgraphs


def test_extract_topk_subgraphs_returns_ego_networks():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    data = Data(edge_index=edge_index)
    data.account_ids = ["a", "b", "c", "d"]
    data.num_nodes = 4

    scores = pd.Series({"a": 0.9, "b": 0.5, "c": 0.2})

    results = extract_topk_subgraphs(data, scores, k=2, hops=1, min_size=2)

    seeds = [r.seed for r in results]
    assert "a" in seeds
    assert all(r.node_count >= 2 for r in results)
