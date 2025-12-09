import torch
from torch_geometric.data import Data

from gnn.modeling import GraphAutoEncoder, LinkPredictionModel, extract_embeddings


def test_extract_embeddings_link_prediction():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    data = Data(x=x, edge_index=edge_index)
    model = LinkPredictionModel(in_channels=3, hidden_dim=4)

    z = extract_embeddings(model, data)

    assert z.shape[0] == data.num_nodes


def test_extract_embeddings_autoencoder():
    x = torch.randn(5, 2)
    data = Data(x=x, edge_index=torch.tensor([[0], [1]]))
    model = GraphAutoEncoder(in_channels=2, hidden_dim=4, bottleneck=2)

    z = extract_embeddings(model, data)

    assert z.shape[0] == data.num_nodes
