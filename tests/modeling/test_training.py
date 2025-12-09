import torch
from torch_geometric.data import Data

from gnn.modeling import GraphAutoEncoder, LinkPredictionModel, train_autoencoder, train_link_prediction


def test_train_link_prediction_runs():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    data = Data(x=x, edge_index=edge_index)
    model = LinkPredictionModel(in_channels=3, hidden_dim=8)

    metrics, trained = train_link_prediction(model, data, epochs=2, lr=1e-2, device="cpu")

    assert "loss" in metrics
    assert 0.0 <= metrics["pos_score_mean"] <= 1.0
    assert 0.0 <= metrics["neg_score_mean"] <= 1.0
    assert "auc" in metrics
    assert trained is model


def test_train_autoencoder_runs():
    x = torch.randn(5, 4)
    data = Data(x=x, edge_index=torch.tensor([[0], [1]]))
    model = GraphAutoEncoder(in_channels=4, hidden_dim=6, bottleneck=3)

    metrics, trained = train_autoencoder(model, data, epochs=2, lr=1e-2, device="cpu", threshold_quantile=0.8)

    assert "loss" in metrics
    assert "threshold" in metrics
    assert "flagged" in metrics
    assert trained is model
