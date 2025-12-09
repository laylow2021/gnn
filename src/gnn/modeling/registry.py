"""Registry to support plug-and-play model selection driven by config."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from gnn.config.schema import ModelConfig, ModelTask
from .models import GraphAutoEncoder, LinkPredictionModel


class ModelRegistry:
    """Instantiate models based on declarative configuration."""

    def __init__(self) -> None:
        self._builders: Dict[ModelTask, Callable[..., object]] = {
            ModelTask.LINK_PREDICTION: self._build_link_prediction,
            ModelTask.ANOMALY_AUTOENCODER: self._build_autoencoder,
        }

    def build(self, config: ModelConfig, in_channels: int, out_channels: Optional[int] = None):
        if config.task not in self._builders:
            raise ValueError(f"Unsupported model task: {config.task}")
        return self._builders[config.task](config, in_channels, out_channels)

    def _build_link_prediction(
        self, config: ModelConfig, in_channels: int, out_channels: Optional[int] = None
    ) -> LinkPredictionModel:
        hidden_dim = int(config.params.get("hidden_dim", 64))
        encoder = config.params.get("encoder", "mlp")
        gat_heads = int(config.params.get("gat_heads", 4))
        return LinkPredictionModel(in_channels=in_channels, hidden_dim=hidden_dim, encoder=encoder, gat_heads=gat_heads)

    def _build_autoencoder(
        self, config: ModelConfig, in_channels: int, out_channels: Optional[int] = None
    ) -> GraphAutoEncoder:
        hidden_dim = int(config.params.get("hidden_dim", 64))
        bottleneck = int(config.params.get("bottleneck", 16))
        return GraphAutoEncoder(in_channels=in_channels, hidden_dim=hidden_dim, bottleneck=bottleneck)
