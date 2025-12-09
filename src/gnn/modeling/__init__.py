"""Modeling and anomaly modules with plug-and-play registry."""

from .models import GraphAutoEncoder, LinkPredictionModel
from .embeddings import extract_embeddings
from .training import train_autoencoder, train_link_prediction
from .registry import ModelRegistry

__all__ = [
    "GraphAutoEncoder",
    "LinkPredictionModel",
    "ModelRegistry",
    "extract_embeddings",
    "train_autoencoder",
    "train_link_prediction",
]
