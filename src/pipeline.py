import torch
import yaml
import os
from typing import Dict, Any
from src.data.processor import DataProcessor
from src.features.classical import ClassicalLayer
from src.graph.builder import GraphBuilder
from src.models.gnn import AMLGraphAutoencoder
from src.utils.reproducibility import get_device, save_checkpoint, load_checkpoint

class AMLPipeline:
    """
    Orchestrator for the Layered AML Detection Pipeline:
    Data Processing -> Classical ML (Clustering/IForest) -> GNN (Graph Autoencoder).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config)
        self.processor = DataProcessor(config)
        self.classical = ClassicalLayer(config)
        self.graph_builder = GraphBuilder(config)
        self.model = None

    def train(self, raw_tx_df: pd.DataFrame):
        """Full end-to-end training pipeline."""
        print("1. Cleaning and Aggregating Data...")
        clean_df = self.processor.clean(raw_tx_df)
        cust_df = self.processor.aggregate_to_customer(clean_df)

        print("2. Running Classical ML (Clustering & IForest)...")
        enriched_cust_df = self.classical.fit_predict(cust_df)

        print("3. Building Graph with Combined Features...")
        # Merge classical scores back to transactions if needed, or pass separately
        # to GraphBuilder to use as node features.
        data = self.graph_builder.build_graph(clean_df)
        # TODO: Ensure node_features in config includes 'local_anomaly_score'
        
        print("4. Training GNN Autoencoder...")
        # Training logic here...
        pass

    def save(self, path: str):
        """Bundle and save all pipeline artifacts."""
        # TODO: Update save_checkpoint to handle DataProcessor and ClassicalLayer
        pass
