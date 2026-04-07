import torch
import yaml
import os
import pandas as pd
from typing import Dict, Any, Optional
# from src.data.processor import DataProcessor
# from src.features.classical import ClassicalLayer
from src.graph.builder import GraphBuilder
from src.models.gnn import AMLGraphAutoencoder, compute_combined_loss
from src.utils.reproducibility import get_device, save_checkpoint, load_checkpoint
from src.explain.interpreter import AnomalyInterpreter
from matplotlib.backends.backend_pdf import PdfPages

class AMLPipeline:
    """
    Orchestrator for the Layered AML Detection Pipeline:
    Data Processing -> Classical ML (Clustering/IForest) -> GNN (Graph Autoencoder).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config)
        # self.processor = DataProcessor(config)
        # self.classical = ClassicalLayer(config)
        self.graph_builder = GraphBuilder(config)
        self.model = None
        self.node_scaler = None
        self.edge_scaler = None

    def train(self, raw_tx_df: pd.DataFrame):
        """Full end-to-end training pipeline."""
        # print("1. Cleaning and Aggregating Data...")
        # clean_df = self.processor.clean(raw_tx_df)
        # cust_df = self.processor.aggregate_to_customer(clean_df)

        # print("2. Running Classical ML (Clustering & IForest)...")
        # enriched_cust_df = self.classical.fit_predict(cust_df)
        
        # Temporary bypass until modules are ready:
        clean_df = raw_tx_df 

        print("3. Building Graph...")
        data = self.graph_builder.build_graph(clean_df)
        
        print("4. Training GNN Autoencoder...")
        # Training logic here... (omitted for brevity)
        pass

    def save(self, path: str):
        """Bundle and save all pipeline artifacts."""
        save_checkpoint(self.model, self.config, path, self.node_scaler, self.edge_scaler)

    def load_from_checkpoint(self, path: str):
        """Load model and scalers from a checkpoint directory."""
        # We need to know in_channels and edge_in_channels to initialize the model
        # These are usually derived from the data or stored in config
        # For MGNN, they are based on node_features and edge_features
        dummy_node_feats = self.config['graph'].get('node_features', [])
        in_channels = len(dummy_node_feats) if dummy_node_feats else 1
        edge_in_channels = len(self.config['graph']['edge_features'])

        self.model, self.config, self.node_scaler, self.edge_scaler = load_checkpoint(
            AMLGraphAutoencoder, path, in_channels, edge_in_channels, device=self.device
        )
        print(f"Pipeline loaded from {path}")

    def score(self, raw_tx_df: pd.DataFrame, pdf_path: str, excel_path: str, 
              node_top_percent: float = 0.05, edge_top_percent: float = 0.05,
              min_edge_risk_quantile: float = 0.50,
              local_neighborhood_node_ids: Optional[list] = None):
        """
        Comprehensive scoring function:
        1. Process and Build Graph
        2. Scale Features
        3. Inference & MSE Calculation
        4. Visualize to PDF & Export to Excel
        5. Optional Local Neighborhood Plotting
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_from_checkpoint() first.")

        print("--- Starting Scoring Process ---")
        self.model.eval()

        # 1. Build Graph
        print("Building graph from raw transactions...")
        data = self.graph_builder.build_graph(raw_tx_df)
        data = data.to(self.device)

        # 2. Scale Features
        if self.node_scaler:
            x_np = data.x.cpu().numpy()
            x_scaled = self.node_scaler.transform(x_np)
            data.x = torch.tensor(x_scaled, dtype=torch.float).to(self.device)
        
        if self.edge_scaler:
            edge_attr_np = data.edge_attr.cpu().numpy()
            edge_attr_scaled = self.edge_scaler.transform(edge_attr_np)
            data.edge_attr = torch.tensor(edge_attr_scaled, dtype=torch.float).to(self.device)

        # 3. Inference
        print("Running inference...")
        with torch.no_grad():
            h, x_recon, edge_attr_recon = self.model(data.x, data.edge_index, data.edge_attr)
            
            # Calculate per-element MSE for interpretation
            node_mse_per_node = torch.mean((data.x - x_recon)**2, dim=1)
            edge_mse_per_edge = torch.mean((data.edge_attr - edge_attr_recon)**2, dim=1)
            
            _, node_loss, edge_loss = compute_combined_loss(
                data.x, x_recon, data.edge_attr, edge_attr_recon, self.config
            )

        print(f"Scoring complete. Mean Node MSE: {node_loss:.4f} | Mean Edge MSE: {edge_loss:.4f}")

        # 4. Reporting
        interpreter = AnomalyInterpreter(self.model, self.config)
        
        print(f"Generating PDF report: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            # Global Perspectives
            interpreter.global_perspective(edge_mse_per_edge, pdf=pdf)
            interpreter.explain_node_anomalies(data.x, x_recon, pdf=pdf)
            interpreter.explain_edge_anomalies(data.edge_attr, edge_attr_recon, pdf=pdf)
            
            # Local Perspectives Logic
            target_ids = []
            if local_neighborhood_node_ids is None:
                # Option 1: None -> Skip plotting local graphs
                pass
            elif len(local_neighborhood_node_ids) > 0:
                # Option 2: List of specific IDs -> Plot those
                target_ids = local_neighborhood_node_ids
            else:
                # Option 3: Empty List -> Plot top node and edge anomalies
                num_nodes = max(1, int(len(node_mse_per_node) * node_top_percent))
                num_edges = max(1, int(len(edge_mse_per_edge) * edge_top_percent))
                
                # Get Top Node Customers
                top_node_indices = torch.topk(node_mse_per_node, num_nodes).indices.tolist()
                top_node_customers = [self.graph_builder.inv_customer_map.get(idx) for idx in top_node_indices]
                
                # Get Top Edge Customers (Source and Target)
                top_edge_indices = torch.topk(edge_mse_per_edge, num_edges).indices.tolist()
                top_edge_sources = data.collapsed_edges_df.iloc[top_edge_indices]['source_id'].tolist()
                top_edge_targets = data.collapsed_edges_df.iloc[top_edge_indices]['target_id'].tolist()
                
                target_ids = list(set(top_node_customers + top_edge_sources + top_edge_targets))
                print(f"Automated Local Perspective: Plotting {len(target_ids)} unique customers from top anomalies.")

            for cust_id in target_ids:
                internal_idx = self.graph_builder.customer_map.get(cust_id)
                if internal_idx is not None and internal_idx in range(len(data.x)):
                    interpreter.local_perspective(
                        internal_idx, data, node_mse_per_node, edge_mse_per_edge, 
                        inv_map=self.graph_builder.inv_customer_map, pdf=pdf,
                        min_edge_risk_quantile=min_edge_risk_quantile
                    )

        print(f"Generating Excel report: {excel_path}")
        interpreter.save_anomalies_to_excel(
            data, node_mse_per_node, edge_mse_per_edge, raw_tx_df, excel_path,
            inv_map=self.graph_builder.inv_customer_map,
            node_top_percent=node_top_percent, edge_top_percent=edge_top_percent
        )

        return node_mse_per_node, edge_mse_per_edge
