import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, k_hop_subgraph
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Optional
import numpy as np
import os

class AnomalyInterpreter:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

    def plot_loss_curves(self, history: Dict[str, List[float]]):
        """Plot total, node, and edge loss over epochs."""
        plt.figure(figsize=(10, 6))
        for label, values in history.items():
            plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def explain_node_anomalies(self, x: torch.Tensor, x_recon: torch.Tensor):
        per_feature_error = (x - x_recon)**2
        mean_error_per_feat = per_feature_error.mean(dim=0).detach().cpu().numpy()
        feat_names = self.config['graph']['node_features']
        importance_df = pd.DataFrame({'Feature': feat_names, 'Importance (Mean MSE)': mean_error_per_feat})
        importance_df = importance_df.sort_values('Importance (Mean MSE)', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance (Mean MSE)', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title("Global Node Feature Importance (Error Contribution)")
        plt.show()
        return importance_df

    def local_node_explanation(self, node_id: int, x: torch.Tensor, x_recon: torch.Tensor):
        node_error = (x[node_id] - x_recon[node_id])**2
        feat_names = self.config['graph']['node_features']
        plt.figure(figsize=(8, 4))
        plt.barh(feat_names, node_error.detach().cpu().numpy(), color='salmon')
        plt.xlabel("Reconstruction Error (MSE)")
        plt.title(f"Local Feature Importance for Node {node_id}")
        plt.show()

    def explain_edge_anomalies(self, edge_attr: torch.Tensor, edge_attr_recon: torch.Tensor):
        per_feat_error = (edge_attr - edge_attr_recon)**2
        mean_error = per_feat_error.mean(dim=0).detach().cpu().numpy()
        feat_names = self.config['graph']['edge_features']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mean_error, y=feat_names, hue=feat_names, palette='magma', legend=False)
        plt.title("Global Edge Feature Importance (Error Contribution)")
        plt.show()

    def local_perspective(self, node_id: int, data: Any, node_mse: torch.Tensor, edge_mse: torch.Tensor, 
                          num_hops: int = 1, inv_map: Optional[Dict[int, Any]] = None):
        """Enhanced neighborhood view with hops and original ID labels."""
        self.model.eval()
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            node_id, num_hops, data.edge_index, relabel_nodes=False
        )
        G = nx.Graph()

        # Determine labels: Use original IDs if inv_map is provided
        labels = {}
        for n_idx in subset.tolist():
            orig_id = inv_map.get(n_idx, n_idx) if inv_map else n_idx
            G.add_node(n_idx, mse=node_mse[n_idx].item(), orig_id=orig_id)
            labels[n_idx] = orig_id

        sub_edges = data.edge_index[:, edge_mask].cpu().numpy()
        sub_edge_mse = edge_mse[edge_mask].cpu().numpy()
        for i in range(sub_edges.shape[1]):
            u, v = sub_edges[:, i]
            G.add_edge(u, v, mse=sub_edge_mse[i])

        node_threshold = torch.quantile(node_mse, 0.95).item()
        edge_threshold = torch.quantile(edge_mse, 0.95).item()

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        for n in G.nodes():
            if n == node_id: node_colors.append('red')
            elif G.nodes[n]['mse'] >= node_threshold: node_colors.append('orange')
            else: node_colors.append('skyblue')

        edge_colors = []
        for u, v in G.edges():
            if G.edges[u, v]['mse'] >= edge_threshold: edge_colors.append('orange')
            else: edge_colors.append('gray')

        nx.draw(G, pos, labels=labels, with_labels=True, node_color=node_colors, edge_color=edge_colors, 
                node_size=800, alpha=0.8, width=2)

        target_label = labels.get(node_id, node_id)
        plt.title(f"{num_hops}-Hop Neighborhood for Customer: {target_label}\n(Red: Target, Amber: High Risk)")
        plt.show()

    def global_perspective(self, edge_mse: torch.Tensor):
        plt.figure(figsize=(10, 6))
        sns.histplot(edge_mse.detach().cpu().numpy(), bins=50, kde=True, color='purple')
        plt.title("Distribution of Edge Reconstruction Errors")
        plt.show()

    def save_anomalies_to_excel(self, data: Any, node_mse: torch.Tensor, edge_mse: torch.Tensor, 
                               tx_df: pd.DataFrame, output_path: str, inv_map: Optional[Dict[int, Any]] = None):
        """Export ranked anomalies with original IDs and audit trails."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        c_id_col = self.config['data']['column_mapping']['customer_id']

        # 1. Node Anomalies
        node_features = self.config['graph']['node_features']
        node_df = pd.DataFrame(data.x.cpu().numpy(), columns=node_features)
        node_df['node_idx'] = range(len(node_df))

        # Map back to original IDs
        if inv_map:
            node_df['customer_id'] = node_df['node_idx'].map(inv_map)
        else:
            node_df['customer_id'] = node_df['node_idx']

        node_df['anomaly_score'] = node_mse.cpu().numpy()
        node_df = node_df.sort_values('anomaly_score', ascending=False)

        # 2. Edge Anomalies
        edge_df = data.collapsed_edges_df.copy()
        edge_df['edge_anomaly_score'] = edge_mse.cpu().numpy()
        edge_df['source_node_score'] = node_mse[edge_df['source'].values].cpu().numpy()
        edge_df['target_node_score'] = node_mse[edge_df['target'].values].cpu().numpy()

        # Original IDs should already be in collapsed_edges_df as source_id/target_id
        # from the updated GraphBuilder, but we ensure columns are clean.

        # Convert list metadata to strings
        audit_cols = ['out_tx_ids', 'out_amounts', 'out_dates', 'in_tx_ids', 'in_amounts', 'in_dates']
        for col in audit_cols:
            if col in edge_df.columns:
                edge_df[col] = edge_df[col].apply(lambda x: str(x))

        edge_df = edge_df.sort_values('edge_anomaly_score', ascending=False)

        # 3. Write to Excel
        with pd.ExcelWriter(output_path) as writer:
            node_df.to_excel(writer, sheet_name='Node Anomalies', index=False)
            edge_df.to_excel(writer, sheet_name='Edge Anomalies', index=False)

            # Map top customers back for context
            top_ids = node_df.head(20)['customer_id'].tolist()
            raw_context = tx_df[tx_df[c_id_col].isin(top_ids)]
            raw_context.to_excel(writer, sheet_name='Top Node Raw TX', index=False)

        print(f"Anomalies exported to {output_path}")