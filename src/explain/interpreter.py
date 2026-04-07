import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, k_hop_subgraph
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Optional
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

class AnomalyInterpreter:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

    def plot_loss_curves(self, history: Dict[str, List[float]], pdf: Optional[PdfPages] = None):
        """Plot total, node, and edge loss over epochs."""
        plt.figure(figsize=(10, 6))
        for label, values in history.items():
            plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

    def explain_node_anomalies(self, x: torch.Tensor, x_recon: torch.Tensor, pdf: Optional[PdfPages] = None):
        """Visualizes global node feature importance using Bar, Violin, and Waterfall plots."""
        per_feature_error = (x - x_recon)**2
        errors_np = per_feature_error.detach().cpu().numpy()
        mean_error_per_feat = errors_np.mean(axis=0)
        
        feat_names = self.config['graph'].get('node_features', [])
        if not feat_names or len(feat_names) != len(mean_error_per_feat):
            feat_names = [f"feat_{i}" for i in range(len(mean_error_per_feat))]
            if len(feat_names) == 1: feat_names = ['tx_count']
            
        importance_df = pd.DataFrame({'Feature': feat_names, 'Importance (Mean MSE)': mean_error_per_feat})
        importance_df = importance_df.sort_values('Importance (Mean MSE)', ascending=False)

        # 1. Bar Chart (Existing)
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=importance_df, x='Importance (Mean MSE)', y='Feature', hue='Feature', palette='viridis')
        if ax.get_legend(): ax.get_legend().remove()
        plt.title("Global Node Feature Importance (Bar: Mean Error)")
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        # 2. Violin Plot (Distribution of errors)
        error_dist_df = pd.DataFrame(errors_np, columns=feat_names)
        # Log scale often helps visualize MSE distributions which are usually heavily skewed
        error_dist_melted = error_dist_df.melt(var_name='Feature', value_name='MSE')
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(data=error_dist_melted, x='MSE', y='Feature', hue='Feature', palette='viridis', inner="quart")
        if ax.get_legend(): ax.get_legend().remove()
        plt.xscale('log')
        plt.title("Global Node Error Distribution (Violin: Quartiles & Density)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        # 3. Waterfall Plot (Contribution to Total Average Error)
        self._plot_waterfall(importance_df['Feature'].tolist(), importance_df['Importance (Mean MSE)'].tolist(), 
                             "Global Node Risk Decomposition (Waterfall)", pdf=pdf)
        
        return importance_df

    def explain_edge_anomalies(self, edge_attr: torch.Tensor, edge_attr_recon: torch.Tensor, pdf: Optional[PdfPages] = None):
        """Visualizes global edge feature importance using Bar, Violin, and Waterfall plots."""
        per_feat_error = (edge_attr - edge_attr_recon)**2
        errors_np = per_feat_error.detach().cpu().numpy()
        mean_error = errors_np.mean(axis=0)
        feat_names = self.config['graph']['edge_features']
        
        importance_df = pd.DataFrame({'Feature': feat_names, 'Importance (Mean MSE)': mean_error})
        importance_df = importance_df.sort_values('Importance (Mean MSE)', ascending=False)

        # 1. Bar Chart
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=importance_df, x='Importance (Mean MSE)', y='Feature', hue='Feature', palette='magma')
        if ax.get_legend(): ax.get_legend().remove()
        plt.title("Global Edge Feature Importance (Bar: Mean Error)")
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        # 2. Violin Plot
        error_dist_df = pd.DataFrame(errors_np, columns=feat_names)
        error_dist_melted = error_dist_df.melt(var_name='Feature', value_name='MSE')
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(data=error_dist_melted, x='MSE', y='Feature', hue='Feature', palette='magma', inner="quart")
        if ax.get_legend(): ax.get_legend().remove()
        plt.xscale('log')
        plt.title("Global Edge Error Distribution (Violin: Quartiles & Density)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        # 3. Waterfall Plot
        self._plot_waterfall(importance_df['Feature'].tolist(), importance_df['Importance (Mean MSE)'].tolist(), 
                             "Global Edge Risk Decomposition (Waterfall)", pdf=pdf)

    def _plot_waterfall(self, labels: List[str], values: List[float], title: str, pdf: Optional[PdfPages] = None):
        """Custom Waterfall plot using Matplotlib."""
        data = pd.DataFrame({'label': labels, 'value': values})
        data['cumulative'] = data['value'].cumsum().shift(1).fillna(0)
        total = data['value'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a robust colormap access
        cmap = plt.get_cmap('tab20c')
        
        # Plot the bars
        for i, row in data.iterrows():
            color = cmap(i % 20)
            ax.bar(row['label'], row['value'], bottom=row['cumulative'], color=color, edgecolor='black', alpha=0.8)
            # Add connector lines
            if i > 0:
                ax.plot([i-1, i], [row['cumulative'], row['cumulative']], color='gray', linestyle='--', alpha=0.5)
        
        # Add a final 'Total' bar
        ax.bar(['Total'], [total], color='gray', edgecolor='black', alpha=0.6)
        ax.plot([len(data)-1, len(data)], [total, total], color='gray', linestyle='--', alpha=0.5)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Cumulative Mean MSE")
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

    def local_perspective(self, node_id: int, data: Any, node_mse: torch.Tensor, edge_mse: torch.Tensor, 
                          num_hops: int = 1, inv_map: Optional[Dict[int, Any]] = None,
                          min_edge_risk_quantile: float = 0.0, pdf: Optional[PdfPages] = None):
        """Enhanced neighborhood view with financial labels and volume-based node scaling."""
        self.model.eval()
        
        # 1. Robust Bidirectional Node Discovery
        from torch_geometric.utils import to_undirected
        undirected_index = to_undirected(data.edge_index)
        subset, _, _, _ = k_hop_subgraph(
            node_id, num_hops, undirected_index, relabel_nodes=False
        )
        
        # 2. Edge Recovery and Feature Alignment
        edge_mask = torch.isin(data.edge_index[0], subset) & torch.isin(data.edge_index[1], subset)
        sub_edge_mse = edge_mse[edge_mask].cpu().numpy()
        collapsed_sub = data.collapsed_edges_df.iloc[edge_mask.cpu().numpy()].copy()
        
        edge_threshold_val = torch.quantile(edge_mse, min_edge_risk_quantile).item()
        G = nx.DiGraph()
        
        labels = {}
        for n_idx in subset.tolist():
            orig_id = inv_map.get(n_idx, n_idx) if inv_map else n_idx
            G.add_node(n_idx, mse=node_mse[n_idx].item(), orig_id=orig_id, volume=0.0)
            labels[n_idx] = orig_id
            
        # 3. Add edges and calculate node volumes for sizing
        edge_labels = {}
        for i, (idx, row) in enumerate(collapsed_sub.iterrows()):
            mse_val = sub_edge_mse[i]
            if mse_val >= edge_threshold_val:
                u, v = int(row['source']), int(row['target'])
                amt = row['total_inferred_amt']
                G.add_edge(u, v, mse=mse_val, amt=amt)
                
                # Accrue volume to nodes for scaling
                G.nodes[u]['volume'] += amt
                G.nodes[v]['volume'] += amt
                
                # Format label (e.g. $5.5k or $900)
                label_str = f"${amt/1000:.1f}k" if amt >= 1000 else f"${amt:.0f}"
                edge_labels[(u, v)] = label_str
        
        nodes_to_remove = [n for n in G.nodes() if G.degree(n) == 0 and n != node_id]
        G.remove_nodes_from(nodes_to_remove)

        node_threshold = torch.quantile(node_mse, 0.95).item()
        edge_high_risk_threshold = torch.quantile(edge_mse, 0.95).item()
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=0.5) 
        
        node_colors = []
        for n in G.nodes():
            if n == node_id: node_colors.append('red') 
            elif G.nodes[n]['mse'] >= node_threshold: node_colors.append('orange') 
            else: node_colors.append('skyblue')
            
        edge_colors = []
        for u, v in G.edges():
            if G.edges[u, v]['mse'] >= edge_high_risk_threshold: edge_colors.append('red')
            else: edge_colors.append('gray')
            
        nx.draw(G, pos, labels={n: labels[n] for n in G.nodes()}, with_labels=True, 
                node_color=node_colors, edge_color=edge_colors, node_size=800, 
                alpha=0.8, width=2, arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='darkgreen')
        
        target_label = labels.get(node_id, node_id)
        plt.title(f"{num_hops}-Hop Neighborhood for Customer: {target_label}\n"
                  f"(Green Labels = Total Amount | Red = High Risk)")
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

    def global_perspective(self, edge_mse: torch.Tensor, pdf: Optional[PdfPages] = None):
        plt.figure(figsize=(10, 6))
        sns.histplot(edge_mse.detach().cpu().numpy(), bins=50, kde=True, color='purple')
        plt.title("Distribution of Edge Reconstruction Errors")
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

    def save_anomalies_to_excel(self, data: Any, node_mse: torch.Tensor, edge_mse: torch.Tensor, 
                               tx_df: pd.DataFrame, output_path: str, inv_map: Optional[Dict[int, Any]] = None,
                               node_top_percent: float = 0.05, edge_top_percent: float = 0.05):
        """
        Export filtered ranked anomalies to Excel and return full DataFrames.
        Includes feature-level MSE decomposition and merged raw transactions.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        c_id_col = self.config['data']['column_mapping']['customer_id']

        # 1. Prepare Full Node Anomalies DF with Feature Decomposition
        # Re-calculate reconstructions to get per-feature MSE
        self.model.eval()
        with torch.no_grad():
            _, x_recon, edge_attr_recon = self.model(data.x, data.edge_index, data.edge_attr)
        
        node_features = self.config['graph'].get('node_features', [])
        x_np = data.x.cpu().numpy()
        if not node_features or len(node_features) != x_np.shape[1]:
            node_features = [f"feat_{i}" for i in range(x_np.shape[1])]
            if len(node_features) == 1: node_features = ['tx_count']

        node_mse_per_feat = (data.x - x_recon)**2
        node_df_full = pd.DataFrame(x_np, columns=node_features)
        for i, feat in enumerate(node_features):
            node_df_full[f'mse_{feat}'] = node_mse_per_feat[:, i].cpu().numpy()
            
        node_df_full['node_idx'] = range(len(node_df_full))
        if inv_map:
            node_df_full['customer_id'] = node_df_full['node_idx'].map(inv_map)
        else:
            node_df_full['customer_id'] = node_df_full['node_idx']
        node_df_full['anomaly_score'] = node_mse.cpu().numpy()
        node_df_full = node_df_full.sort_values('anomaly_score', ascending=False)

        # 2. Prepare Full Edge Anomalies DF with Feature Decomposition
        edge_features = self.config['graph']['edge_features']
        edge_mse_per_feat = (data.edge_attr - edge_attr_recon)**2
        
        edge_df_full = data.collapsed_edges_df.copy()
        for i, feat in enumerate(edge_features):
            edge_df_full[f'mse_{feat}'] = edge_mse_per_feat[:, i].cpu().numpy()
            
        edge_df_full['edge_anomaly_score'] = edge_mse.cpu().numpy()
        edge_df_full['source_node_score'] = node_mse[edge_df_full['source'].values].cpu().numpy()
        edge_df_full['target_node_score'] = node_mse[edge_df_full['target'].values].cpu().numpy()
        edge_df_full = edge_df_full.sort_values('edge_anomaly_score', ascending=False)

        # 3. Filter for Excel Export
        num_top_nodes = max(1, int(len(node_df_full) * node_top_percent))
        num_top_edges = max(1, int(len(edge_df_full) * edge_top_percent))
        
        node_df_excel = node_df_full.head(num_top_nodes)
        edge_df_excel = edge_df_full.head(num_top_edges).copy()

        # Convert list metadata to strings for Excel readability
        audit_cols = ['out_tx_ids', 'out_amounts', 'out_dates', 'in_tx_ids', 'in_amounts', 'in_dates']
        for col in audit_cols:
            if col in edge_df_excel.columns:
                edge_df_excel[col] = edge_df_excel[col].apply(lambda x: str(x))

        # 4. Write to Excel
        with pd.ExcelWriter(output_path) as writer:
            node_df_excel.to_excel(writer, sheet_name='Top Node Anomalies', index=False)
            edge_df_excel.to_excel(writer, sheet_name='Top Edge Anomalies', index=False)

            # Combined Raw Transactions for Top Node and Edge Anomalies
            top_node_ids = node_df_excel['customer_id'].tolist()
            top_edge_sources = edge_df_excel['source_id'].tolist()
            top_edge_targets = edge_df_excel['target_id'].tolist()
            
            all_involved_customer_ids = list(set(top_node_ids + top_edge_sources + top_edge_targets))
            
            merged_raw_tx = tx_df[tx_df[c_id_col].isin(all_involved_customer_ids)].drop_duplicates()
            merged_raw_tx.to_excel(writer, sheet_name='Top Anomaly Raw TX', index=False)

        print(f"Exported top {node_top_percent*100:.1f}% nodes and {edge_top_percent*100:.1f}% edges to {output_path}")
        return node_df_full, edge_df_full
