import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, GraphConv, HeteroConv, Linear
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple, Union

class BehaviorGraphBuilder:
    """
    Constructs a Behavioral Bipartite Graph for AML detection.
    Transforms raw transactions into a graph of Customers linked to Behavioral Nodes
    (Date, DayOfWeek, AmountBin, etc.) to capture hidden synchronization.
    """
    def __init__(self, config: Dict):
        """
        Args:
            config:
                Mappings:
                    'customer': col_name
                    'amount': col_name
                    'date': col_name
                    'bank': col_name
                    'direction': col_name
                Parameters:
                    'date_window': int (Default 0). Connects transaction to T +/- window days.
                                   Increases temporal fuzzy matching.
                    'amount_bins': list[float]. Custom bin edges for amount discretization.
                                   Controls strictness of amount similarity.
                    'customer_feature_cols': list[str] (Optional). Columns to use as node features.
                                             If None, uses simple Transaction Count.
        """
        self.config = config
        self.cust_map = {}
        self.node_maps = {} 
        
    def build(self, df: pd.DataFrame) -> HeteroData:
        data = HeteroData()
        
        # 1. Customer Nodes (Anchor)
        cust_col = self.config.get('customer')
        if not cust_col or cust_col not in df.columns:
            raise ValueError("Customer column is required.")
            
        unique_cust = df[cust_col].unique()
        self.cust_map = {id_: i for i, id_ in enumerate(unique_cust)}
        df['cust_idx'] = df[cust_col].map(self.cust_map)
        
        # Custom Customer Features
        feat_cols = self.config.get('customer_feature_cols')
        if feat_cols:
            cust_feats = df.groupby(cust_col)[feat_cols].mean()
            cust_feats = cust_feats.reindex(unique_cust).fillna(0)
            scaler = StandardScaler()
            cust_x = scaler.fit_transform(cust_feats.values)
            data['customer'].x = torch.from_numpy(cust_x).float()
        else:
            cust_feats = df.groupby(cust_col).size().to_frame('count')
            cust_feats = cust_feats.reindex(unique_cust).fillna(0)
            scaler = StandardScaler()
            cust_x = scaler.fit_transform(cust_feats.values)
            data['customer'].x = torch.from_numpy(cust_x).float()
        
        # Pre-calc weights if amount exists
        weights = None
        if 'amount' in self.config and self.config['amount'] in df.columns:
            # Transformation: Log scale to compress range
            amt = df[self.config['amount']]
            log_amt = np.log1p(amt)
            # Normalize to 0-1 range approx
            weights = torch.tensor(log_amt.values / log_amt.max(), dtype=torch.float)
        
        # 2. Behavioral Nodes
        
        # A. Bank Entity
        if 'bank' in self.config and self.config['bank'] in df.columns:
            self._add_simple_node_type(data, df, 'bank', self.config['bank'], 'uses_bank', weights)

        # B. Amount Binning
        if 'amount' in self.config and self.config['amount'] in df.columns:
            amt_col = self.config['amount']
            bins = self.config.get('amount_bins', [-1, 100, 1000, 5000, 9000, 10000, 100000, float('inf')])
            labels = [f"Bin_{i}" for i in range(len(bins)-1)]
            df['amt_bin'] = pd.cut(df[amt_col], bins=bins, labels=labels).astype(str)
            
            if 'direction' in self.config and self.config['direction'] in df.columns:
                df['amt_bin'] = df[self.config['direction']].astype(str) + "_" + df['amt_bin']
            
            self._add_simple_node_type(data, df, 'amount_bin', 'amt_bin', 'transacts_vol', weights)

        # C. Date Nodes (Synchronized Days)
        if 'date' in self.config and self.config['date'] in df.columns:
            date_col = self.config['date']
            df[date_col] = pd.to_datetime(df[date_col])
            
            # --- Absolute Date (With Sliding Window) ---
            df['abs_date'] = df[date_col].dt.date.astype(str)
            
            unique_dates = df['abs_date'].unique()
            date_map = {val: i for i, val in enumerate(unique_dates)}
            self.node_maps['date'] = date_map
            
            data['date'].x = torch.eye(len(unique_dates)) if len(unique_dates) < 100 else torch.randn(len(unique_dates), 16)
            
            window = self.config.get('date_window', 0)
            src_list, dst_list, w_list = [], [], []
            
            for offset in range(-window, window + 1):
                target_dates = (df[date_col] + pd.Timedelta(days=offset)).dt.date.astype(str)
                mapped_dst = target_dates.map(date_map)
                mask = mapped_dst.notna()
                
                valid_src = torch.tensor(df.loc[mask, 'cust_idx'].values, dtype=torch.long)
                valid_dst = torch.tensor(mapped_dst[mask].values.astype(int), dtype=torch.long)
                
                if weights is not None:
                    # Decay factor: 1.0 for exact day, 0.5 for neighbors
                    decay = 1.0 / (abs(offset) + 1.0)
                    valid_w = weights[torch.tensor(mask.values)] * decay
                    w_list.append(valid_w)
                
                src_list.append(valid_src)
                dst_list.append(valid_dst)
                
            final_src = torch.cat(src_list)
            final_dst = torch.cat(dst_list)
            final_w = torch.cat(w_list) if w_list else None
            
            data['customer', 'active_on_date', 'date'].edge_index = torch.stack([final_src, final_dst], dim=0)
            if final_w is not None:
                data['customer', 'active_on_date', 'date'].edge_weight = final_w
                
            data['date', 'rev_active_on_date', 'customer'].edge_index = torch.stack([final_dst, final_src], dim=0)
            if final_w is not None:
                data['date', 'rev_active_on_date', 'customer'].edge_weight = final_w
            
            # --- Cycle Nodes ---
            df['day_of_month'] = df[date_col].dt.day.astype(str)
            self._add_simple_node_type(data, df, 'day_of_month', 'day_of_month', 'active_on_day', weights)
            
            df['day_of_week'] = df[date_col].dt.day_name().astype(str)
            self._add_simple_node_type(data, df, 'day_of_week', 'day_of_week', 'active_on_weekday', weights)

        return data

    def _add_simple_node_type(self, data: HeteroData, df: pd.DataFrame, node_name: str, col_name: str, edge_name: str, weights: Optional[torch.Tensor] = None):
        unique_vals = df[col_name].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        self.node_maps[node_name] = mapping
        
        data[node_name].x = torch.eye(len(unique_vals)) if len(unique_vals) < 100 else torch.randn(len(unique_vals), 16)
        
        src = torch.tensor(df['cust_idx'].values, dtype=torch.long)
        dst = torch.tensor(df[col_name].map(mapping).values, dtype=torch.long)
        
        # Forward
        data['customer', edge_name, node_name].edge_index = torch.stack([src, dst], dim=0)
        if weights is not None:
            data['customer', edge_name, node_name].edge_weight = weights
            
        # Reverse
        data[node_name, f'rev_{edge_name}', 'customer'].edge_index = torch.stack([dst, src], dim=0)
        if weights is not None:
            data[node_name, f'rev_{edge_name}', 'customer'].edge_weight = weights


class BehavioralGNN(torch.nn.Module):
    """
    GNN Architecture that aggregates signals from multiple behavioral node types.
    Supports 'sage' (mapped to GraphConv for weighted bipartite support) and 'gat'.
    """
    def __init__(self, data_metadata, hidden_channels=64, out_channels=64, num_layers=2, architecture='sage'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lin = Linear(-1, out_channels)
        self.architecture = architecture.lower()
        
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in data_metadata[1]:
                if self.architecture == 'gat':
                    conv_dict[edge_type] = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=1, concat=False)
                else:
                    # GraphConv supports both bipartite and edge_weight
                    conv_dict[edge_type] = GraphConv((-1, -1), hidden_channels)
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        """
        Args:
            edge_weight_dict: Dictionary of edge weights for each edge type.
        """
        for conv in self.convs:
            # Only pass weight dicts if they are provided
            kwargs = {}
            if edge_weight_dict is not None:
                if self.architecture == 'gat':
                    kwargs['edge_attr_dict'] = edge_weight_dict
                else:
                    kwargs['edge_weight_dict'] = edge_weight_dict
            
            x_dict = conv(x_dict, edge_index_dict, **kwargs)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply projection head to Customer nodes
        x_dict['customer'] = self.lin(x_dict['customer'])
        return x_dict

def get_edge_weight_dict(data):
    edge_weight_dict = {}
    for edge_type in data.edge_types:
        if 'edge_weight' in data[edge_type]:
            edge_weight_dict[edge_type] = data[edge_type].edge_weight
    return edge_weight_dict if edge_weight_dict else None

def train_behavior_gnn(model, data, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    target_edge_types = [et for et in data.edge_types if et[0] == 'customer']
    edge_weight_dict = get_edge_weight_dict(data)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        x_dict = model(data.x_dict, data.edge_index_dict, edge_weight_dict)
        
        total_loss = 0
        
        for et in target_edge_types:
            src_type, _, dst_type = et
            src_idx, dst_idx = data[et].edge_index
            
            pos_score = (x_dict[src_type][src_idx] * x_dict[dst_type][dst_idx]).sum(dim=-1)
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            
            neg_dst_idx = torch.randint(0, data[dst_type].num_nodes, (len(src_idx),), device=src_idx.device)
            neg_score = (x_dict[src_type][src_idx] * x_dict[dst_type][neg_dst_idx]).sum(dim=-1)
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            
            total_loss += pos_loss + neg_loss
            
        total_loss.backward()
        optimizer.step()
        
    return model

def detect_behavioral_anomalies(model, data, cust_map):
    model.eval()
    edge_weight_dict = get_edge_weight_dict(data)
    
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict, edge_weight_dict)
        emb = x_dict['customer'].cpu().numpy()
        
    iso = IsolationForest(contamination=0.05, random_state=42)
    scores = -iso.fit_predict(emb)
    raw_scores = -iso.decision_function(emb)
    
    inv_map = {v: k for k, v in cust_map.items()}
    
    results = pd.DataFrame({
        'cust_idx': range(len(emb)),
        'risk_score': raw_scores,
        'is_anomaly': scores
    })
    results['customer_id'] = results['cust_idx'].map(inv_map)
    return results
