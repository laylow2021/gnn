import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

# --- Compatibility Check ---
try:
    from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
    HAS_EXTENSIONS = True
except (ImportError, RuntimeError):
    HAS_EXTENSIONS = False
    print("[!] Warning: 'pyg-lib' or 'torch-sparse' not found. Loaders are disabled.")
    print("[*] Switching to Native PyTorch Full-Batch Mode.")

# ==========================================
# 1. Data Preparation Module
# ==========================================

def prepare_hetero_data(df, customer_id_col, bank_id_col, cust_feature_cols, bank_feature_cols, edge_feature_cols):
    data = HeteroData()
    
    unique_cust_ids = df[customer_id_col].unique()
    unique_bank_ids = df[bank_id_col].unique()
    
    cust_id_map = {id_: i for i, id_ in enumerate(unique_cust_ids)}
    bank_id_map = {id_: i for i, id_ in enumerate(unique_bank_ids)}
    
    df['cust_idx'] = df[customer_id_col].map(cust_id_map)
    df['bank_idx'] = df[bank_id_col].map(bank_id_map)
    
    print("Processing Customer Features...")
    cust_df = df.drop_duplicates(subset=[customer_id_col]).set_index(customer_id_col)
    cust_df = cust_df.reindex(unique_cust_ids)
    scaler_cust = StandardScaler()
    cust_x = cust_df[cust_feature_cols].fillna(0).values
    cust_x = scaler_cust.fit_transform(cust_x)
    data['customer'].x = torch.from_numpy(cust_x).float()
    
    print("Processing Bank Account Features...")
    bank_df = df.drop_duplicates(subset=[bank_id_col]).set_index(bank_id_col)
    bank_df = bank_df.reindex(unique_bank_ids)
    scaler_bank = StandardScaler()
    bank_x = bank_df[bank_feature_cols].fillna(0).values
    bank_x = scaler_bank.fit_transform(bank_x)
    data['bank_account'].x = torch.from_numpy(bank_x).float()

    print("Constructing Edges...")
    scaler_edge = StandardScaler()
    edge_attr = df[edge_feature_cols].fillna(0).values
    edge_attr = scaler_edge.fit_transform(edge_attr)
    edge_attr_tensor = torch.from_numpy(edge_attr).float()
    
    src_c = torch.tensor(df['cust_idx'].values, dtype=torch.long)
    dst_b = torch.tensor(df['bank_idx'].values, dtype=torch.long)
    
    data['customer', 'funds', 'bank_account'].edge_index = torch.stack([src_c, dst_b], dim=0)
    data['customer', 'funds', 'bank_account'].edge_attr = edge_attr_tensor
    data['bank_account', 'pays', 'customer'].edge_index = torch.stack([dst_b, src_c], dim=0)
    data['bank_account', 'pays', 'customer'].edge_attr = edge_attr_tensor
    
    return data, cust_id_map

# ==========================================
# 2. Unified GNN Architecture
# ==========================================

class UnifiedAMLGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2, architecture='sage'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if architecture == 'gat':
                def make_conv(): return GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=2, concat=False)
            else:
                def make_conv(): return SAGEConv((-1, -1), hidden_channels)
            
            self.convs.append(HeteroConv({
                ('customer', 'funds', 'bank_account'): make_conv(),
                ('bank_account', 'pays', 'customer'): make_conv(),
            }, aggr='sum'))

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
        return x_dict

# ==========================================
# 3. Training & Inference (Full Batch)
# ==========================================

def train_full_batch(model, data, epochs=10, lr=0.01):
    """
    Standard PyTorch training loop. No special libraries required.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_type = ('customer', 'funds', 'bank_account')
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        # Positive scores
        src, dst = data[edge_type].edge_index
        pos_out = (x_dict['customer'][src] * x_dict['bank_account'][dst]).sum(dim=-1)
        
        # Negative scores
        neg_dst = torch.randint(0, data['bank_account'].num_nodes, (src.size(0),), device=src.device)
        neg_out = (x_dict['customer'][src] * x_dict['bank_account'][neg_dst]).sum(dim=-1)
        
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out)) + \
               F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:03d}: Loss: {loss.item():.4f}")
    return model

def inference_full_batch(model, data, cust_mapping):
    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        emb = x_dict['customer'].cpu().numpy()
        
        print("Running Anomaly Detection (Isolation Forest)...")
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        scores = -iso_forest.fit_predict(emb)
        raw_scores = -iso_forest.decision_function(emb)
        
        results = pd.DataFrame({
            'cust_idx': range(len(emb)),
            'risk_score': raw_scores,
            'is_anomaly': scores
        })
        return results

# ==========================================
# 4. Main execution
# ==========================================

if __name__ == "__main__":
    print("--- AML GNN Unsupervised Pipeline (Native Mode) ---")
    
    # Mock Data
    num_tx = 1000
    df = pd.DataFrame({
        'customer_name': [f'Cust_{np.random.randint(0, 100)}' for _ in range(num_tx)],
        'bank_account_id': [f'Acct_{np.random.randint(0, 50)}' for _ in range(num_tx)],
        'amount': np.random.rand(num_tx) * 1000,
        'hour': np.random.randint(0, 24, num_tx),
        'score': np.random.rand(num_tx),
        'tenure': np.random.randint(0, 100, num_tx),
        'vol': np.random.rand(num_tx) * 1000,
        'flag': np.random.randint(0, 2, num_tx)
    })

    # Prepare
    data, cust_map = prepare_hetero_data(
        df, 'customer_name', 'bank_account_id',
        ['score', 'tenure'], ['vol', 'flag'], ['amount', 'hour']
    )
    
    # Initialize
    model = UnifiedAMLGNN(hidden_channels=32, out_channels=32)
    
    # Train (Always Full Batch to avoid library issues)
    model = train_full_batch(model, data, epochs=10)
    
    # Inference
    results = inference_full_batch(model, data, cust_map)
    
    # Map back
    inv_map = {v: k for k, v in cust_map.items()}
    results['customer_id'] = results['cust_idx'].map(inv_map)
    
    print("\n--- Top Anomalies ---")
    print(results.sort_values('risk_score', ascending=False).head())