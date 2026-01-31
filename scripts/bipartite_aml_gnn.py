import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

# ==========================================
# 1. Data Preparation Module
# ==========================================

def prepare_hetero_data(df, customer_id_col, bank_id_col, cust_feature_cols, bank_feature_cols, edge_feature_cols):
    """
    Prepares a PyG HeteroData object from a pandas DataFrame for a bipartite financial network.
    """
    data = HeteroData()
    
    # --- ID Mapping ---
    unique_cust_ids = df[customer_id_col].unique()
    unique_bank_ids = df[bank_id_col].unique()
    
    cust_id_map = {id_: i for i, id_ in enumerate(unique_cust_ids)}
    bank_id_map = {id_: i for i, id_ in enumerate(unique_bank_ids)}
    
    df['cust_idx'] = df[customer_id_col].map(cust_id_map)
    df['bank_idx'] = df[bank_id_col].map(bank_id_map)
    
    # --- Node Features ---
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

    # --- Edge Construction ---
    print("Constructing Edges...")
    
    scaler_edge = StandardScaler()
    edge_attr = df[edge_feature_cols].fillna(0).values
    edge_attr = scaler_edge.fit_transform(edge_attr)
    edge_attr_tensor = torch.from_numpy(edge_attr).float()
    
    src_c = torch.tensor(df['cust_idx'].values, dtype=torch.long)
    dst_b = torch.tensor(df['bank_idx'].values, dtype=torch.long)
    
    # 1. Customer -> funds -> Bank Account
    data['customer', 'funds', 'bank_account'].edge_index = torch.stack([src_c, dst_b], dim=0)
    data['customer', 'funds', 'bank_account'].edge_attr = edge_attr_tensor
    
    # 2. Bank Account -> pays -> Customer
    data['bank_account', 'pays', 'customer'].edge_index = torch.stack([dst_b, src_c], dim=0)
    data['bank_account', 'pays', 'customer'].edge_attr = edge_attr_tensor
    
    print(f"Graph constructed: {data}")
    return data, cust_id_map

# ==========================================
# 2. Unified GNN Architecture
# ==========================================

class UnifiedAMLGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2, architecture='sage'):
        super().__init__()
        self.num_layers = num_layers
        self.architecture = architecture
        
        self.convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = self._create_layer(hidden_channels, architecture)
            self.convs.append(conv)

    def _create_layer(self, hidden_channels, arch):
        if arch == 'sage':
            def make_conv(): return SAGEConv((-1, -1), hidden_channels)
        elif arch == 'gat':
            def make_conv(): return GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=2, concat=False)
        else: # rgcn or others map to sage for simplicity in hetero context
             def make_conv(): return SAGEConv((-1, -1), hidden_channels)

        return HeteroConv({
            ('customer', 'funds', 'bank_account'): make_conv(),
            ('bank_account', 'pays', 'customer'): make_conv(),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}
        
        return x_dict

# ==========================================
# 3. Training & Inference Workflow
# ==========================================

def train_unsupervised(model, data, epochs=10, lr=0.01):
    """
    Trains the GNN using negative sampling (Link Prediction) to learn structural embeddings.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We will use the 'customer -> funds -> bank' edge type for link prediction
    edge_type = ('customer', 'funds', 'bank_account')
    
    # Use LinkNeighborLoader for automatic negative sampling
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[10] * 2,
        batch_size=128,
        edge_label_index=(edge_type, data[edge_type].edge_index),
        neg_sampling_ratio=1.0, # 1 negative for every positive
        shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_examples = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # 1. Forward Pass to get Embeddings
            x_dict = model(batch.x_dict, batch.edge_index_dict)
            
            # 2. Get Score for Edges (both Pos and Neg)
            # LinkNeighborLoader appends negative samples to edge_label_index
            # and provides edge_label (1 for pos, 0 for neg)
            
            # The edge_label_index contains the indices of the edges to score
            # It includes both positive and negative edges if neg_sampling_ratio > 0
            src, dst = batch[edge_type].edge_label_index
            
            # Calculate dot product
            # Note: The loader returns a subgraph, so indices in edge_label_index 
            # are mapped to the local batch indices in x_dict
            out = (x_dict['customer'][src] * x_dict['bank_account'][dst]).sum(dim=-1)
            
            # 3. Loss
            target = batch[edge_type].edge_label
            loss = F.binary_cross_entropy_with_logits(out, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * src.size(0)
            total_examples += src.size(0)
            
        print(f"Epoch {epoch+1:03d}: Loss: {total_loss / total_examples:.4f}")

    return model

def inference_anomaly_detection(model, data):
    """
    1. Generates embeddings for all customers.
    2. Uses IsolationForest to detect anomalies in the embedding space.
    """
    model.eval()
    
    # Standard NeighborLoader for Node Embeddings
    loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        batch_size=128,
        input_nodes=('customer', None),
        shuffle=False
    )
    
    all_embeddings = []
    all_indices = []
    
    print("Generating Embeddings...")
    with torch.no_grad():
        for batch in loader:
            x_dict = model(batch.x_dict, batch.edge_index_dict)
            
            # Extract Customer Embeddings
            batch_size = batch['customer'].batch_size
            emb = x_dict['customer'][:batch_size].cpu().numpy()
            
            all_embeddings.append(emb)
            all_indices.extend(batch['customer'].n_id[:batch_size].cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # --- Anomaly Detection ---
    print("Running Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    # IsolationForest returns -1 for anomalies, 1 for normal
    # We invert this so higher score = more anomalous
    scores = -iso_forest.fit_predict(all_embeddings) 
    
    # We can also use decision_function for a continuous score (lower is more anomalous)
    # standardizing to 0-1 risk score would be ideal, but raw decision score works.
    # decision_function: average anomaly score. Lower is worse.
    raw_scores = -iso_forest.decision_function(all_embeddings)
            
    results = pd.DataFrame({
        'cust_idx': all_indices,
        'risk_score': raw_scores,
        'is_anomaly': scores # 1 = anomaly, -1 = normal (after our inversion of IF output)
    })
    return results

def train_full_batch(model, data, epochs=10, lr=0.01):
    """
    Fallback training loop that processes the entire graph at once.
    Useful when 'torch-sparse' or 'pyg-lib' are not available for efficient sampling.
    """
    print("\n--- Running Full-Batch Training (Fallback) ---")
    print("Warning: This requires enough memory to hold the entire graph/embeddings.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_type = ('customer', 'funds', 'bank_account')
    
    # Create negative edges for the entire graph upfront (simple random approximation)
    # In a real full-batch scenario, we might resample these every epoch or use a smaller subset
    # For simplicity/speed in fallback mode, we assume the graph fits in memory.
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Full Graph)
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        # 2. Score Positive Edges
        # Use all existing edges
        src, dst = data[edge_type].edge_index
        pos_out = (x_dict['customer'][src] * x_dict['bank_account'][dst]).sum(dim=-1)
        
        # 3. Score Negative Edges
        # Randomly sample negatives (same amount as positives)
        neg_dst = torch.randint(0, data['bank_account'].num_nodes, (src.size(0),), device=src.device)
        neg_out = (x_dict['customer'][src] * x_dict['bank_account'][neg_dst]).sum(dim=-1)
        
        # 4. Loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:03d}: Loss: {loss.item():.4f}")
        
    return model

# ==========================================
# 4. User Configuration & Execution
# ==========================================

if __name__ == "__main__":
    print("--- AML GNN Unsupervised Pipeline ---")
    
    # --- MOCK DATA GENERATION ---
    print("Generating Mock Data...")
    num_tx = 1000
    mock_data = {
        'transaction_id': range(num_tx),
        'customer_name': [f'Cust_{np.random.randint(0, 100)}' for _ in range(num_tx)],
        'bank_account_id': [f'Acct_{np.random.randint(0, 50)}' for _ in range(num_tx)],
        'amount': np.random.rand(num_tx) * 10000,
        'hour_of_day': np.random.randint(0, 24, num_tx),
        'cust_risk_score': np.random.rand(num_tx),
        'cust_tenure': np.random.randint(1, 365, num_tx),
        'bank_volume': np.random.rand(num_tx) * 100000,
        'bank_flags': np.random.randint(0, 2, num_tx),
    }
    df = pd.DataFrame(mock_data)

    # ---------------------------------------------------------
    # USER CONFIGURATION BLOCK
    # ---------------------------------------------------------
    # df = pd.read_csv('my_data.csv')
    
    CUSTOMER_ID = 'customer_name'
    BANK_ID = 'bank_account_id'
    
    CUSTOMER_FEATURES = ['cust_risk_score', 'cust_tenure']
    BANK_FEATURES = ['bank_volume', 'bank_flags']
    EDGE_FEATURES = ['amount', 'hour_of_day']
    
    MODEL_CHOICE = 'sage' 
    # ---------------------------------------------------------
    
    # 1. Prepare Data
    print("Preparing Graph Data...")
    hetero_data, cust_mapping = prepare_hetero_data(
        df, CUSTOMER_ID, BANK_ID, 
        CUSTOMER_FEATURES, BANK_FEATURES, EDGE_FEATURES
    )
    
    # 2. Initialize Model
    print(f"Initializing {MODEL_CHOICE.upper()} Model...")
    model = UnifiedAMLGNN(
        hidden_channels=64, 
        out_channels=64, # Output embedding dimension
        num_layers=2, 
        architecture=MODEL_CHOICE
    )
    
    # 3. Train
    print("Starting Unsupervised Training...")
    try:
        # Try using the efficient loader first
        model = train_unsupervised(model, hetero_data, epochs=5, lr=0.01)
    except (ImportError, OSError, RuntimeError) as e:
        print(f"\n[!] Loader-based training failed: {e}")
        print("[*] Switching to Full-Batch training (Native PyTorch mode).")
        model = train_full_batch(model, hetero_data, epochs=5, lr=0.01)
    
    # 4. Inference (Anomaly Detection)
    print("Running Anomaly Detection...")
    # Inference also normally uses loader, but for full-batch fallback we can pass full graph
    # We'll try loader first, then fallback to direct forward pass
    try:
        risk_scores = inference_anomaly_detection(model, hetero_data)
    except:
         print("[*] Using Full-Batch Inference")
         model.eval()
         with torch.no_grad():
             x_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
             emb = x_dict['customer'].cpu().numpy()
             
             iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
             scores = -iso_forest.fit_predict(emb)
             raw_scores = -iso_forest.decision_function(emb)
             
             inv_cust_map = {v: k for k, v in cust_mapping.items()}
             risk_scores = pd.DataFrame({
                'cust_idx': list(inv_cust_map.keys()), # Assuming mapped 0..N
                'risk_score': raw_scores,
                'is_anomaly': scores 
             })
             # We need to ensure indices match the order. 
             # In full batch, x_dict['customer'] is ordered 0..N
             risk_scores['cust_idx'] = range(len(risk_scores))

    # Map back to IDs
    inv_cust_map = {v: k for k, v in cust_mapping.items()}
    risk_scores['customer_id'] = risk_scores['cust_idx'].map(inv_cust_map)
    
    print("\n--- Top Anomalies Detected ---")
    print(risk_scores.sort_values('risk_score', ascending=False).head())
