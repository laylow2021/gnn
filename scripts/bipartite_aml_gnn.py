import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv, RGCNConv, HeteroConv, Linear
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==========================================
# 1. Data Preparation Module
# ==========================================

def prepare_hetero_data(df, customer_id_col, bank_id_col, cust_feature_cols, bank_feature_cols, edge_feature_cols, label_col=None):
    """
    Prepares a PyG HeteroData object from a pandas DataFrame for a bipartite financial network.
    """
    data = HeteroData()
    
    # --- ID Mapping ---
    # Map string IDs to integer indices
    # We use a global mapping strategy: unique IDs for customers and unique IDs for banks
    unique_cust_ids = df[customer_id_col].unique()
    unique_bank_ids = df[bank_id_col].unique()
    
    cust_id_map = {id_: i for i, id_ in enumerate(unique_cust_ids)}
    bank_id_map = {id_: i for i, id_ in enumerate(unique_bank_ids)}
    
    # Add mapped columns to df for easy edge creation
    df['cust_idx'] = df[customer_id_col].map(cust_id_map)
    df['bank_idx'] = df[bank_id_col].map(bank_id_map)
    
    # --- Node Features ---
    # 1. Customer Features
    # We need to aggregate features if there are multiple transactions per customer, 
    # OR we assume the input df has one row per transaction and we need a separate customer feature DF.
    # The prompt implies 'df' contains everything. 
    # Usually, node features are static per node. 
    # For this script, we will take the FIRST occurrence of features for each node to build the node feature matrix.
    # In a real pipeline, you would perform aggregation (mean/max) preprocessing before this.
    
    print("Processing Customer Features...")
    cust_df = df.drop_duplicates(subset=[customer_id_col]).set_index(customer_id_col)
    cust_df = cust_df.reindex(unique_cust_ids) # Ensure order matches indices
    
    scaler_cust = StandardScaler()
    cust_x = cust_df[cust_feature_cols].fillna(0).values
    cust_x = scaler_cust.fit_transform(cust_x)
    data['customer'].x = torch.from_numpy(cust_x).float()
    
    # 2. Bank Account Features
    print("Processing Bank Account Features...")
    bank_df = df.drop_duplicates(subset=[bank_id_col]).set_index(bank_id_col)
    bank_df = bank_df.reindex(unique_bank_ids)
    
    scaler_bank = StandardScaler()
    bank_x = bank_df[bank_feature_cols].fillna(0).values
    bank_x = scaler_bank.fit_transform(bank_x)
    data['bank_account'].x = torch.from_numpy(bank_x).float()
    
    # --- Labels ---
    if label_col and label_col in df.columns:
        # Assumes labels are at customer level. 
        # If a customer is flagged in ANY transaction, we consider them illicit (or take the label from the cust_df)
        # Using the value from the unique customer dataframe
        if label_col in cust_df.columns:
            cust_y = cust_df[label_col].values
            data['customer'].y = torch.from_numpy(cust_y).float()
        else:
            print("Warning: Label column not found in customer-deduplicated data.")

    # --- Edge Construction ---
    print("Constructing Edges...")
    
    # Edge Features
    scaler_edge = StandardScaler()
    edge_attr = df[edge_feature_cols].fillna(0).values
    edge_attr = scaler_edge.fit_transform(edge_attr)
    edge_attr_tensor = torch.from_numpy(edge_attr).float()
    
    # 1. Customer -> funds -> Bank Account
    src_c = torch.tensor(df['cust_idx'].values, dtype=torch.long)
    dst_b = torch.tensor(df['bank_idx'].values, dtype=torch.long)
    
    data['customer', 'funds', 'bank_account'].edge_index = torch.stack([src_c, dst_b], dim=0)
    data['customer', 'funds', 'bank_account'].edge_attr = edge_attr_tensor
    
    # 2. Bank Account -> pays -> Customer
    # Creating reverse edges to allow information flow back to customers
    # We use the same transactions but reversed direction for the graph structure
    src_b = dst_b
    dst_c = src_c
    
    data['bank_account', 'pays', 'customer'].edge_index = torch.stack([src_b, dst_c], dim=0)
    data['bank_account', 'pays', 'customer'].edge_attr = edge_attr_tensor # Same features, flows back
    
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
        self.lin = Linear(-1, out_channels)
        
        for _ in range(num_layers):
            conv = self._create_layer(hidden_channels, architecture)
            self.convs.append(conv)

    def _create_layer(self, hidden_channels, arch):
        """
        Creates a HeteroConv layer based on the selected architecture.
        Uses (-1, -1) for lazy initialization of input channels.
        """
        if arch == 'sage':
            # SAGEConv aggregates neighbor features (mean/max)
            def make_conv(): return SAGEConv((-1, -1), hidden_channels)
        elif arch == 'gat':
            # GATConv uses attention mechanisms
            def make_conv(): return GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=2, concat=False)
        elif arch == 'rgcn':
            # RGCN distinguishes edge types. 
            # In PyG HeteroConv, we can simulate RGCN behavior by using standard GCN/SAGE 
            # but having separate weights per relation (which HeteroConv does by default).
            # True RGCNConv is for homogeneous graphs with edge_type tensor. 
            # Here we use SAGE as the base for the HeteroConv relations to act as "Relational" convolution.
             def make_conv(): return SAGEConv((-1, -1), hidden_channels)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        return HeteroConv({
            ('customer', 'funds', 'bank_account'): make_conv(),
            ('bank_account', 'pays', 'customer'): make_conv(),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # x_dict: Dictionary of node features
        # edge_index_dict: Dictionary of edge indices
        
        for i, conv in enumerate(self.convs):
            # Pass edge_attr if the specific conv supports it (SAGE/GCN usually don't take edge features easily 
            # without modification, but GAT can. For simplicity in this unified class, we rely on node aggregation.
            # Extending to edge features requires custom MessagePassing classes.)
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply ReLU and Dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}
        
        # We only care about Customer classifications
        return self.lin(x_dict['customer'])

# ==========================================
# 3. Training & Inference Workflow
# ==========================================

def train(model, data, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # NeighborLoader for scalability
    # We define input nodes as ('customer', mask) if we had a train_mask, 
    # here we assume all labeled customers are for training for simplicity or passed indices.
    # We'll use all customers that have a label (which should be all in 'data' if prepared correctly with labels).
    
    # Create a mask for valid training nodes (nodes with labels)
    # Check if 'y' exists
    if not hasattr(data['customer'], 'y'):
        raise ValueError("Data object has no labels ('y') for training.")
        
    labeled_indices = torch.arange(data['customer'].num_nodes)
    
    loader = NeighborLoader(
        data,
        # Sample 10 neighbors for each node for 2 hops
        num_neighbors=[10] * 2,
        # Use a batch size of 128 for training nodes
        batch_size=128,
        input_nodes=('customer', labeled_indices),
        shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_examples = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # Helper to match output size with batch labels
            # The batch size of the seed nodes is defined in batch['customer'].batch_size
            batch_size = batch['customer'].batch_size
            out = out[:batch_size].squeeze()
            target = batch['customer'].y[:batch_size]
            
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            
        print(f"Epoch {epoch+1:03d}: Loss: {total_loss / total_examples:.4f}")

    return model

def inference(model, data):
    model.eval()
    # For inference, we can process full graph if it fits in memory, 
    # or use NeighborLoader again. For 'risk score' output, we want it for ALL customers.
    
    loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2, # Same neighbor sampling
        batch_size=128,
        input_nodes=('customer', None), # None means all nodes of this type
        shuffle=False
    )
    
    all_preds = []
    all_indices = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['customer'].batch_size
            
            scores = torch.sigmoid(out[:batch_size]).squeeze().cpu().numpy()
            
            # If batch_size is 1, scores is scalar 
            if np.ndim(scores) == 0:
                scores = [scores]
                
            all_preds.extend(scores)
            # Original node indices are stored in batch.n_id or batch['customer'].n_id
            # However, NeighborLoader maps global IDs to local batch IDs.
            # PyG's NeighborLoader preserves original indices in `n_id`
            all_indices.extend(batch['customer'].n_id[:batch_size].cpu().numpy())
            
    # Combine into a result dict or df
    # Map back using original indices
    results = pd.DataFrame({
        'cust_idx': all_indices,
        'risk_score': all_preds
    })
    return results

# ==========================================
# 4. User Configuration & Execution
# ==========================================

if __name__ == "__main__":
    print("--- AML GNN Detection Pipeline ---")
    
    # --- MOCK DATA GENERATION (for standalone execution) ---
    # In a real scenario, the user would uncomment the pd.read_csv line below.
    print("Generating Mock Data...")
    num_tx = 1000
    mock_data = {
        'transaction_id': range(num_tx),
        'customer_name': [f'Cust_{np.random.randint(0, 100)}' for _ in range(num_tx)],
        'bank_account_id': [f'Acct_{np.random.randint(0, 50)}' for _ in range(num_tx)],
        'amount': np.random.rand(num_tx) * 10000,
        'hour_of_day': np.random.randint(0, 24, num_tx),
        # Customer Features (mocked as repeating columns for simplicity)
        'cust_risk_score': np.random.rand(num_tx),
        'cust_tenure': np.random.randint(1, 365, num_tx),
        # Bank Features
        'bank_volume': np.random.rand(num_tx) * 100000,
        'bank_flags': np.random.randint(0, 2, num_tx),
        # Labels (attached to customers, mocked here per transaction but consistent per customer logic needed)
        'is_laundering': np.random.choice([0, 1], num_tx, p=[0.95, 0.05])
    }
    df = pd.DataFrame(mock_data)
    
    # Ensure consistency of labels per customer for the mock
    cust_labels = df.groupby('customer_name')['is_laundering'].max()
    df['is_laundering'] = df['customer_name'].map(cust_labels)

    # ---------------------------------------------------------
    # USER CONFIGURATION BLOCK
    # ---------------------------------------------------------
    # df = pd.read_csv('my_data.csv')  # <-- User loads data here
    
    CUSTOMER_ID = 'customer_name'
    BANK_ID = 'bank_account_id'
    
    CUSTOMER_FEATURES = ['cust_risk_score', 'cust_tenure']
    BANK_FEATURES = ['bank_volume', 'bank_flags']
    EDGE_FEATURES = ['amount', 'hour_of_day']
    LABEL_COL = 'is_laundering'
    
    MODEL_CHOICE = 'sage' # Options: 'sage', 'gat', 'rgcn'
    # ---------------------------------------------------------
    
    # 1. Prepare Data
    print("Preparing Graph Data...")
    hetero_data, cust_mapping = prepare_hetero_data(
        df, CUSTOMER_ID, BANK_ID, 
        CUSTOMER_FEATURES, BANK_FEATURES, EDGE_FEATURES, 
        LABEL_COL
    )
    
    # 2. Initialize Model
    print(f"Initializing {MODEL_CHOICE.upper()} Model...")
    model = UnifiedAMLGNN(
        hidden_channels=64, 
        out_channels=1, 
        num_layers=2, 
        architecture=MODEL_CHOICE
    )
    
    # 3. Train
    print("Starting Training...")
    model = train(model, hetero_data, epochs=5, lr=0.01)
    
    # 4. Inference
    print("Running Inference...")
    risk_scores = inference(model, hetero_data)
    
    # Map back to IDs
    inv_cust_map = {v: k for k, v in cust_mapping.items()}
    risk_scores['customer_id'] = risk_scores['cust_idx'].map(inv_cust_map)
    
    print("\n--- Risk Scoring Results (Top 5 High Risk) ---")
    print(risk_scores.sort_values('risk_score', ascending=False).head())
