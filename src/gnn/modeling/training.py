import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from src.gnn.config.schema import Schema
from src.gnn.modeling.models import FraudGAT

def negative_sampling(edge_index, num_nodes_src, num_nodes_dst):
    # Randomly sample negative edges
    # For a real implementation, use torch_geometric.utils.structured_negative_sampling 
    # or batched negative sampling. Here we do a simple random permutation.
    
    num_neg_samples = edge_index.size(1)
    
    # Random source nodes
    neg_src = torch.randint(0, num_nodes_src, (num_neg_samples,), dtype=torch.long)
    # Random dest nodes
    neg_dst = torch.randint(0, num_nodes_dst, (num_neg_samples,), dtype=torch.long)
    
    return torch.stack([neg_src, neg_dst], dim=0)

def train_model(data: HeteroData, hidden_channels=64, epochs=10, lr=0.01):
    """
    Trains the FraudGAT model using link prediction (self-supervised).
    """
    metadata = data.metadata()
    model = FraudGAT(hidden_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Target Edge for Link Prediction: Customer -> Owns -> Account
    target_edge_type = (Schema.NODE_CUSTOMER, Schema.EDGE_OWNS, Schema.NODE_ACCOUNT)
    
    # Check if edge exists
    if target_edge_type not in data.edge_index_dict:
        print("Warning: Target edge type for training not found.")
        return model, {}

    edge_index = data[target_edge_type].edge_index
    num_cust = data[Schema.NODE_CUSTOMER].num_nodes
    num_acc = data[Schema.NODE_ACCOUNT].num_nodes

    model.train()
    
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass
        z_dict = model(data.x_dict, data.edge_index_dict)
        
        # We want to maximize similarity of connected (customer, account) pairs
        # and minimize similarity of random pairs.
        
        # Positive Edges
        # Get embeddings for source (customer) and dest (account)
        z_cust = z_dict[Schema.NODE_CUSTOMER]
        z_acc = z_dict[Schema.NODE_ACCOUNT]
        
        # Calculate score for positive edges (dot product)
        # z_cust[edge_index[0]] shape: [num_edges, hidden]
        pos_score = (z_cust[edge_index[0]] * z_acc[edge_index[1]]).sum(dim=1)
        
        # Negative Edges
        neg_edge_index = negative_sampling(edge_index, num_cust, num_acc)
        neg_score = (z_cust[neg_edge_index[0]] * z_acc[neg_edge_index[1]]).sum(dim=1)
        
        # Loss: Binary Cross Entropy with Logits
        # Pos label: 1, Neg label: 0
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    return model, z_dict
