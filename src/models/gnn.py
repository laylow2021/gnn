import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Dict, Any

class AMLGraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels: int, edge_in_channels: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        hc = config['model']['hidden_channels']
        oc = config['model']['out_channels']
        
        # Encoder: GATv2Conv layers
        self.conv1 = GATv2Conv(in_channels, hc, edge_dim=edge_in_channels)
        self.conv2 = GATv2Conv(hc, oc, edge_dim=edge_in_channels)
        
        # Node Decoder
        self.node_decoder = torch.nn.Sequential(
            torch.nn.Linear(oc, hc),
            torch.nn.ReLU(),
            torch.nn.Linear(hc, in_channels)
        )
        
        # Edge Decoder (Inferred features)
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(oc * 2, hc),
            torch.nn.ReLU(),
            torch.nn.Linear(hc, edge_in_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        """Encodes node/edge data and reconstructs both."""
        # 1. Encoding
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr)
        
        # 2. Node Reconstruction
        x_recon = self.node_decoder(h)
        
        # 3. Edge Reconstruction (Inner-product based MLP)
        edge_src, edge_dst = edge_index
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)
        edge_attr_recon = self.edge_decoder(edge_features)
        
        return h, x_recon, edge_attr_recon

def compute_combined_loss(x, x_recon, edge_attr, edge_attr_recon, config: Dict[str, Any]):
    """Calculate $Loss_{Total} = 0.7(\\text{Node MSE}) + 0.3(\\text{Edge Attr MSE})$."""
    node_mse = F.mse_loss(x, x_recon)
    edge_mse = F.mse_loss(edge_attr, edge_attr_recon)
    
    total_loss = (config['model']['node_loss_weight'] * node_mse + 
                  config['model']['edge_loss_weight'] * edge_mse)
    
    return total_loss, node_mse, edge_mse
