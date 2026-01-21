import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear, to_hetero

class FraudGAT(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, metadata: tuple):
        super().__init__()
        
        # We need to transform node features of different types to the same hidden dimension first
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            # We assume we can lazily initialize or we know input dims. 
            # PyG LazyLinear or just Linear(-1, ...) is useful.
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # GAT Convolutions
        # Layer 1
        self.conv1 = HeteroConv({
            edge_type: GATConv(hidden_channels, hidden_channels, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        # Layer 2
        self.conv2 = HeteroConv({
            edge_type: GATConv(hidden_channels, out_channels, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # 1. Project raw features to hidden dimension
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x_dict_out[node_type] = self.lin_dict[node_type](x).relu()
        
        # 2. First GAT Layer
        x_dict_out = self.conv1(x_dict_out, edge_index_dict)
        x_dict_out = {key: x.relu() for key, x in x_dict_out.items()}

        # 3. Second GAT Layer
        x_dict_out = self.conv2(x_dict_out, edge_index_dict)
        
        # We don't apply activation on the final embedding usually, 
        # or we might normalize it.
        return x_dict_out
