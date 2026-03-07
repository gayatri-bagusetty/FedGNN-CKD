import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim) 
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)  # 0.5→0.2
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)  # NO RESIDUAL
        x = F.dropout(x, p=0.2, training=self.training)  # 0.5→0.2
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Layer 4
        x = self.conv4(x, edge_index)
        return x
