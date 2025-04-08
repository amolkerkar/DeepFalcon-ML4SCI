# EdgeConv model using MLP
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

class JetEdgeConvNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.edge_conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.edge_conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.edge_conv1(x, edge_index)
        x = self.edge_conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)