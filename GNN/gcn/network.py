import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()

        # GCN layer 1
        self.conv1 = GCNConv(in_channels=num_node_features, out_channels=64)
        # GCN layer 2 (classification layer)
        self.conv2 = GCNConv(in_channels=64, out_channels=num_classes)

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layer 1
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)

        # GCN layer 2
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return torch.nn.functional.softmax(x, dim=1)
