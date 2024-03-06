import torch
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()

        # GAT layer 1
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=8, heads=8)
        # GAT layer 2 (classification layer)
        self.conv2 = GATConv(in_channels=8*8, out_channels=num_classes, heads=1, concat=False)

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT layer 1
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)

        # GAT layer 2
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return torch.nn.functional.softmax(x, dim=1)