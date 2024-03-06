import torch

from basic_gnn import BasicGNN

class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()

        self.conv1 = BasicGNN(num_node_features, 16)
        self.conv2 = BasicGNN(16, num_classes)

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)

        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return torch.nn.functional.softmax(x, dim=1)