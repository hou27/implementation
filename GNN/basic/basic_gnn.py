import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
import torch_scatter

class BasicGNN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.self_weight = Parameter(torch.FloatTensor(input_dims, output_dims))
        self.neighbor_weight = Parameter(torch.FloatTensor(input_dims, output_dims))
        self.bias = Parameter(torch.FloatTensor(output_dims))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_weight)
        nn.init.xavier_uniform_(self.neighbor_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        adj = self.symmetric_normalization(edge_index, x.size(0))

        self_support = torch.mm(x, self.self_weight)
        neighbor_support = torch.spmm(adj, torch.mm(x, self.neighbor_weight))
        output = self_support + neighbor_support + self.bias
        # output = torch.relu(output)
        return output

    def symmetric_normalization(self, edge_index, num_nodes):
        row, col = edge_index

        # Make D
        deg = torch_scatter.scatter_add(torch.ones_like(row), row, dim_size=num_nodes).pow(-0.5)

        # D^(-0.5)AD^(-0.5)
        deg_row = deg[row]
        deg_col = deg[col]
        norm = deg_row * deg_col
        norm_adj = torch.sparse.FloatTensor(edge_index, norm, torch.Size([num_nodes, num_nodes]))

        return norm_adj
