"""
Reference: GRAPH ATTENTION NETWORKS (2018).

https://github.com/PetarV-/GAT
https://github.com/gordicaleksa/pytorch-GAT
"""

import torch.nn as nn

from layer import GATLayer

class GAT(nn.Module):
    def __init__(self, num_in_features, num_classes, add_skip_connection=True, bias=True, dropout=0.6):
        super().__init__()

        self.gat_net = nn.Sequential(
            GATLayer(
                num_in_features=num_in_features,  # consequence of concatenation
                num_out_features=8,
                num_of_heads=8,
                concat=True,
                activation=nn.ELU(),
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias
            ),
            GATLayer(
                num_in_features=8 * 8,  # consequence of concatenation
                num_out_features=num_classes,
                num_of_heads=1,
                concat=False,  # last GAT layer does mean avg, the others do concat
                activation=None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias
            )
        )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)