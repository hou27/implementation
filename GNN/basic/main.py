import torch
from torch_geometric.datasets import Planetoid

from basic_gnn import BasicGNN
from run import train, test

# Cora Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

data = dataset[0]
model = BasicGNN(dataset.num_node_features, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(100):
    train(model, data, optimizer)
    if epoch % 10 == 0:
        acc = test(model, data)
        print('Epoch: {:03d}, Accuracy: {:.4f}'.format(epoch, acc))
print('Epoch: {:03d}, Accuracy: {:.4f}'.format(epoch, acc))