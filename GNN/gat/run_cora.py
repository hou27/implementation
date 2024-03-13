import time
from torch_geometric.datasets import Planetoid

#
# Cora specific constants
#

# Thomas Kipf et al. first used this split in GCN paper and later Petar Veličković et al. in GAT paper
CORA_TRAIN_RANGE = [0, 140]  # we're using the first 140 nodes as the training nodes
CORA_VAL_RANGE = [140, 140+500]
CORA_TEST_RANGE = [1708, 1708+1000]
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

from run import train_gat

if __name__ == '__main__':
    print('Cora')
    dataset = Planetoid(root='./data/Cora', name='Cora')
    time_start = time.time()
    train_gat(time_start, dataset, CORA_TRAIN_RANGE, CORA_VAL_RANGE, CORA_TEST_RANGE, CORA_NUM_INPUT_FEATURES, CORA_NUM_CLASSES)
    print(f'Total training time: {(time.time() - time_start):.2f} [s]')