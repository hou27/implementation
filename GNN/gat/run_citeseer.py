import time
from torch_geometric.datasets import Planetoid

#
# Citeseer specific constants
#

CITESEER_TRAIN_RANGE = [0, 120]
CITESEER_VAL_RANGE = [120, 120+500]
CITESEER_TEST_RANGE = [2308, 2308+1000]
CITESEER_NUM_INPUT_FEATURES = 3703
CITESEER_NUM_CLASSES = 6

from run import train_gat

if __name__ == '__main__':
    print('CiteSeer')
    dataset = Planetoid(root='./data/Citeseer', name='Citeseer')
    time_start = time.time()
    train_gat(time_start, dataset, CITESEER_TRAIN_RANGE, CITESEER_VAL_RANGE, CITESEER_TEST_RANGE, CITESEER_NUM_INPUT_FEATURES, CITESEER_NUM_CLASSES)
    print(f'Total training time: {(time.time() - time_start):.2f} [s]')