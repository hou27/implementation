import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid 미분
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
