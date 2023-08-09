import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# tanh 미분
def d_tanh(x):
    return 1 - tanh(x) ** 2