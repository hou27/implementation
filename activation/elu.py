import numpy as np

def elu(x):
    return np.where(x>0, x, np.exp(x) - 1)