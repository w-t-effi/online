import numpy as np


def normalize(x):
    std = np.std(x, axis=0) + np.full(np.std(x, axis=0).shape, 10e-99)
    x /= std
    return x
