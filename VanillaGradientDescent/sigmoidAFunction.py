import numpy as np


def sigmoidActivation(x):
    return 1.0 / (1 + np.exp(-x))
