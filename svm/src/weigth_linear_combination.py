import numpy as np

def weight(b, n):
    total = np.sum(b)
    if np.isclose(total, 0.0):
        return np.ones_like(b, dtype=float) / len(b)

    x = np.divide(b, total)
    x = np.power(x, n)
    powered_total = np.sum(x)
    if np.isclose(powered_total, 0.0):
        return np.ones_like(b, dtype=float) / len(b)
    x = np.divide(x, powered_total)
    return x

