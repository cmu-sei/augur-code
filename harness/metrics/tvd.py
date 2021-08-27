import numpy as np


def metric_distance(p, q):
    """Calculates total variation distance."""
    tvd = np.sum(np.abs(p - q)) / 2
    print(f"Total variation distance: {tvd}")
    return tvd

if __name__ == "__main__":
    n = 100
    p = np.random.random(n)
    q = np.random.random(n)
    tvd = metric_distance(p,q)
    print(f"Length of return is {tvd.size}")
    