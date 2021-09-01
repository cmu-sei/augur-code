import numpy as np
from scipy.stats import wasserstein_distance


def metric_distance(p, q):
    """Calculates total variation distance."""
    tvd = np.mean(np.abs(p - q)) / 2
    print(f"Total variation distance: {tvd}")
    return tvd

if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    tvd = metric_distance(p,q)
    print(f"Half the Wasserstein Distance: {wasserstein_distance(p,q) / 2}")
    print(f"Length of return is {tvd.size}")
    