import numpy as np
from scipy.stats import wasserstein_distance

def metric_distance(p, q):
    """Calculates Wasserstein distance."""
    wass = np.mean(np.abs(p - q))
    print(f"Wasserstein distance: {wass}")
    return wass

if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    wass = metric_distance(p,q)
    print(f"From numpy: {wasserstein_distance(p,q)}")
    print(f"Length of return is {wass.size}")
    