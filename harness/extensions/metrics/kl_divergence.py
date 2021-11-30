import numpy as np
from scipy.stats import entropy


def metric_distance(p, q):
    """Calculates KL-divergence."""
    kl_div = np.sum(np.where(p != 0, p * np.log2(p/q), 0))
    #print(f"KL div: {kl_div}")
    #print(f"Entropy: {entropy(p, q)}")
    return kl_div
