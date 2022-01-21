import numpy as np
from scipy.stats import energy_distance


def metric_distance(p, q):
    """Calculates Energy distance."""
    energy_dist = energy_distance(p,q)
    #print(f"Energy distance: {energy_dist}")
    return energy_dist


if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    energy_dist = metric_distance(p,q)
    print(f"Length of return is {np.array(energy_dist).size}")
    