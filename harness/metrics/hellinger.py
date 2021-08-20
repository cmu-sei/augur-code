import numpy as np


def metric_distance(p, q):
    """Calculates Hellinger distance."""
    hellinger_dist = np.sqrt(np.where(p != 0, np.sum((np.sqrt(p) - np.sqrt(q)) ** 2), 0)) / np.sqrt(2)
    print(f"Hellinger Distance: {hellinger_dist}")
    return hellinger_dist
