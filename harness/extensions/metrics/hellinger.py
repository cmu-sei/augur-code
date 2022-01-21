import numpy as np


def metric_distance(p, q):
    """Calculates Hellinger distance."""
    hellinger_dist = np.sqrt(np.sum(np.where(p != 0, (np.sqrt(p) - np.sqrt(q)) ** 2, 0))) / np.sqrt(2)
    #print(f"Hellinger Distance: {hellinger_dist}")
    return hellinger_dist


if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    hellinger_dist = metric_distance(p,q)
    print(f"Length of return is {hellinger_dist.size}")
    