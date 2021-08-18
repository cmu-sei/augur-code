import numpy as np


def metric_reduction(probability_distribution):
    """Does not implement any reduction."""
    return probability_distribution


def metric_distance(p, q):
    """Calculates Hellinger distance."""
    hellinger_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    print(f"Hellinger Distance: {hellinger_dist}")
    return hellinger_dist


if __name__ == "__main__":
    p1 = 1
    q1 = 1
    d1 = metric_distance(p1, q1)
    print(f"{d1} equals {0}")

    p2 = np.random.random(100)
    q2 = np.random.random(100)
    d2 = metric_distance(p2, q2)
    print(f"{d2} is close to {2}")
