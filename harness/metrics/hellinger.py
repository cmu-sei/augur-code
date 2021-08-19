import numpy as np


RANGE_STEP = 0.001
dist_range = None

def metric_pdf(data, pdf_params):
    """Calculates the normal distribution on the dataset."""
    global dist_range
    if dist_range is None:
        dist_range = np.arange(pdf_params.get("range_start"), pdf_params.get("range_end"), RANGE_STEP)
    mean = np.mean(data)
    std_dev = np.std(data)
    print(f"Mean: {mean}, Std Dev: {std_dev}")
    return norm.pdf(dist_range, mean, std_dev)


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
