import numpy as np
from scipy.stats import norm

RANGE_STEP = 0.001
dist_range = None


def metric_pdf(data, pdf_params):
    """Calculates the normal distribution on the dataset."""
    global dist_range
    if dist_range is None:
        dist_range = np.arange(pdf_params.get("range_start"), pdf_params.get("range_end"), RANGE_STEP)
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.pdf(dist_range, mean, std_dev)


def metric_reduction(probability_distribution):
    """Does not implement any reduction."""
    return probability_distribution


def metric_distance(p, q):
    """Calculates KL-divergence."""
    return np.sum(np.where(p != 0, p * np.log(p/q), 0))
