import numpy as np
from scipy.stats import norm


def metric_pdf(data, pdf_params):
    """Calculates the normal distribution on the dataset."""
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.pdf(data, mean, std_dev)


def metric_reduction(probability_distribution):
    """Does not implement any reduction."""
    return probability_distribution


def metric_distance(p, q):
    """Calculates KL-divergence."""
    return np.sum(np.where(p != 0, p * np.log(p/q), 0))
