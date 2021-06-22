import numpy
from scipy.stats import norm


def metric_pdf(data, pdf_params):
    """Calculates the normal distribution on the dataset."""
    return norm.pdf(data, float(pdf_params.get("mean")), float(pdf_params.get("std_dev")))


def metric_reduction(probability_distribution):
    """Does not implement any reduction."""
    return probability_distribution


def metric_distance(p, q):
    """Calculates KL-divergence."""
    return sum(p[i] * numpy.log2(p[i]/q[i]) for i in range(len(p)))
