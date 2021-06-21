import numpy
from scipy.stats import norm


def metric_pdf(dataset, pdf_params):
    """Calculates the normal distribution on the dataset."""
    return norm.pdf(dataset.get_output(), float(pdf_params("mean")), float(pdf_params("std_dev")))


def metric_reduction(probability_distribution):
    """Does not imeplement any reduction."""
    return probability_distribution


def metric_distance(p, q):
    """Calculates KL-dirvergence."""
    return sum(p[i] * numpy.log2(p[i]/q[i]) for i in range(len(p)))
