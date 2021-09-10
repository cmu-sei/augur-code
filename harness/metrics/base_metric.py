import importlib

import numpy as np
from scipy.stats import norm, gaussian_kde

from utils.logging import print_and_log


def load_metric_module(module_name):
    """Loads a metric module given the name"""
    return importlib.import_module("metrics." + module_name)


def create_metric(metric_info):
    """Factory method."""
    metric_type = metric_info.get("type")
    if metric_type == "DistanceMetric":
        metric = DistanceMetric()
    elif metric_type == "ErrorBased":
        metric = ErrorBased()
    else:
        raise Exception(f"Non-supported metric type: {metric_type}")
    return metric


class Metric:
    """Generic Metric class, will load specific metric module as required."""
    metric_module = None
    metric_params = None

    def load_metric_functions(self, config):
        """Loads a metric module from config."""
        self.metric_module = load_metric_module(config.get("module"))
        self.metric_params = config.get("params")

    def check_module_loaded(self):
        """Ensures a metrics module is loaded."""
        if self.metric_module is not None:
            return True
        else:
            raise Exception("No metric module configured!")

    def initial_setup(self, dataset, predictions):
        """Method to be called once before starting to work with this metric."""
        raise NotImplementedError()

    def step_setup(self, timebox):
        """Method to be called once in each step/iteration."""
        raise NotImplementedError()

    def calculate_metric(self):
        """Generic method to calculate the value of this metric. Should be overriden by submetrics."""
        raise NotImplementedError()


class ErrorBased(Metric):
    """Implements an error-based metric that calculates error based on output."""
    timebox = None
    dataset = None
    predictions = None

    def initial_setup(self, dataset, predictions):
        """Overriden."""
        self.dataset = dataset
        self.predictions = predictions

    def step_setup(self, timebox):
        """Overriden."""
        self.timebox = timebox

    """Implements an error-based metric."""
    def _calculate_error(self):
        if self.check_module_loaded():
            return self.metric_module.metric_error(self.timebox, self.dataset, self.predictions)

    def calculate_metric(self):
        """Overriden."""
        return self._calculate_error()


class DensityEstimator:
    """Implements most common density functions."""

    # Helper array with range of potential valid values used to calculate distributions.
    DIST_RANGE_STEP = 0.001
    dist_range = None

    def setup_valid_range(self, params):
        """ Compute the auto_range based on the mean, standard deviation, and spread """
        range_start = params.get("range_start")
        range_end = params.get("range_end")
        print_and_log(f"Range: {range_start} to {range_end}")
        self.dist_range = np.arange(range_start, range_end, self.DIST_RANGE_STEP)

    def calculate_custom_dist(self, data, metric_module, params):
        """Calls custom density function implemented in defined module."""
        try:
            print_and_log("Using custom density function.")
            return metric_module.metric_density(data, self.dist_range, params)
        except AttributeError:
            error_msg = "Custom distribution density function not implemented, aborting."
            print_and_log(error_msg)
            raise Exception(error_msg)

    def calculate_normal_dist(self, data):
        """Normal dist calculation."""
        mean = np.mean(data)
        std_dev = np.std(data)
        print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        return norm.pdf(self.dist_range, mean, std_dev)

    def calculate_kde_dist(self, data):
        """KDE estimator dist calculation."""
        kernel = gaussian_kde(data)
        return kernel.evaluate(self.dist_range)


class DistanceMetric(Metric):
    """Implements a distance-based metric that can load metric-specific functions from a config."""
    prev_probability_distribution = []  # P
    curr_probability_distribution = []  # Q
    density_estimator = DensityEstimator()

    def _calculate_probability_distribution(self, data):
        """Calculates and returns the probability distribution for the given data."""
        distribution = self.metric_params.get("distribution")
        print_and_log(f"Using distribution: {distribution}")
        if distribution == "custom":
            if self.check_module_loaded():
                return self.density_estimator.calculate_custom_dist(data, self.metric_module, self.metric_params)
        elif distribution == "normal":
            return self.density_estimator.calculate_normal_dist(data)
        elif distribution == "kernel_density":
            return self.density_estimator.calculate_kde_dist(data)
        else:
            raise Exception(f"Unsupported distribution type: {distribution}")

    def _set_dimensionality_reduction(self):
        """Reduces dimensionality for the current probability_distribution."""
        if self.check_module_loaded():
            try:
                self.curr_probability_distribution = self.metric_module.metric_reduction(self.curr_probability_distribution)
            except AttributeError:
                print_and_log("Dimensionality reduction not available for this metric.")

    def _calculate_distance(self):
        """Calculates the distance defined for the current prob dist and the reference one."""
        if self.check_module_loaded():
            return self.metric_module.metric_distance(self.prev_probability_distribution, self.curr_probability_distribution)

    def initial_setup(self, dataset, predictions):
        """Overriden."""
        # Calculate and store the probability distribution for the whole dataset.
        self.density_estimator.setup_valid_range(self.metric_params)
        self.prev_probability_distribution = self._calculate_probability_distribution(dataset.get_output())

    def step_setup(self, timebox):
        """Overriden."""
        # Calculate the probability distribution for the timebox.
        if len(self.curr_probability_distribution) > 0:
            # Only set P to Q when we already have a Q (since in the first loop P is initialized in the initial_setup).
            self.prev_probability_distribution = self.curr_probability_distribution
        self.curr_probability_distribution = self._calculate_probability_distribution(timebox.get_expected_results())
        self._set_dimensionality_reduction()

    def calculate_metric(self):
        """Overriden."""
        return self._calculate_distance()
