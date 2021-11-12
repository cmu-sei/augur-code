import importlib

import numpy as np
from scipy.stats import norm

from utils.logging import print_and_log


def load_metric_module(module_name):
    """Loads a metric module given the name"""
    return importlib.import_module("metrics." + module_name)


def create_metric(metric_info):
    """Factory method."""
    metric_type = metric_info.get("type")
    if metric_type == "DistanceMetric":
        metric = DistanceMetric()
    elif metric_type == "ErrorMetric":
        metric = ErrorMetric()
    else:
        raise Exception(f"Non-supported metric type: {metric_type}")
    return metric


def check_module_loaded(metric_module):
    """Ensures a metrics module is loaded."""
    if metric_module is not None:
        return True
    else:
        raise Exception("No metric module configured!")


class Metric:
    """Generic Metric class, will load specific metric module as required."""
    metric_module = None
    config_params = None
    time_series = None
    ts_predictions = None

    def load_metric_functions(self, config):
        """Loads a metric module from config."""
        self.metric_module = load_metric_module(config.get("module"))
        self.config_params = config.get("params")

    def initial_setup(self, time_series, ts_predictions):
        """Method to be called once before starting to work with this metric."""
        self.time_series = time_series
        self.ts_predictions = ts_predictions

    def step_setup(self, time_interval_id):
        """Method to be called once in each step/iteration."""
        raise NotImplementedError()

    def calculate_metric(self):
        """Generic method to calculate the value of this metric. Should be overriden by submetrics."""
        raise NotImplementedError()


class ErrorMetric(Metric):
    """Implements an error-based metric that calculates error based on output."""
    time_interval_id = None

    def step_setup(self, time_interval_id):
        """Overriden."""
        self.time_interval_id = time_interval_id

    def calculate_metric(self):
        """Implements an error-based metric."""
        if check_module_loaded(self.metric_module):
            return self.metric_module.metric_error(self.time_interval_id, self.time_series, self.ts_predictions)


class DensityEstimator:
    """Implements most common density functions."""
    dist_range = None     # Helper array with range of potential valid values used to calculate distributions.
    config_params = None
    metric_module = None

    def __init__(self, config_params, metric_module):
        """Gets the config params and sets up the dist range."""
        self.config_params = config_params
        self.metric_module = metric_module
        self.setup_valid_range()

    def setup_valid_range(self):
        """ Compute the dist range based the configuration."""
        if self.dist_range is None:
            range_start = self.config_params.get("range_start")
            range_end = self.config_params.get("range_end")
            range_step = self.config_params.get("range_step")
            print_and_log(f"Range: {range_start} to {range_end}")
            self.dist_range = np.arange(range_start, range_end, range_step)

    def calculate_probability_distribution(self, data, metric_params):
        """Calculates and returns the probability distribution for the given data."""
        distribution = self.config_params.get("distribution")
        print_and_log(f"Using distribution: {distribution}")
        if distribution == "custom":
            if check_module_loaded(self.metric_module):
                return self._calculate_custom_dist(data, metric_params, self.metric_module)
        elif distribution == "normal":
            return self._calculate_normal_dist(data, metric_params)
        else:
            raise Exception(f"Unsupported distribution type: {distribution}")

    def _calculate_custom_dist(self, data, metric_params):
        """Calls custom density function implemented in defined module."""
        try:
            print_and_log("Using custom density function.")
            return self.metric_module.metric_density(data, self.dist_range, metric_params, self.config_params)
        except AttributeError:
            error_msg = "Custom distribution density function not implemented, aborting."
            print_and_log(error_msg)
            raise Exception(error_msg)

    def _calculate_normal_dist(self, data, params):
        """Normal dist calculation."""
        mean = np.mean(data)
        std_dev = params.get("std_dev")
        if std_dev is None:
            raise Exception("Can't calculate normal distribution; standard deviation provided for data is None")
        print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        return norm.pdf(self.dist_range, mean, std_dev)


class DistanceMetric(Metric):
    """Implements a distance-based metric that can load metric-specific functions from a config."""
    prev_probability_distribution = []  # P
    curr_probability_distribution = []  # Q
    density_estimator = None

    def initial_setup(self, time_series, ts_predictions):
        """Overriden."""
        super().initial_setup(time_series, ts_predictions)
        self.density_estimator = DensityEstimator(self.config_params, self.metric_module)

    def step_setup(self, time_interval_id):
        """Overriden."""
        # Calculate the probability distribution for the time interval.
        self.prev_probability_distribution = self.density_estimator.calculate_probability_distribution(self.time_series.get_aggregated(),
                                                                                                       self.ts_predictions.get_pdf_params(time_interval_id))
        self.curr_probability_distribution = self.ts_predictions.get_pdf(time_interval_id)

    def calculate_metric(self):
        """Calculates the distance defined for the current prob dist and the reference one."""
        if check_module_loaded(self.metric_module):
            return self.metric_module.metric_distance(self.prev_probability_distribution, self.curr_probability_distribution)
