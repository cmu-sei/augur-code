import numpy as np
from scipy.stats import norm

from utils.logging import print_and_log


class DensityEstimator:
    """Implements most common density functions."""
    dist_range = None     # Helper array with range of potential valid values used to calculate distributions.
    distribution = ""
    config_params = None
    external_module = None

    def __init__(self, config_params, external_module=None):
        """Gets the config params and sets up the dist range. External_module may contain external implementation
        of a density function."""
        self.config_params = config_params
        self.external_module = external_module
        self.setup_valid_range()

    def setup_valid_range(self):
        """ Compute the dist range based the configuration."""
        if self.dist_range is None:
            range_start = self.config_params.get("range_start")
            range_end = self.config_params.get("range_end")
            range_step = self.config_params.get("range_step")
            print_and_log(f"Range: {range_start} to {range_end}")
            self.dist_range = np.arange(range_start, range_end, range_step)

    def calculate_probability_distribution(self, distribution, data, density_params):
        """Calculates and returns the probability distribution for the given data."""
        #print_and_log(f"Using distribution: {distribution}")
        if distribution == "custom":
            if self.external_module is None:
                raise Exception("Custom density requested, but no custom module set up.")
            return self._calculate_custom_dist(data, density_params)
        elif distribution == "normal":
            if data is None:
                return self._calculate_normal_dist_from_params(density_params)
            else:
                return self._calculate_normal_dist(data, density_params)
        else:
            raise Exception(f"Unsupported distribution type: {distribution}.")

    def _calculate_custom_dist(self, data, density_params):
        """Calls custom density function implemented in defined module."""
        try:
            print_and_log("Using custom density function.")
            return self.external_module.metric_density(data, self.dist_range, density_params, self.config_params)
        except AttributeError:
            error_msg = "Custom distribution density function not implemented, aborting."
            print_and_log(error_msg)
            raise Exception(error_msg)

    def _calculate_normal_dist_from_params(self, density_params):
        """Normal dist calculation, mean and std dev from params. data is ignored."""
        mean = density_params.get("mean")
        std_dev = density_params.get("std_dev")
        if mean is None or std_dev is None:
            raise Exception("Can't calculate normal distribution; one of the params is None")
        #print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        dist = norm.pdf(self.dist_range, mean, std_dev)
        return dist

    def _calculate_normal_dist(self, data, density_params):
        """Normal dist calculation, using only std dev from params."""
        mean = np.mean(data)
        std_dev = density_params.get("std_dev")
        if std_dev is None:
            raise Exception("Can't calculate normal distribution; standard deviation provided for data is None")
        #print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        dist = norm.pdf(self.dist_range, mean, std_dev)
        return dist
