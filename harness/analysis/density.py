# Augur: A Step Towards Realistic Drift Detection in Production MLSystems - Code
# Copyright 2022 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by Carnegie Mellon University.
# 
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:
# 1. Tensorflow (https://github.com/tensorflow/tensorflow/blob/master/LICENSE) Copyright 2014 The Regents of the University of California.
# 2. Pandas (https://github.com/pandas-dev/pandas/blob/main/LICENSE) Copyright 2021 AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, and open source contributors.
# 3. scikit-learn (https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) Copyright 2021 The scikit-learn developers.
# 4. numpy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) Copyright 2021 NumPy Developers.
# 5. scipy (https://github.com/scipy/scipy/blob/main/LICENSE.txt) Copyright 2021 SciPy Developers.
# 6. statsmodels (https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) Copyright 2018 Jonathan E. Taylor, Scipy developers, statsmodels Developers.
# 7. matplotlib (https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) Copyright 2016 Matplotlib development team.
# 
# DM22-0044

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
