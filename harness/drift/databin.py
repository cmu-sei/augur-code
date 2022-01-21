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

import random


def create_bins(bin_info_list, shuffle):
    """Creates a list of bins based on a string list of bin info."""
    bins = []
    for bin_info in bin_info_list:
        bins.append(DataBin(bin_info[0], bin_info[1], shuffle))
    return bins


def sort_into_bins(ids, values, bins):
    """Sorts the ids into bins, based on the given values matching bin values."""
    for sample_idx, value in enumerate(values):
        for bin_idx, bin in enumerate(bins):
            if value == bin.value:
                bins[bin_idx].add(ids[sample_idx])
                break

    return bins


class DataBin:
    """Class to represent a bin of data of a certain classification."""
    name = ""
    value = None
    shuffle = True
    ids = []
    id_queue = []

    def __init__(self, new_name, new_value, shuffle=True):
        self.name = new_name
        self.value = new_value
        self.shuffle=shuffle
        self.ids = []

    def setup_queue(self):
        self.id_queue = self.ids.copy()
        if self.shuffle:
            random.shuffle(self.id_queue)

    def get_queue_length(self):
        return len(self.id_queue)

    def pop_from_queue(self):
        if len(self.id_queue) > 0:
            return self.id_queue.pop(0)
        else:
            return None

    def add(self, new_id):
        self.ids.append(new_id)

    def get_random(self):
        return random.choice(self.ids)

    def size(self):
        return len(self.ids)

    def info(self):
        return self.name + ": " + str(self.size()) + " samples "

    def to_string(self):
        return self.name + ": " + str(self.ids)
