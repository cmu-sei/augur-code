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

import importlib
import abc
from abc import abstractmethod

import numpy as np
import pandas as pd

from utils import dataframe_helper

DATASETS_PACKAGE = "extensions.datasets."


def load_model_module(module_name):
    """Loads a model module given the name"""
    return importlib.import_module(DATASETS_PACKAGE + module_name)


def load_dataset_class(full_class_name):
    """Returns a class instance of the given dataset class name."""
    parts = full_class_name.rsplit(".", 1)
    module_name = parts[0]
    class_name = parts[1]
    dataset_module = importlib.import_module(DATASETS_PACKAGE + module_name)
    class_type = getattr(dataset_module, class_name)
    return class_type


class DataSet(abc.ABC):
    """A base dataset, only containing ids. Meant to be an abstract class for more detailed ones to build on."""

    ID_KEY = "id"
    TIMESTAMP_KEY = "timestamp"

    x_ids = np.empty(0, dtype=str)
    timestamps = np.empty(0, int)

    def get_ids(self):
        """Returns the dataset ids."""
        return self.x_ids

    def get_number_of_samples(self):
        """Gets the current size in num of samples."""
        return self.x_ids.size

    def set_timestamps(self, timestamps):
        """Sets the timestamps."""
        self.timestamps = timestamps

    def get_timestamps(self):
        """Returns the timestamps."""
        return self.timestamps

    def has_timestamps(self):
        """If the sum of the timestamps is zero, they have not been loaded."""
        return np.sum(self.timestamps) != 0

    @abstractmethod
    def add_sample(self, position, sample):
        """Adds a sample in the given position."""
        if position >= self.x_ids.size:
            raise Exception(f"Invalid position ({position}) given when adding sample (size is {self.x_ids.size})")
        self.x_ids[position] = sample[DataSet.ID_KEY]
        if self.TIMESTAMP_KEY in sample.keys():
            self.timestamps[position] = sample[DataSet.TIMESTAMP_KEY]

    def _get_id_position(self, id_to_find):
        """Gets the position of a given id"""
        position_info = np.where(self.x_ids == id_to_find)
        if len(position_info[0]) == 0:
            raise Exception(f"Id {id_to_find} not found")
        return position_info[0]

    def get_sample_by_id(self, id_to_find):
        """Returns a sample given its id."""
        position = self._get_id_position(id_to_find)
        return self.get_sample(position[0])

    @abstractmethod
    def get_sample(self, position):
        """Returns a sample associated to this id, just containing the id, as a dictionary."""
        if position < len(self.x_ids):
            return {DataSet.ID_KEY: self.x_ids[position], DataSet.TIMESTAMP_KEY: self.timestamps[position]}
        else:
            return {}

    @abstractmethod
    def load_from_file(self, dataset_filename, id_key=None, timestamp_key=None, convert_to_timestamp=None):
        """Loads ids and times from a JSON file into this object. Returns the dataframe for loading the rest.
           convert_to_timestamp should be a function that converts from whatever format the dataset has its datetime,
           to Unix timestamp."""
        # Load the referencing dataset from a file into a dataframe.
        dataset_df = dataframe_helper.load_dataframe_from_file(dataset_filename)
        print("Sample row: ")
        print(dataset_df.head(1))

        if id_key is None:
            id_key = DataSet.ID_KEY
        if timestamp_key is None:
            timestamp_key = DataSet.TIMESTAMP_KEY
        if convert_to_timestamp is None:
            convert_to_timestamp = lambda x: x

        try:
            self.x_ids = np.array(dataset_df[id_key])
            if timestamp_key in dataset_df.columns:
                self.timestamps = convert_to_timestamp(np.array(dataset_df[timestamp_key]))
            else:
                self.timestamps = np.zeros(self.x_ids.size)
        except KeyError as ex:
            raise Exception(f"Could not load ids from dataset '{dataset_filename}': {type(ex).__name__}: {str(ex)}")
        print("Done storing ids", flush=True)

        return dataset_df

    @abstractmethod
    def allocate_space(self, size):
        """Pre-allocates the space for this dataset to avoid scalability issues, when the size is known."""
        self.x_ids = np.empty(size, dtype=str)
        self.timestamps = np.zeros(size, dtype=int)

    @abstractmethod
    def as_dataframe(self, include_all_data=True):
        """Adds internal data to a new dataframe."""
        dataset_df = pd.DataFrame()
        if include_all_data:
            dataset_df[DataSet.ID_KEY] = self.x_ids
            dataset_df[DataSet.TIMESTAMP_KEY] = self.timestamps
        return dataset_df

    def save_to_file(self, output_filename):
        """Stores Numpy arrays with a dataset into a JSON file."""
        dataset_df = self.as_dataframe()
        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)

    def get_model_input(self):
        """Returns the inputs to be used as a numpy array."""
        raise NotImplementedError()

    def get_single_input(self):
        """For models that have multiple separate inputs, this returns only one of them for sizing purposes.
        If dataset provides just one input, this should return the same as get_model_input."""
        return self.get_model_input()

    def get_output(self):
        """Has to return a numpy array with the output."""
        raise NotImplementedError()

    def set_output(self, new_output):
        """Gets a numpy array and sets it as the output."""
        raise NotImplementedError()
