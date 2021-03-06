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

from datasets import dataset


class IcebergDataSet(dataset.DataSet):
    """A dataset following the Kaggle competition format of SAR data."""
    BAND1_KEY = "band_1"
    BAND2_KEY = "band_2"
    ANGLE_KEY = "inc_angle"
    ICEBERG_KEY = "is_iceberg"
    COMBINED_BANDS_KEY = "merged_bands"

    BAND_WIDTH = 75
    BAND_HEIGHT = 75
    BAND_DEPTH = 3

    x_band1 = np.empty(0)
    x_band2 = np.empty(0)
    x_angle = np.empty(0)
    y_output = np.empty(0, int)
    x_combined_bands = np.empty((0, BAND_WIDTH, BAND_HEIGHT, BAND_DEPTH))

    def load_from_file(self, dataset_filename):
        """Loads data from a JSON file into this object."""
        dataset_df = super().load_from_file(dataset_filename)

        # We set the samples with no info on inc_angle to 0 as its value, to simplify.
        dataset_df.inc_angle = dataset_df.inc_angle.replace('na', 0)
        dataset_df.inc_angle = dataset_df.inc_angle.astype(float).fillna(0.0)
        print("Done cleaning up angle", flush=True)

        # Store locally the data parts.
        self.x_band1 = np.array(dataset_df[IcebergDataSet.BAND1_KEY])
        self.x_band2 = np.array(dataset_df[IcebergDataSet.BAND2_KEY])
        self.x_angle = np.array(dataset_df[IcebergDataSet.ANGLE_KEY])
        if IcebergDataSet.ICEBERG_KEY in dataset_df.columns:
            self.y_output = np.array(dataset_df[IcebergDataSet.ICEBERG_KEY])

        # Generate the combined bands.
        self.post_process_data()
        print("Done loading data into numpy arrays", flush=True)

    def post_process_data(self):
        """Sets up a combined set of inputs containing each separate band, plus a combined image of both bands."""
        square_x_band1 = np.array([np.array(band).astype(np.float32).reshape(self.BAND_WIDTH, self.BAND_HEIGHT) for band in self.x_band1])
        square_x_band2 = np.array([np.array(band).astype(np.float32).reshape(self.BAND_WIDTH, self.BAND_HEIGHT) for band in self.x_band2])
        self.x_combined_bands = np.concatenate([square_x_band1[:, :, :, np.newaxis],
                                                square_x_band2[:, :, :, np.newaxis],
                                                ((square_x_band1+square_x_band2)/2)[:, :, :, np.newaxis]
                                                ], axis=-1)

    def allocate_space(self, size):
        """Pre-allocates the space for this dataset to avoid scalability issues, when the size is known."""
        super().allocate_space(size)
        self.x_band1 = np.zeros((size, self.BAND_WIDTH * self.BAND_HEIGHT))
        self.x_band2 = np.zeros((size, self.BAND_WIDTH * self.BAND_HEIGHT))
        self.x_angle = np.zeros(size)
        self.y_output = np.zeros(size, dtype=int)
        self.x_combined_bands = np.zeros((size, self.BAND_WIDTH, self.BAND_HEIGHT, self.BAND_DEPTH))

    def add_sample(self, position, sample):
        """Adds a sample from a dictionary to a given position."""
        super().add_sample(position, sample)
        if position >= self.get_number_of_samples():
            raise Exception(f"Invalid position ({position}) given when adding sample (size is {self.get_number_of_samples()}. Please use allocate_space() to prepare arrays first.)")
        self.x_band1[position] = sample[IcebergDataSet.BAND1_KEY]
        self.x_band2[position] = sample[IcebergDataSet.BAND2_KEY]
        self.x_angle[position] = sample[IcebergDataSet.ANGLE_KEY]
        self.x_combined_bands[position] = sample[IcebergDataSet.COMBINED_BANDS_KEY]
        if IcebergDataSet.ICEBERG_KEY in sample:
            self.y_output[position] = sample[IcebergDataSet.ICEBERG_KEY]

    def get_sample(self, position):
        """Returns a sample as as dict."""
        sample = super().get_sample(position)
        if len(self.x_band1) > position:
            sample[IcebergDataSet.BAND1_KEY] = self.x_band1[position]
            sample[IcebergDataSet.BAND2_KEY] = self.x_band2[position]
            sample[IcebergDataSet.ANGLE_KEY] = self.x_angle[position]
            sample[IcebergDataSet.COMBINED_BANDS_KEY] = self.x_combined_bands[position]
        if len(self.y_output) > position:
            sample[IcebergDataSet.ICEBERG_KEY] = self.y_output[position]
        return sample

    def as_dataframe(self, include_all_data=True):
        """Adds internal data to a new dataframe."""
        dataset_df = super().as_dataframe()
        if include_all_data:
            dataset_df[IcebergDataSet.BAND1_KEY] = self.x_band1
            dataset_df[IcebergDataSet.BAND2_KEY] = self.x_band2
            dataset_df[IcebergDataSet.ANGLE_KEY] = self.x_angle
            dataset_df[IcebergDataSet.ICEBERG_KEY] = self.y_output
        return dataset_df

    def get_model_input(self):
        """Returns the 2 inputs to be used: the combined bands and the angle."""
        return [self.x_combined_bands, self.x_angle]

    def get_single_input(self):
        """For models that have multiple separate inputs, this returns only one of them for sizing purposes.
        If dataset provides just one input, this should return the same as get_model_input."""
        return self.x_combined_bands

    def get_output(self):
        return self.y_output

    def set_output(self, new_output):
        self.y_output = new_output
