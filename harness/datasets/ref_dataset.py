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

import secrets

import numpy as np

from datasets.dataset import DataSet


def load_full_from_ref_and_base(dataset_class, ref_dataset_file, base_dataset_file):
    """Given filenames for ref and base datasets, creates a full one based on them."""
    print("Loading ref dataset...", flush=True)
    reference_dataset = RefDataSet()
    reference_dataset.load_from_file(ref_dataset_file)

    print("Loading base dataset...", flush=True)
    base_dataset = dataset_class()
    base_dataset.load_from_file(base_dataset_file)

    print("Creating full dataset from both...", flush=True)
    full_dataset = dataset_class()
    full_dataset = reference_dataset.create_from_reference(base_dataset, full_dataset)

    return full_dataset, reference_dataset


class RefDataSet(DataSet):
    """A dataset referencing the ids of another dataset."""
    ORIGINAL_ID_KEY = "original_id"
    x_original_ids = np.empty(0, str)

    SAMPLE_GROUP_ID_KEY = "sample_group_id"
    sample_group_ids = np.empty(0, str)

    def get_original_ids(self):
        return self.x_original_ids

    def get_sample(self, position):
        """Returns a sample at the given position."""
        sample = super().get_sample(position)
        if position < len(self.x_original_ids):
            sample[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids[position]
            sample[RefDataSet.SAMPLE_GROUP_ID_KEY] = self.sample_group_ids[position]
        return sample

    def add_sample(self, position, sample):
        """Adds a sample in the given position."""
        if position >= self.x_original_ids.size:
            raise Exception(f"Invalid position ({position}) given when adding sample (size is {self.x_original_ids.size})")
        self.x_original_ids[position] = sample[self.ORIGINAL_ID_KEY]
        self.sample_group_ids[position] = sample[self.SAMPLE_GROUP_ID_KEY]

    def get_num_sample_groups(self):
        """Gets the number of sample groups in this dataset."""
        return np.unique(self.sample_group_ids).size

    def get_sample_group_size(self):
        """Gets the size of each sample group. All are assumed to have the same size."""
        unique, counts = np.unique(self.sample_group_ids, return_counts=True)
        return counts[0]

    def allocate_space(self, size):
        """Pre-allocates the space for this dataset to avoid scalability issues, when the size is known."""
        self.x_original_ids = np.empty(size, dtype=str)
        self.sample_group_ids = np.zeros(size, dtype=str)

    def load_from_file(self, dataset_filename):
        """Loads data from a JSON file into this object."""
        dataset_df = super().load_from_file(dataset_filename)

        self.x_original_ids = np.array(dataset_df[RefDataSet.ORIGINAL_ID_KEY])
        self.sample_group_ids = np.array(dataset_df[RefDataSet.SAMPLE_GROUP_ID_KEY])
        print("Done loading original ids", flush=True)

    def add_reference(self, original_id, sample_group_id=0):
        """Adds an original id by reference. Generates automatically a new id."""
        id = secrets.token_hex(10)
        self.x_ids = np.append(self.x_ids, id)
        self.x_original_ids = np.append(self.x_original_ids, original_id)
        self.sample_group_ids = np.append(self.sample_group_ids, sample_group_id)

    def add_multiple_references(self, original_ids, sample_group_id=0):
        """Adds multiple original ids by reference."""
        for id in original_ids:
            self.add_reference(id, sample_group_id)

    def as_dataframe(self, include_all_data=True):
        """Adds internal data to a new dataframe."""
        dataset_df = super().as_dataframe()
        if include_all_data:
            dataset_df[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids
            dataset_df[RefDataSet.SAMPLE_GROUP_ID_KEY] = self.sample_group_ids
        return dataset_df

    def create_from_reference(self, base_dataset, new_dataset):
        """Creates a new dataset by getting the full samples of a reference from the base dataset."""
        new_dataset_size = self.x_original_ids.size
        new_dataset.allocate_space(new_dataset_size)
        for idx, original_id in enumerate(self.x_original_ids):
            # Only show print update every 500 ids.
            if idx % 500 == 0:
                print(f"Finished preparing {idx} samples out of {new_dataset_size}", flush=True)

            # Get the full sample, but replace the timestamp (if any) with the one from the reference dataset.
            full_sample = base_dataset.get_sample_by_id(original_id)
            full_sample[DataSet.TIMESTAMP_KEY] = self.timestamps[idx]
            new_dataset.add_sample(idx, full_sample)
        return new_dataset
