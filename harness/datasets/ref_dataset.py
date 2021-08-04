import secrets

import numpy as np
import pandas as pd

from utils import dataframe_helper
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

    TIMEBOX_ID_KEY = "timebox_id"
    timebox_ids = np.empty(0, str)

    def get_original_ids(self):
        return self.x_original_ids

    def get_sample(self, position):
        """Returns a sample at the given position."""
        sample = super().get_sample(position)
        if position < len(self.x_original_ids):
            sample[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids[position]
            sample[RefDataSet.TIMEBOX_ID_KEY] = self.timebox_ids[position]
        return sample

    def load_from_file(self, dataset_filename):
        """Loads data from a JSON file into this object."""
        dataset_df = super().load_ids_from_file(dataset_filename)

        self.x_original_ids = np.array(dataset_df[RefDataSet.ORIGINAL_ID_KEY])
        self.timebox_ids = np.array(dataset_df[RefDataSet.TIMEBOX_ID_KEY])
        print("Done loading original ids", flush=True)

    def add_reference(self, original_id, timebox_id=0):
        """Adds an original id by reference. Generates automatically a new id."""
        id = secrets.token_hex(10)
        self.x_ids = np.append(self.x_ids, id)
        self.x_original_ids = np.append(self.x_original_ids, original_id)
        self.timebox_ids = np.append(self.timebox_ids, timebox_id)

    def add_multiple_references(self, original_ids, timebox_id=0):
        """Adds multiple original ids by reference."""
        for id in original_ids:
            self.add_reference(id, timebox_id)

    def save_to_file(self, output_filename):
        """Saves a dataset by only storing its ids and the references to the original ids it has."""
        dataset_df = self.as_dataframe()
        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)

    def as_dataframe(self):
        """Adds internal data to a new dataframe."""
        dataset_df = pd.DataFrame()
        dataset_df[DataSet.ID_KEY] = self.x_ids
        dataset_df[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids
        dataset_df[RefDataSet.TIMEBOX_ID_KEY] = self.timebox_ids
        return dataset_df

    def create_from_reference(self, base_dataset, new_dataset):
        """Creates a new dataset by getting the full samples of a reference from the base dataset."""
        position = 0
        new_dataset_size = self.x_original_ids.size
        new_dataset.allocate_space(new_dataset_size)
        for original_id in self.x_original_ids:
            # Only show print update every 500 ids.
            if position % 500 == 0:
                print(f"Finished preparing {position} samples out of {new_dataset_size}", flush=True)
            full_sample = base_dataset.get_sample_by_id(original_id)
            new_dataset.add_sample(position, full_sample)
            position += 1
        return new_dataset
