import secrets

import numpy as np
import pandas as pd

from utils import dataframe_helper
from datasets import dataset


class RefDataSet(dataset.DataSet):
    """A dataset referencing the ids of another dataset."""
    ORIGINAL_ID_KEY = "original_id"
    x_original_ids = np.empty(0, str)

    def get_original_ids(self):
        return self.x_original_ids

    def get_sample(self, position):
        """Returns a sample at the given position."""
        sample = super().get_sample(position)
        if position < len(self.x_original_ids):
            sample[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids[position]
        return sample

    def load_from_file(self, dataset_filename):
        """Loads data from a JSON file into this object."""
        dataset_df = super().load_ids_from_file(dataset_filename)

        self.x_original_ids = np.array(dataset_df["original_id"])
        print("Done loading original ids", flush=True)

    def add_reference(self, original_id):
        """Adds an original id by reference. Generates automatically a new id."""
        id = secrets.token_hex(10)
        self.x_ids = np.append(self.x_ids, id)
        self.x_original_ids = np.append(self.x_original_ids, original_id)

    def add_multiple_references(self, original_ids):
        """Adds multiple original ids by reference."""
        for id in original_ids:
            self.add_reference(id)

    def save_to_file(self, output_filename):
        """Saves a dataset by only storing its ids and the references to the original ids it has."""
        dataset_df = pd.DataFrame()
        dataset_df[dataset.DataSet.ID_KEY] = self.x_ids
        dataset_df[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids

        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)

    def create_from_reference(self, base_dataset, new_dataset):
        """Creates a new dataset by getting the full samples of a reference from the base dataset."""
        for original_id in self.x_original_ids:
            full_sample = base_dataset.get_sample_by_id(original_id)
            new_dataset.add_sample(full_sample)
        return new_dataset
