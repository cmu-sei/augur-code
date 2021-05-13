import secrets

import numpy as np
import pandas as pd

from utils import dataframe_helper


class RefDataSet:
    """A dataset referencing the ids of another dataset."""
    x_ids = np.empty(0, str)
    x_original_ids = np.empty(0, str)
    base_dataset_filename = None

    def get_number_of_samples(self):
        """Gets the current size in num of samples."""
        return self.x_ids.size

    def load_from_file(self, dataset_filename):
        """Loads data from a JSON file into this object."""

        # Load the referencing dataset from a file into a dataframe.
        dataset_df = dataframe_helper.load_dataframe_from_file(dataset_filename)
        print("Sample row: ")
        print(dataset_df.head(1))

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
        dataset_df["id"] = self.x_ids
        dataset_df["original_id"] = self.x_original_ids

        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)
