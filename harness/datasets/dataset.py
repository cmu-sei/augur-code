import importlib

import numpy as np

from utils import dataframe_helper


def load_model_module(module_name):
    """Loads a model module given the name"""
    return importlib.import_module("datasets." + module_name)


def create_dataset_class(full_class_name):
    """Returns a class instance of the given dataset class name."""
    parts = full_class_name.rsplit(".", 1)
    module_name = parts[0]
    class_name = parts[1]
    dataset_module = importlib.import_module("datasets." + module_name)
    dataset_instance = getattr(dataset_module, class_name)()
    return dataset_instance


class DataSet:
    """A base dataset, only containing ids. Meant to be an abstract class for more detailed ones to build on."""

    ID_KEY = "id"
    x_ids = np.empty(0, str)

    def get_number_of_samples(self):
        """Gets the current size in num of samples."""
        return self.x_ids.size

    def add_sample(self, sample):
        self.x_ids = np.append(self.x_ids, sample[DataSet.ID_KEY])

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

    def get_sample(self, position):
        """Returns a sample associated to this id, just containing the id, as a dictionary."""
        if position < len(self.x_ids):
            return {DataSet.ID_KEY: self.x_ids[position]}
        else:
            return {}

    def load_ids_from_file(self, dataset_filename, id_key=None):
        """Loads ids from a JSON file into this object. Returns the dataframe for loading the rest."""
        if id_key is None:
            id_key = DataSet.ID_KEY

        # Load the referencing dataset from a file into a dataframe.
        dataset_df = dataframe_helper.load_dataframe_from_file(dataset_filename)
        print("Sample row: ")
        print(dataset_df.head(1))

        self.x_ids = np.array(dataset_df[id_key])
        print("Done storing ids", flush=True)

        return dataset_df
