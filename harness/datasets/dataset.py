import numpy as np

from utils import dataframe_helper


class TrainingSet(object):
    """Represents a subset of data to train."""
    x_train = []
    y_train = []
    x_validation = None
    y_validation = None
    num_train_samples = 0
    num_validation_samples = 0

    def has_validation(self):
        return self.x_validation is not None and self.y_validation is not None


class DataSet:
    """A base dataset, only containing ids."""

    ID_KEY = "id"
    x_ids = np.empty(0, str)

    def get_number_of_samples(self):
        """Gets the current size in num of samples."""
        return self.x_ids.size

    def add_sample(self, sample):
        self.x_ids = np.append(self.x_ids, sample[DataSet.ID_KEY])

    def get_id_position(self, id_to_find):
        """Gets the position of a given id"""
        position_info = np.where(self.x_ids == id_to_find)
        if len(position_info[0]) == 0:
            raise Exception(f"Id {id_to_find} not found")
        return position_info[0]

    def get_sample_by_id(self, id_to_find):
        """Returns a sample given its id."""
        position = self.get_id_position(id_to_find)
        return self.get_sample(position[0])

    def get_sample(self, position):
        """Returns a sample of this id"""
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
        print("Done loading ids", flush=True)

        return dataset_df
