import numpy as np
import pandas as pd


from utils import dataframe_helper
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
        dataset_df = super().load_ids_from_file(dataset_filename)

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
        if position >= self.x_band1.size:
            raise Exception(f"Invalid position ({position}) given when adding sample (size is {self.x_band1.size})")
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

    def save_to_file(self, output_filename):
        """Stores Numpy arrays with a dataset into a JSON file."""
        dataset_df = pd.DataFrame()
        dataset_df[dataset.DataSet.ID_KEY] = self.x_ids
        dataset_df[IcebergDataSet.BAND1_KEY] = self.x_band1
        dataset_df[IcebergDataSet.BAND2_KEY] = self.x_band2
        dataset_df[IcebergDataSet.ANGLE_KEY] = self.x_angle
        dataset_df[IcebergDataSet.ICEBERG_KEY] = self.y_output

        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)


