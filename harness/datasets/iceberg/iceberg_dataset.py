import numpy as np
import pandas as pd
import sklearn.model_selection as skm

from utils import dataframe_helper
from datasets import dataset
from datasets.dataset import TrainingSet


class IcebergDataSet(dataset.DataSet):
    """A dataset following the Kaggle competition format of SAR data."""
    BAND1_KEY = "band_1"
    BAND2_KEY = "band_2"
    ANGLE_KEY = "inc_angle"
    ICEBERG_KEY = "is_iceberg"
    COMBINED_BANDS_KEY = "merged_bands"

    x_band1 = np.empty(0)
    x_band2 = np.empty(0)
    x_angle = np.empty(0)
    y_output = np.empty(0, int)
    x_combined_bands = np.empty(0)

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

        # Put all training data into X_train (bands 1 and 2), X_angle_train, and y_train.
        # Load each image from a one-dimension vector into a 74x75 numpy matrix, for both bands
        x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset_df["band_1"]])
        x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset_df["band_2"]])

        # Put all training data into X_train (bands 1 and 2), X_angle_train, and y_train.
        # Not sure how colons are useful here, nor why there is a 3rd array which is
        # equal to x_band1 (2*x_band1/2).
        self.x_combined_bands = np.concatenate([x_band1[:, :, :, np.newaxis],
                                                x_band2[:, :, :, np.newaxis],
                                                ((x_band1+x_band2)/2)[:, :, :, np.newaxis]
                                                ], axis=-1)

        print("Done loading data into numpy arrays", flush=True)

    def add_sample(self, sample):
        """Adds a sample from a dictionary."""
        super().add_sample(sample)
        self.x_band1 = np.append(self.x_band1, sample[IcebergDataSet.BAND1_KEY])
        self.x_band2 = np.append(self.x_band2, sample[IcebergDataSet.BAND2_KEY])
        self.x_combined_bands = np.append(self.x_combined_bands, sample[IcebergDataSet.COMBINED_BANDS_KEY])
        self.x_angle = np.append(self.x_angle, sample[IcebergDataSet.ANGLE_KEY])
        if IcebergDataSet.ICEBERG_KEY in sample:
            self.y_output = np.append(self.y_output, sample[IcebergDataSet.ICEBERG_KEY])
        print(f"Len bands: {self.x_band1.size}, {self.x_combined_bands.size}, len angle: {self.x_angle.size}")

    def get_sample(self, position):
        """Returns a sample as as dict."""
        print(f"Base Len bands: {self.x_band1.size}, {self.x_combined_bands.size}, len angle: {self.x_angle.size}")
        sample = super().get_sample(position)
        if len(self.x_band1) > position:
            sample[IcebergDataSet.BAND1_KEY] = self.x_band1[position]
            sample[IcebergDataSet.BAND2_KEY] = self.x_band2[position]
            sample[IcebergDataSet.COMBINED_BANDS_KEY] = self.x_combined_bands[position]
            sample[IcebergDataSet.ANGLE_KEY] = self.x_angle[position]
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
        dataset_df["id"] = self.x_ids
        dataset_df["band_1"] = self.x_band1
        dataset_df["band_2"] = self.x_band2
        dataset_df["inc_angle"] = self.x_angle
        dataset_df["is_iceberg"] = self.y_output

        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)

    def split_data(self, validation_percentage):
        """Split training set into train and validation (75% to actually train)"""
        x_band_t, x_band_v, x_angle_t, x_angle_v, y_train, y_validation = skm.train_test_split(self.x_combined_bands,
                                                                                               self.x_angle,
                                                                                               self.y_output,
                                                                                               random_state=42,
                                                                                               test_size=validation_percentage)
        print("Done splitting validation data from train data", flush=True)
        training_set = TrainingSet()
        training_set.x_train = [x_band_t, x_angle_t]
        training_set.x_validation = [x_band_v, x_angle_v]
        training_set.y_train = y_train
        training_set.y_validation = y_validation
        training_set.num_train_samples = y_train.shape[0]
        training_set.num_validation_samples = y_validation.shape[0]
        return training_set

    def get_fold_data(self, train_index, test_index):
        """Prepares a training set for the given dataset and indexes"""
        training_set = TrainingSet()
        training_set.x_train = [self.x_combined_bands[train_index], self.x_angle[train_index]]
        training_set.y_train = self.y_output[train_index]
        training_set.x_validation = [self.x_combined_bands[test_index], self.x_angle[test_index]]
        training_set.y_validation = self.y_output[test_index]
        training_set.num_train_samples = training_set.y_train.shape[0]
        training_set.num_validation_samples = training_set.y_validation.shape[0]
        return training_set
