
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


class DataSet(object):

    x_ids = []
    x_band1 = []
    x_band2 = []
    x_angle = []
    y_results = []
    x_combined_bands = []

    # Loads data from a JSON file into Numpy arrays.
    def load_data(self, input_filename):
        # Data for each sample has 3 values: band1, band2, inc_angle. band1 and band2 are 2 radar images,
        # each consisting of 75x75 "pixels", floats with a dB unit, obtained by pinging
        # at an angle of inc_angle (band1 and 2 differ on how the response was received).
        print("Loading input file: " + input_filename, flush=True)
        dataset_df = pd.read_json(input_filename)
        print("Done loading data. Rows: " + str(dataset_df.shape[0]), flush=True)

        print("Sample row: ")
        print(dataset_df.head(1))

        # We set the samples with no info on inc_angle to 0 as its value, to simplify.
        dataset_df.inc_angle = dataset_df.inc_angle.replace('na', 0)
        dataset_df.inc_angle = dataset_df.inc_angle.astype(float).fillna(0.0)
        print("Done cleaning up angle", flush=True)

        self.x_ids = np.array(dataset_df["id"])
        self.x_band1 = np.array(dataset_df["band_1"])
        self.x_band2 = np.array(dataset_df["band_2"])
        self.x_angle = np.array(dataset_df.inc_angle)

        if "is_iceberg" in dataset_df.columns:
            self.y_results = np.array(dataset_df["is_iceberg"])

        # Load each image from a one-dimension vector into a 74x75 numpy matrix, for both bands
        x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset_df["band_1"]])
        x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset_df["band_2"]])

        # Put all training data into X_train (bands 1 and 2), X_angle_train, and y_train.
        # Not sure how colons are useful here, nor why there is a 3rd array which is
        # equal to x_band1 (2*x_band1/2).
        self.x_combined_bands = np.concatenate([x_band1[:, :, :, np.newaxis],
                                                x_band2[:, :, :, np.newaxis],
                                                ((x_band1+x_band2 )/2)[:, :, :, np.newaxis]
                                               ], axis=-1)
        print("Done loading train data into numpy arrays", flush=True)

        return

    # Stores Numpy arrays with a dataset into a JSON file.
    def save_data(self, output_filename):
        # TODO: implement this
        #df_from_arr = pd.DataFrame(data=[arr1, arr2])


        return
