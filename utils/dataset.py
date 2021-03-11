
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# Loads data from a JSON file into Numpy arrays.
def load_data(input_filename):
    # Data for each sample has 3 values: band1, band2, inc_angle. band1 and band2 are 2 radar images,
    # each consisting of 75x75 "pixels", floats with a dB unit, obtained by pinging
    # at an angle of inc_angle (band1 and 2 differ on how the response was received).
    print("Loading input files", flush=True)
    dataset = pd.read_json(input_filename)
    print("Done loading data. Rows: " + str(dataset.shape[0]), flush=True)

    print("Sample row: ")
    print(dataset.head(1))

    # We set the samples with no info on inc_angle to 0 as its value, to simplify.
    dataset.inc_angle = dataset.inc_angle.replace('na', 0)
    dataset.inc_angle = dataset.inc_angle.astype(float).fillna(0.0)
    print("Done cleaning up angle", flush=True)

    # Load each image from a one-dimension vector into a 74x75 numpy matrix, for both bands
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in dataset["band_2"]])

    # Put all training data into X_train (bands 1 and 2), X_angle_train, and y_train.
    # Not sure how colons are useful here, nor why there is a 3rd array which is
    # equal to x_band1 (2*x_band1/2).
    x_bands = np.concatenate([
           x_band1[:, :, :, np.newaxis],
           x_band2[:, :, :, np.newaxis],
           ((x_band1+x_band2 )/2)[:, :, :, np.newaxis]
        ],
        axis=-1)
    x_angle = np.array(dataset.inc_angle)
    y_results = np.array(dataset["is_iceberg"])
    print("Done loading train data into numpy arrays", flush=True)

    return [x_bands, x_angle, y_results]


# Stores Numpy arrays with a dataset into a JSON file.
def save_data(x_bands, x_angle, y_results):
    # TODO: implement this
    return
