
import numpy as np
import pandas as pd


# A dataset following the Kaggle competition format of SAR data.
class DataSet(object):

    x_ids = []
    x_band1 = []
    x_band2 = []
    x_angle = []
    y_results = []
    x_combined_bands = []
    x_original_ids = []

    # Loads data from a JSON file into this object.
    def load_data(self, dataset_filename, basedataset_filename=None):
        dataset_df = DataSet.load_dataset(dataset_filename, basedataset_filename)
        self.prepare_dataset(dataset_df)
        return

    # Loads data from a JSON file into a Pandas dataframe.
    # If a base dataset filename is present, it means that the dataset only contains pointers by id to the data in the
    # base dataset file.
    @staticmethod
    def load_dataset(dataset_filename, base_dataset_filename=None):
        print("Loading input file: " + dataset_filename, flush=True)
        dataset_df = pd.read_json(dataset_filename)
        print("Done loading data. Rows: " + str(dataset_df.shape[0]), flush=True)

        # Load actual data from base dataset, if any.
        # TODO: maybe optimize this?
        if base_dataset_filename is None:
            print("No base dataset found.")
            final_dataset_df = dataset_df
        else:
            final_dataset_df = DataSet.get_from_base(dataset_df, base_dataset_filename)

        print("Sample row: ")
        print(final_dataset_df.head(1))

        return final_dataset_df

    # Go over the original_ids listed in the dataset, and create a new dataframe by appending each of those
    # rows from the base dataset.
    @staticmethod
    def get_from_base(referencing_df, base_filename):
        print("Getting dataset from base.")
        base_dataset_df = pd.read_json(base_filename)
        new_columns = base_dataset_df.columns.tolist()
        new_columns.append("original_id")
        final_dataset_df = pd.DataFrame(columns=new_columns)
        for index, row in referencing_df.iterrows():
            base_id = row["original_id"]
            original_row = base_dataset_df.loc[base_dataset_df["id"] == base_id]
            original_row["id"] = row["id"]
            original_row["original_id"] = base_id
            final_dataset_df = final_dataset_df.append(original_row, ignore_index=True)

        #print(final_dataset_df)
        return final_dataset_df

    # Cleans up and prepares dataset data from raw dataset info.
    # Data for each sample has 3 values: band1, band2, inc_angle. band1 and band2 are 2 radar images,
    # each consisting of 75x75 "pixels", floats with a dB unit, obtained by pinging
    # at an angle of inc_angle (band1 and 2 differ on how the response was received).
    def prepare_dataset(self, dataset_df):
        # We set the samples with no info on inc_angle to 0 as its value, to simplify.
        dataset_df.inc_angle = dataset_df.inc_angle.replace('na', 0)
        dataset_df.inc_angle = dataset_df.inc_angle.astype(float).fillna(0.0)
        print("Done cleaning up angle", flush=True)

        # Store locally the data parts.
        self.x_ids = np.array(dataset_df["id"])
        self.x_band1 = np.array(dataset_df["band_1"])
        self.x_band2 = np.array(dataset_df["band_2"])
        self.x_angle = np.array(dataset_df.inc_angle)
        if "is_iceberg" in dataset_df.columns:
            self.y_results = np.array(dataset_df["is_iceberg"])
        if "original_id" in dataset_df.columns:
            self.x_original_ids = np.array(dataset_df["original_id"])

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

    #  Merges data sets from two files into one.
    @staticmethod
    def merge_datasets(dataset1_filename, dataset2_filename, output_filename):
        print("Loading input file: " + dataset1_filename, flush=True)
        dataset1_df = pd.read_json(dataset1_filename)
        print("Done loading data. Rows: " + str(dataset1_df.shape[0]), flush=True)

        print("Loading input file: " + dataset2_filename, flush=True)
        dataset2_df = pd.read_json(dataset2_filename)
        print("Done loading data. Rows: " + str(dataset2_df.shape[0]), flush=True)

        print("Merging DataFrames", flush=True)
        merged_df = pd.concat([dataset1_df, dataset2_df])

        print("Saving DataFrame to JSON file", flush=True)
        merged_df.to_json(output_filename, orient="records", indent=4)
        print("Finished saving JSON file", flush=True)

    # Stores Numpy arrays with a dataset into a JSON file.
    def save_data(self, output_filename):
        dataset_df = pd.DataFrame()
        dataset_df["id"] = self.x_ids
        dataset_df["band_1"] = self.x_band1
        dataset_df["band_2"] = self.x_band2
        dataset_df["inc_angle"] = self.x_angle
        dataset_df["is_iceberg"] = self.y_results

        print("Saving DataFrame to JSON file", flush=True)
        dataset_df.to_json(output_filename, orient="records", indent=4)
        print("Finished saving JSON file", flush=True)
        return
