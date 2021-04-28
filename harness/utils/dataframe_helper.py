import os.path

import pandas as pd

# Dataframe handler helper functions.


#  Merges data from two JSON files into one.
def merge_files(file1, file2, output_filename):
    dataframe1 = load_dataframe_from_file(file1)
    dataframe2 = load_dataframe_from_file(file2)

    print("Merging DataFrames", flush=True)
    merged_df = pd.concat([dataframe1, dataframe2])

    save_dataframe_to_file(merged_df, output_filename)


# Loads a JSON file into a dataframe, with default params and log output.
def load_dataframe_from_file(filename):
    print("Loading input file: " + filename, flush=True)
    if not os.path.exists(filename):
        raise Exception(f"Dataset on path {filename} does not exist.")
    dataset_df = pd.read_json(filename)
    print("Done loading data. Rows: " + str(dataset_df.shape[0]), flush=True)
    return dataset_df


# Stores a pandas dataframe to a JSON file, with default params and log output.
def save_dataframe_to_file(dataframe, filename):
    print("Saving DataFrame to JSON file " + filename + " (rows: " + str(dataframe.shape[0]) + ")", flush=True)
    dataframe.to_json(filename, orient="records", indent=4)
    print("Finished saving JSON file", flush=True)
