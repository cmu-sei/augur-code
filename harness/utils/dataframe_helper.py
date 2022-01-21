# Augur: A Step Towards Realistic Drift Detection in Production MLSystems - Code
# Copyright 2022 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by Carnegie Mellon University.
# 
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:
# 1. Tensorflow (https://github.com/tensorflow/tensorflow/blob/master/LICENSE) Copyright 2014 The Regents of the University of California.
# 2. Pandas (https://github.com/pandas-dev/pandas/blob/main/LICENSE) Copyright 2021 AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, and open source contributors.
# 3. scikit-learn (https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) Copyright 2021 The scikit-learn developers.
# 4. numpy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) Copyright 2021 NumPy Developers.
# 5. scipy (https://github.com/scipy/scipy/blob/main/LICENSE.txt) Copyright 2021 SciPy Developers.
# 6. statsmodels (https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) Copyright 2018 Jonathan E. Taylor, Scipy developers, statsmodels Developers.
# 7. matplotlib (https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) Copyright 2016 Matplotlib development team.
# 
# DM22-0044

import os.path

import pandas as pd

# Dataframe handler helper functions.


def merge_files(file1, file2, output_filename):
    """Merges data from two JSON files into one."""
    dataframe1 = load_dataframe_from_file(file1)
    dataframe2 = load_dataframe_from_file(file2)

    print("Merging DataFrames", flush=True)
    merged_df = pd.concat([dataframe1, dataframe2])

    save_dataframe_to_file(merged_df, output_filename)


def load_dataframe_from_file(filename):
    """Loads a JSON file into a dataframe, with default params and log output."""
    print("Loading input file: " + filename, flush=True)
    if not os.path.exists(filename):
        raise Exception(f"Dataset on path {filename} does not exist.")
    dataset_df = pd.read_json(filename)
    print("Done loading data. Rows: " + str(dataset_df.shape[0]), flush=True)
    return dataset_df


def save_dataframe_to_file(dataframe, filename):
    """Stores a pandas dataframe to a JSON file, with default params and log output."""
    print("Saving DataFrame to JSON file " + filename + " (rows: " + str(dataframe.shape[0]) + ")", flush=True)
    dataframe.to_json(filename, orient="records", indent=4)
    print("Finished saving JSON file", flush=True)
