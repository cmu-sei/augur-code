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

import json
import os
import shutil
import datetime

from analysis.predictions import Predictions
from analysis.timeseries import TimeSeries
import analysis.metric as augur_metrics
from utils.logging import print_and_log
import training.timeseries_model as timeseries_model


def predict(model, model_input, threshold):
    """Generates predictions based on model, returns object with raw and classified predictions."""
    raw_predictions = model.predict(model_input).flatten()
    print_and_log(f"Predictions shape: {raw_predictions.shape}")
    predictions = Predictions(threshold)
    predictions.store_raw_predictions(raw_predictions)
    predictions.classify_raw_predictions()
    return predictions


def ts_predict(ts_fit_model, time_series):
    """Fits model and generates predictions based on model."""
    ts_predictions = timeseries_model.predict(ts_fit_model, time_series)
    return ts_predictions


def calculate_accuracy(predictions, time_series):
    """Calculates the accuracy of a classifier using time intervals defined in a time series."""
    accuracy_by_interval = {}
    starting_sample_idx = 0
    print_and_log("Calculating accuracy by interval.")
    for interval_index, time_interval in enumerate(time_series.get_time_intervals()):
        # Calculate the size of this interval, and obtain a slice from the current idx of that size.
        interval_sample_size = time_series.get_num_samples(interval_index)
        predictions_slice = predictions.create_slice(starting_sample_idx, interval_sample_size)

        # Calculate the accuracy of the slice and store it.
        accuracy = None
        if predictions_slice is not None:
            accuracy = predictions_slice.get_accuracy()
        accuracy_by_interval[interval_index] = accuracy

        # Update the starting idx for next iteration.
        starting_sample_idx += interval_sample_size

    print_and_log("Finished calculating accuracy by interval.")
    return accuracy_by_interval


def calculate_metrics(time_series, ts_predictions, config):
    """Calculates metrics for the given configs and dataset."""
    if not config.contains("metrics"):
        print_and_log("No metrics configured.")
        return None

    metrics = config.get("metrics")
    results = {}
    for metric_info in metrics:
        metric_name = metric_info.get('name')
        print_and_log(f"Loading metric: {metric_name}")
        metric = augur_metrics.create_metric(metric_info)
        metric.load_metric_functions(metric_info)
        metric.initial_setup(time_series, ts_predictions)

        print_and_log(f"Calculating metric: {metric_name}")
        for interval_index, time_interval in enumerate(time_series.get_time_intervals()):
            if interval_index not in results.keys():
                results[interval_index] = {}
                results[interval_index]["interval"] = time_interval.timestamp()
                results[interval_index]["metrics"] = []

            # Calculate metric.
            try:
                #print_and_log(f"Calculating metric for interval: {time_interval}")
                metric.step_setup(interval_index)
                metric_value = metric.calculate_metric()
            except Exception as ex:
                print_and_log(f"WARNING: Could not prepare or calculate metric {metric_name} for interval {interval_index}: {str(ex)}")
                continue

            # Accumulate results.
            results[interval_index]["metrics"].append({"name": metric_name, "value": metric_value})

    return results


def add_accuracy(metric_results, accuracy_list):
    """Merges the accuracy results into the metrics results."""
    for interval_index, accuracy in accuracy_list.items():
        metric_results[interval_index]["accuracy"] = accuracy
    return metric_results


def analyze(full_dataset, predictions, config):
    """Analyzes the data."""
    # Aggregate dataset and calculate original dataset classifier accuracy by time interval.
    time_series = TimeSeries()
    time_series.aggregate_by_timestamp(config.get("time_interval").get("starting_interval"),
                                       config.get("time_interval").get("interval_unit"),
                                       predictions.get_predictions(),
                                       full_dataset.get_timestamps())
    accuracy = calculate_accuracy(predictions, time_series)

    # Load and run time-series model on the aggregated data.
    ts_predictions = None #timeseries.create_test_time_series(0, 1000, 1001)    # TEST
    try:
        print_and_log("Time-series model loading and executing.")
        ts_model = timeseries_model.load(config.get("input").get("ts_model"))
        ts_predictions = ts_predict(ts_model, time_series)
        print_and_log("Time-series model finished running.")
    except Exception as ex:
        print_and_log(f"WARNING: Could not load or run time-series model: {str(ex)}")
        raise ex

    # Calculate metrics and return combined results.
    metric_results = calculate_metrics(time_series, ts_predictions, config)
    metric_results = add_accuracy(metric_results, accuracy)
    return metric_results


def save_predictions(full_dataset, predictions, output_filename, reference_dataset=None):
    """Saves the ids, predictions and metrics into a JSON file."""
    print_and_log("Creating predictions DataFrame")
    if reference_dataset:
        output_df = reference_dataset.as_dataframe()
    else:
        output_df = full_dataset.as_dataframe(include_all_data=False)

    predictions.save_to_file(output_filename, output_df)


def save_metrics(metrics, metrics_filename):
    """Stores the given metrics to an output."""
    if len(metrics) == 0:
        print_and_log("No metrics to store to file.")
        return

    print_and_log("Saving metrics to JSON file")
    with open(metrics_filename, "w") as outfile:
        json.dump(metrics, outfile, indent=4)
    print_and_log("Finished saving JSON file")


def save_updated_dataset(updated_dataset, predictions, output_filename):
    """Saves a dataset to a JSON file, adding the given predictions first."""
    updated_dataset.set_output(predictions)
    updated_dataset.save_to_file(output_filename)


def package_results(config, packaged_folder_base, log_file_name):
    """Copies all results to a date-time folder to store experiment results."""
    dataset = config.get("input").get("dataset")
    model = config.get("input").get("model")
    ts_model = config.get("input").get("ts_model")
    predictions = config.get("output").get("predictions_output")
    metrics = config.get("output").get("metrics_output")

    # Create time-stamped folder to store results.
    print_and_log("Storing exp results in folder.")
    exp_descriptor = os.path.splitext(os.path.basename(config.config_filename))[0]
    package_folder_name = "exp-" + exp_descriptor + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    full_folder_path = os.path.join(packaged_folder_base, package_folder_name)
    if os.path.exists(full_folder_path):
        shutil.rmtree(full_folder_path)
    os.makedirs(full_folder_path)

    # Create subfolder for config.
    config_folder = os.path.join(full_folder_path, "predictor_config")
    os.makedirs(config_folder)

    # Copy all files to the folder to be zipped.
    shutil.copy(config.config_filename, config_folder)
    shutil.copy(dataset, full_folder_path)
    shutil.copytree(model, os.path.join(full_folder_path, os.path.basename(os.path.normpath(model))))
    shutil.copy(ts_model, full_folder_path)
    shutil.copy(predictions, full_folder_path)
    shutil.copy(metrics, full_folder_path)
    shutil.copy(log_file_name, full_folder_path)

    # Create zip as well.
    print_and_log("Storing exp results in zip file.")
    shutil.make_archive(full_folder_path, "zip", full_folder_path)
