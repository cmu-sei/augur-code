import importlib
import random
import datetime
import os
import pandas

from datasets import dataset
from datasets import ref_dataset
from drift import databin
from utils.logging import print_and_log

DRIFTS_PACKAGE = "extensions.drifts."


def load_drift_config(drift_config):
    """Loads the drift module and params from the drift configuration."""
    print_and_log("Drift condition: " + drift_config.get("condition"))
    print_and_log("Drift module: " + drift_config.get("module"))
    params = drift_config.get("params")
    print_and_log("Drift  params: ")
    print_and_log(params)

    # Import module dynamically.
    try:
        drift_module = importlib.import_module(DRIFTS_PACKAGE + drift_config.get("module"))
    except ModuleNotFoundError:
        print_and_log("Could not find module implementing drift algorithm: " + drift_config.get("module") +
                      ". Aborting.")
        exit(1)

    return drift_module, params


def load_dataset(dataset_filename, dataset_class_name):
    """Load dataset to drift."""
    dataset_class = dataset.load_dataset_class(dataset_class_name)
    base_dataset = dataset_class()
    base_dataset.load_from_file(dataset_filename)
    return base_dataset


def load_bins(base_dataset, bin_params, bin_value="results", shuffle=True):
    """Loads a dataset into bins"""
    # Sort into bins.
    print_and_log(f"Bins: {bin_params}")
    values = get_bin_values(base_dataset, bin_value)
    bins = databin.create_bins(bin_params, shuffle)
    bins = databin.sort_into_bins(base_dataset.get_ids(), values, bins)

    # Setup queues.
    print_and_log("Filled bins: ")
    for curr_bin in bins:
        print_and_log(f" - {curr_bin.info()}")
        curr_bin.setup_queue()

    return bins


def get_bin_values(base_dataset, bin_value):
    """Gets the values to be used when sorting into bins for the given dataset, from the configured options."""
    values = None
    if bin_value == "results":
        values = base_dataset.get_output()
    elif bin_value == "all":
        # We set all values to 0, assuming single bin will also set its value to 0.
        values = [0] * base_dataset.get_number_of_samples()
    else:
        raise Exception(f"Invalid bin value configured: {bin_value}")
    return values


def apply_drift(input_bins, drift_module, params):
    """Applies drift on a given dataset"""
    print("Applying drift to generated drifted dataset.")
    max_num_samples = params.get("max_num_samples")
    sample_group_size = params.get("sample_group_size")
    shuffle_on = params.get("sample_group_shuffle") if "sample_group_shuffle" in params else True

    # Loop until we get all samples we want.
    drifted_dataset = ref_dataset.RefDataSet()
    curr_bin_offset = 0
    sample_group_id = 0
    while drifted_dataset.get_number_of_samples() < max_num_samples:
        print_and_log(f"Now getting data for sample group of size {sample_group_size}, using bin offset {curr_bin_offset}")
        sample_group_sample_ids = generate_sample_group_samples(drift_module, sample_group_id, curr_bin_offset, input_bins, sample_group_size, params)

        # Randomize results in sample group to avoid stacking bin results at the end.
        if shuffle_on:
            print_and_log("Shuffling sample group samples.")
            random.shuffle(sample_group_sample_ids)
        drifted_dataset.add_multiple_references(sample_group_sample_ids, sample_group_id)

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
        sample_group_id += 1
    print_and_log("Finished applying drift")
    return drifted_dataset


def add_timestamps(base_dataset, drifted_dataset, timestamp_params):
    """Adds sequential timestamps to a dataset."""
    enabled = timestamp_params.get("enabled")
    if not enabled:
        print_and_log("Timestamp generation not enabled.")
        return

    num_samples = drifted_dataset.get_number_of_samples()
    start_datetime = pandas.to_datetime(timestamp_params.get("start_datetime"))
    increment_unit = timestamp_params.get("increment_unit")

    # Generate sequential timestamps for as many samples as we have, with the given start time and increment.
    print_and_log(f"Generating timestamps from starting time {start_datetime} and increment unit {increment_unit}")
    timestamps = [0.0] * num_samples
    for i in range(0, num_samples):
        increment = pandas.to_timedelta(i, increment_unit)
        timestamps[i] = pandas.Timestamp(start_datetime + increment).timestamp()

    drifted_dataset.set_timestamps(timestamps)
    print_and_log("Generated and stored timestamps.")


def generate_sample_group_samples(drift_module, sample_group_id, curr_bin_offset, input_bins, sample_group_size, params):
    """Chooses samples for a given sample group size, and from the given current bin index."""

    # Get all values for this sample group.
    sample_group_sample_ids = []
    for sample_index in range(0, sample_group_size):
        bin_idx = drift_module.get_bin_index(sample_index, sample_group_id, curr_bin_offset, len(input_bins), params)
        print_and_log(f"Selecting from bin {bin_idx}")
        curr_bin = input_bins[bin_idx]

        if curr_bin.get_queue_length() == 0:
            print_and_log(f"No more items in queue, resetting it.")
            curr_bin.setup_queue()
        next_sample_id = curr_bin.pop_from_queue()
        sample_group_sample_ids.append(next_sample_id)
    return sample_group_sample_ids


def test_drift(config, drift_module, drift_params, bin_params):
    """Tests an existing drifted dataset based on its expected properties defined in params."""
    print("Testing drifted dataset")
    dataset_class_name = config.get("dataset_class")
    drifted_dataset_file = config.get("output")
    base_dataset_file = config.get("dataset")

    dataset_class = dataset.load_dataset_class(dataset_class_name)
    full_dataset, reference_dataset = ref_dataset.load_full_from_ref_and_base(dataset_class, drifted_dataset_file, base_dataset_file)

    drift_module.test(full_dataset, drift_params, bin_params)


def get_drift_stamped_name(config_filename):
    """Returns a time-stamped name to save the drift to."""
    folder = os.path.dirname(config_filename)
    descriptor = os.path.splitext(os.path.basename(config_filename))[0]
    drift_file_name = "drift-" + descriptor + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return os.path.join(folder, drift_file_name + ".json")
