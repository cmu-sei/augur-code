import importlib
import random
import datetime
import os
import shutil
import time

from datasets import ref_dataset
from utils import arguments
from utils.config import Config
from utils import databin
from utils import logging
from utils.logging import print_and_log
from training import predictions
from datasets import dataset

DEFAULT_CONFIG_FILENAME = "./drifter_config.json"
DRIFT_EXP_CONFIG_FOLDER = "../experiments/drifter"


def load_bins(dataset_filename, dataset_class_name, bin_params, predictions_filename=None, bin_value="results", shuffle=True):
    """Loads a dataset into bins"""

    # Load dataset to drift.
    dataset_class = dataset.load_dataset_class(dataset_class_name)
    base_dataset = dataset_class()
    base_dataset.load_from_file(dataset_filename)

    # If present, load predictions info.
    if predictions_filename is not None:
        preds = predictions.Predictions()
        preds.load_from_file(predictions_filename)
        # TODO: Connect predictions with full dataset. Assumption right now is that samples are in the same order.

    # Sort into bins.
    print_and_log(f"Bins: {bin_params}")
    values = get_bin_values(base_dataset, bin_value)
    bins = databin.create_bins(bin_params, shuffle)
    bins = databin.sort_into_bins(base_dataset.get_ids(), values, bins)

    # Setup queues.
    print_and_log("Filled bins: ")
    for bin in bins:
        print_and_log(f" - {bin.info()}")
        bin.setup_queue()

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


def load_drift_config(drift_config):
    """Loads the drift module and params from the drift configuration."""
    print_and_log("Drift condition: " + drift_config.get("condition"))
    print_and_log("Drift module: " + drift_config.get("module"))
    params = drift_config.get("params")
    print_and_log("Drift  params: ")
    print_and_log(params)

    # Import module dynamically.
    try:
        drift_module = importlib.import_module("drifts." + drift_config.get("module"))
    except ModuleNotFoundError:
        print_and_log("Could not find module implementing drift algorithm: " + drift_config.get("module") +
                      ". Aborting.")
        exit(1)

    return drift_module, params


def apply_drift(input_bins, drift_module, params):
    """Applies drift on a given dataset"""
    print("Applying drift to generated drifted dataset.")
    max_num_samples = params.get("max_num_samples")
    timebox_size = params.get("timebox_size")
    shuffle_on = params.get("timebox_shuffle") if "timebox_shuffle" in params else True

    # Loop until we get all samples we want.
    drifted_dataset = ref_dataset.RefDataSet()
    curr_bin_offset = 0
    timebox_id = 0
    while drifted_dataset.get_number_of_samples() < max_num_samples:
        print_and_log(f"Now getting data for timebox of size {timebox_size}, using bin offset {curr_bin_offset}")
        timebox_sample_ids = generate_timebox_samples(drift_module, timebox_id, curr_bin_offset, input_bins, timebox_size, params)

        # Randomize results in timebox to avoid stacking bin results at the end.
        if shuffle_on:
            print_and_log("Shuffling timebox samples.")
            random.shuffle(timebox_sample_ids)
        drifted_dataset.add_multiple_references(timebox_sample_ids, timebox_id)

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
        timebox_id += 1
    print_and_log("Finished applying drift")
    return drifted_dataset


def add_timestamps(drifted_dataset, timestamp_params):
    """Adds sequential timestamps to a dataset."""
    enabled = timestamp_params.get("enabled")
    if not enabled:
        print_and_log("Timestamps not enabled.")
        return

    start_datetime = time.strptime(timestamp_params.get("start_datetime"), '%Y-%m-%d')
    increment_in_days = timestamp_params.get("increment")
    num_samples = drifted_dataset.get_number_of_samples()
    DAY_TO_SECONDS = 24 * 60 * 60

    # Generate sequential timestamps for as many samples as we have, with the given start time and increment.
    timestamps = [0] * num_samples
    timestamps[0] = time.mktime(start_datetime)
    for i in range(1, num_samples):
        timestamps[i] = timestamps[i - 1] + (increment_in_days * DAY_TO_SECONDS)

    drifted_dataset.set_timestamps(timestamps)
    print_and_log("Generated and stored timestamps.")


def generate_timebox_samples(drift_module, timebox_id, curr_bin_offset, input_bins, timebox_size, params):
    """Chooses samples for a given timebox size, and from the given current bin index."""

    # Get all values for this timebox.
    timebox_sample_ids = []
    for sample_index in range(0, timebox_size):
        bin_idx = drift_module.get_bin_index(sample_index, timebox_id, curr_bin_offset, len(input_bins), params)
        print_and_log(f"Selecting from bin {bin_idx}")
        curr_bin = input_bins[bin_idx]

        if curr_bin.get_queue_length() == 0:
            print_and_log(f"No more items in queue, resetting it.")
            curr_bin.setup_queue()
        next_sample_id = curr_bin.pop_from_queue()
        timebox_sample_ids.append(next_sample_id)
    return timebox_sample_ids


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


# Main code.
def main():
    logging.setup_logging("drifter.log")

    # Allow selecting configs for experiments, and load it.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, DRIFT_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load scenario data.
    drift_module, params = load_drift_config(config.get("drift_scenario"))

    if args.test:
        test_drift(config, drift_module, params, config.get("bins"))
    else:
        # Check optional predictions input.
        full_predictions = None
        if config.contains("predictions"):
            full_predictions = config.get("predictions")

        # Apply drift.
        bin_value = config.get("bin_value") if config.contains("bin_value") else "results"
        bin_shuffle = config.get("bin_shuffle") if config.contains("bin_shuffle") else True
        bins = load_bins(config.get("dataset"), config.get("dataset_class"), config.get("bins"), full_predictions, bin_value, bin_shuffle)
        drifted_dataset = apply_drift(bins, drift_module, params)
        add_timestamps(drifted_dataset, config.get("timestamps"))

        # Save it to regular file, and timestamped file.
        drifted_dataset.save_to_file(config.get("output"))
        print("Copying output file to timestamped backup.")
        shutil.copyfile(config.get("output"), get_drift_stamped_name(config.get("output")))


if __name__ == '__main__':
    main()
