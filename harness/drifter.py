import sys
import importlib

from datasets import ref_dataset
from utils.config import Config
from utils import databin
from utils import logging
from utils.logging import print_and_log
from datasets import dataset

DEFAULT_CONFIG_FILENAME = "./drifter_config.json"
DRIFT_EXP_CONFIG_FOLDER = "../experiments/drift"


def load_bins(dataset_filename, dataset_class_name, bin_params):
    """Loads a dataset into bins"""

    # Load dataset to drift.
    base_dataset = dataset.create_dataset_class(dataset_class_name)
    base_dataset.load_from_file(dataset_filename)

    # Sort into bins.
    print_and_log(f"Bins: {bin_params}")
    bins = databin.create_bins(bin_params)
    bins = databin.sort_into_bins(base_dataset.x_ids, base_dataset.y_output, bins)
    print_and_log("Filled bins: ")
    for bin in bins:
        print_and_log(f" - {bin.info()}")
        bin.setup_queue()

    return bins


def apply_drift(input_bins, drift_config):
    """Applies drift on a given dataset"""

    print_and_log("Drift condition: " + drift_config.get("condition"))
    print_and_log("Drift function: " + drift_config.get("method"))
    params = drift_config.get("params")
    print_and_log("Generating drift with params: ")
    print_and_log(params)

    # Import module dynamically.
    drift_module = importlib.import_module(drift_config.get("method"))

    max_num_samples = params.get("max_num_samples")
    timebox_size = params.get("timebox_size")
    drifted_dataset = ref_dataset.RefDataSet()

    # Loop until we get all samples we want.
    curr_bin_offset = 0
    while drifted_dataset.get_number_of_samples() < max_num_samples:
        print_and_log(f"Now getting data for timebox of size {timebox_size}, using bin offset {curr_bin_offset}")
        timebox_sample_ids = generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params)
        drifted_dataset.add_multiple_references(timebox_sample_ids)

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
    print_and_log("Finished applying drift")
    return drifted_dataset


def generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params):
    """Chooses samples for a given timebox size, and from the given current bin index."""

    # Get all values for this timebox.
    timebox_sample_ids = []
    for sample_index in range(0, timebox_size):
        bin_idx = drift_module.get_bin_index(sample_index, curr_bin_offset, len(input_bins), params)
        print_and_log(f"Selecting from bin {bin_idx}")
        curr_bin = input_bins[bin_idx]

        if curr_bin.get_queue_length() == 0:
            print_and_log(f"No more items in queue, resetting it.")
            curr_bin.setup_queue()
        next_sample_id = curr_bin.pop_from_queue()
        timebox_sample_ids.append(next_sample_id)
    return timebox_sample_ids


# Main code.
def main():
    logging.setup_logging("drifter.log")

    # Allow selecting configs for experiments, and load it.
    config_file = Config.choose_from_folder(sys.argv, DRIFT_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Apply drift.
    try:
        bins = load_bins(config.get("dataset"), config.get("dataset_class"), config.get("bins"))
        drifted_dataset = apply_drift(bins, config.get("drift_scenario"))
        drifted_dataset.save_to_file(config.get("output"))
    except ModuleNotFoundError:
        print_and_log("Could not find module implementing drift algorithm: " + config.get("drift_scenario").get("method") +
              ". Aborting.")


if __name__ == '__main__':
    main()
