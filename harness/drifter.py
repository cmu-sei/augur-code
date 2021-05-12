import sys
import importlib

from utils import dataset as augur_dataset
from utils.config import Config
from utils import databin

DEFAULT_CONFIG_FILENAME = "./drifter_config.json"
DRIFT_EXP_CONFIG_FOLDER = "../experiments/drift"


def apply_drift(input_bins, drift_config):
    """Applies drift on a given dataset"""

    print("Drift condition: " + drift_config.get("condition"), flush=True)
    print("Drift function: " + drift_config.get("method"), flush=True)
    params = drift_config.get("params")
    print("Generating drift with params: ")
    print(params)

    # Import module dynamically.
    drift_module = importlib.import_module(drift_config.get("method"))

    max_num_samples = params.get("max_num_samples")
    timebox_size = params.get("timebox_size")
    drifted_dataset = augur_dataset.DataSet()

    # Loop until we get all samples we want.
    curr_bin_offset = 0
    while drifted_dataset.get_number_of_samples() < max_num_samples:
        print(f"Now getting data for timebox of size {timebox_size}, using bin offset {curr_bin_offset}")
        timebox_sample_ids = generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params)
        drifted_dataset.add_multiple_by_reference(timebox_sample_ids)

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
    print("Finished applying drift", flush=True)
    return drifted_dataset


def generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params):
    """Chooses samples for a given timebox size, and from the given current bin index."""

    # Get all values for this timebox.
    timebox_sample_ids = []
    for sample_index in range(0, timebox_size):
        bin_idx = drift_module.get_bin_index(sample_index, curr_bin_offset, len(input_bins), params)
        print(f"Selecting from bin {bin_idx}")
        curr_bin = input_bins[bin_idx]

        if curr_bin.get_queue_length() == 0:
            print(f"No more items in queue, resetting it.")
            curr_bin.setup_queue()
        next_sample_id = curr_bin.pop_from_queue()
        timebox_sample_ids.append(next_sample_id)
    return timebox_sample_ids


# Main code.
def main():
    # Allow selecting configs for experiments, and load it.
    config_file = Config.choose_from_folder(sys.argv, DRIFT_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to drift.
    base_dataset = augur_dataset.DataSet()
    base_dataset.load_data(config.get("dataset"))

    # Sort into bins.
    bin_info_list = config.get("bins")
    print(f"Bins: {bin_info_list}", flush=True)
    bins = databin.create_bins(bin_info_list)
    bins = databin.sort_into_bins(base_dataset.x_ids, base_dataset.y_output, bins)
    print("Filled bins: ")
    for bin in bins:
        print(f" - {bin.info()}")
        bin.setup_queue()

    # Apply drift.
    try:
        drifted_dataset = apply_drift(bins, config.get("drift_scenario"))
        drifted_dataset.save_by_reference(config.get("output"))
    except ModuleNotFoundError:
        print("Could not find module implementing drift algorithm: " + config.get("drift_scenario").get("method") + ". Aborting.")


if __name__ == '__main__':
    main()
