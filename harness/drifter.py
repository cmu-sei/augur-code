import importlib

from utils import dataset as augur_dataset
from utils.config import Config
from utils import databin

CONFIG_FILENAME = "./drifter_config.json"


# Applies drift on a given dataset
def apply_drift(input_bins, drift_config):
    print("Drift condition: " + drift_config.get("condition"), flush=True)
    print("Drift function: " + drift_config.get("method"), flush=True)
    params = drift_config.get("params")
    print("Generating drift with params: ")
    print(params)

    # Import module dynamically.
    drift_module = importlib.import_module(drift_config.get("method"))

    # Loop until we get all samples we want.
    max_num_samples = params.get("max_num_samples")
    timebox_size = params.get("timebox_size")
    drifted_dataset = augur_dataset.DataSet()
    curr_bin_offset = 0
    while drifted_dataset.get_number_of_samples() < max_num_samples:
        print(f"Now getting data for timebox of size {timebox_size}, using bin offset {curr_bin_offset}")
        timebox_sample_ids = generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params)
        drifted_dataset.add_multiple_by_reference(timebox_sample_ids)

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
    print("Finished applying drift", flush=True)
    return drifted_dataset


# Chooses samples for a given timebox size, and from the given current bin index.
def generate_timebox_samples(drift_module, curr_bin_offset, input_bins, timebox_size, params):
    # Get all values for this timebox.
    timebox_sample_ids = []
    for sample_index in range(0, timebox_size):
        bin_idx = drift_module.get_bin_index(sample_index, curr_bin_offset, len(input_bins), params)
        print(f"Selecting from bin {bin_idx}")
        next_sample_id = input_bins[bin_idx].get_random()
        timebox_sample_ids.append(next_sample_id)
    return timebox_sample_ids


# Main code.
def main():
    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

    # Load dataset to drift.
    base_dataset = augur_dataset.DataSet()
    base_dataset.load_data(config.get("dataset"))

    # Sort into bins.
    bins = databin.sort_into_bins(base_dataset.x_ids, base_dataset.y_output, ["no_iceberg", "iceberg"], [0, 1])
    print("Bins: ")
    for bin in bins:
        print(bin.info())

    # Apply drift.
    try:
        drifted_dataset = apply_drift(bins, config.get("drift_scenario"))
        drifted_dataset.save_by_reference(config.get("output"))
    except ModuleNotFoundError:
        print("Could not find module implemeting drift algorithm: " + config.get("drift_scenario").get("method") + ". Aborting.")


if __name__ == '__main__':
    main()
