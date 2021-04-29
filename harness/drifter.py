import importlib

from utils import dataset as augur_dataset
from utils.config import Config
from utils import databin

CONFIG_FILENAME = "./drifter_config.json"


# Applies drift on a given dataset
def apply_drift(input_dataset, drift_config):
    print("Drift condition: " + drift_config.get("condition"), flush=True)

    # Import module dynamically, and call the drift generation method
    drift_module = importlib.import_module(drift_config.get("method"))
    drifted_dataset = drift_module.generate_drift(input_dataset, drift_config.get("params"))

    print("Finished applying drift", flush=True)
    return drifted_dataset


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
        drifted_dataset = apply_drift(base_dataset, config.get("drift_scenario"))

        # Save drift dataset.
        drifted_dataset.save_by_reference(config.get("output"))
    except ModuleNotFoundError:
        print("Could not find module implemeting drift algorithm: " + config.get("drift_scenario").get("method") + ". Aborting.")


if __name__ == '__main__':
    main()
