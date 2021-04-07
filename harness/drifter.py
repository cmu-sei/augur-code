import importlib

from utils import dataset as augur_dataset
from utils.config import Config

CONFIG_FILENAME = "./drifter_config.json"


# Applies drift on a given dataset
def apply_drift(input_dataset, drift_config):
    print("Drift condition: " + drift_config.get("condition"), flush=True)

    # Import module dynamically, and call the drift generation method
    drift_module = importlib.import_module(drift_config.get("method"))
    drifted_dataset = drift_module.generate_drift(input_dataset, drift_config.get("params"))

    return drifted_dataset


# Main code.
def main():
    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

    # Load dataset to drift.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("dataset"))

    # Apply drift.
    drifted_dataset = apply_drift(dataset, config.get("drift_scenario"))

    # Save drift dataset.
    drifted_dataset.save_by_reference(config.get("output"))


if __name__ == '__main__':
    main()
