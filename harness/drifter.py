import shutil

from drift import drift_generator
from utils import arguments
from utils.config import Config
from utils import logging


DEFAULT_CONFIG_FILENAME = "./drifter_config.json"
DRIFT_EXP_CONFIG_FOLDER = "../experiments/drifter"


def main():
    logging.setup_logging("drifter.log")

    # Allow selecting configs for experiments, and load it.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, DRIFT_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load scenario data.
    drift_module, params = drift_generator.load_drift_config(config.get("drift_scenario"))

    if args.test:
        drift_generator.test_drift(config, drift_module, params, config.get("bins"))
    else:
        # Sort dataset into bins.
        base_dataset = drift_generator.load_dataset(config.get("dataset"), config.get("dataset_class"))
        bin_value = config.get("bin_value") if config.contains("bin_value") else "results"
        bin_shuffle = config.get("bin_shuffle") if config.contains("bin_shuffle") else True
        bins = drift_generator.load_bins(base_dataset, config.get("bins"), bin_value, bin_shuffle)

        # Apply drift.
        drifted_dataset = drift_generator.apply_drift(bins, drift_module, params)
        drift_generator.add_timestamps(base_dataset, drifted_dataset, config.get("timestamps"))

        # Save it to regular file, and timestamped file.
        drifted_dataset.save_to_file(config.get("output"))
        print("Copying output file to timestamped backup.")
        shutil.copyfile(config.get("output"), drift_generator.get_drift_stamped_name(config.get("output")))


if __name__ == '__main__':
    main()
