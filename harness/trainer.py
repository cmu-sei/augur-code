import logging

import numpy as np

from training import model_utils
from utils.config import Config
from utils import logging
from utils import arguments
from utils.logging import print_and_log
from datasets import dataset
from training.model_trainer import ModelTrainer
from analysis.timeseries import TimeSeries
import training.timeseries_model as timeseries_model

LOG_FILE_NAME = "training.log"
DEFAULT_CONFIG_FILENAME = "./trainer_config.json"
RANDOM_SEED = 555


def aggregate_data(dataset_instance, time_interval_params):
    """Aggregate the dataset data into a time series."""
    time_series = TimeSeries()
    if dataset_instance.has_timestamps():
        # If the dataset comes with timestamps, aggregate using them.
        time_series.aggregate_by_timestamp(time_interval_params.get("starting_interval"),
                                           time_interval_params.get("interval_unit"),
                                           dataset_instance.get_output(),
                                           dataset_instance.get_timestamps())
    else:
        # If the dataset doesn't have timestamps, use the samples_per_interval config to aggregate.
        samples_per_interval = int(time_interval_params.get("samples_per_time_interval"))
        time_series.aggregate_by_number_of_samples(time_interval_params.get("starting_interval"),
                                                   time_interval_params.get("interval_unit"),
                                                   dataset_instance.get_output(),
                                                   samples_per_interval)
    return time_series


# Main code.
def main():
    np.random.seed(RANDOM_SEED)
    logging.setup_logging(LOG_FILE_NAME)

    # Parse args and load config.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, "./", DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    print_and_log("--------------------------------------------------------------------")
    print_and_log("Starting trainer session.")

    # Loading model and dataset.
    main_model_module = dataset.load_model_module(config.get("model_module"))
    main_trainer = ModelTrainer(main_model_module, config.get("hyper_parameters"))

    dataset_class = dataset.load_dataset_class(config.get("dataset_class"))
    dataset_instance = dataset_class()
    dataset_instance.load_from_file(config.get("dataset"))

    # Run steps depending on config.
    if config.get("training") == "on":
        trained_model = main_trainer.split_and_train(dataset_instance)
        model_utils.save_model_to_file(trained_model, config.get("model"))
    if config.get("cross_validation") == "on":
        main_trainer.cross_validate(dataset_instance)
    if config.get("evaluation") == "on":
        trained_model = model_utils.load_model_from_file(config.get("model"))
        default_evaluation_input = dataset_instance.get_model_input()
        default_evaluation_output = dataset_instance.get_output()
        main_trainer.evaluate(trained_model, default_evaluation_input, default_evaluation_output)
    if config.get("time_series_training") == "on":
        time_interval_params = config.get("time_interval")
        time_series = aggregate_data(dataset_instance, time_interval_params)
        print_and_log(f"Finished aggregating data, number of aggregated intervals: {time_series.get_num_intervals()}")

        print_and_log(f"Training time-series model")
        trained_model = timeseries_model.create_fit_model(time_series.get_time_intervals(),
                                                          time_series.get_aggregated(),
                                                          time_interval_params.get("interval_unit"),
                                                          config.get("ts_hyper_parameters"))

        print_and_log(f"Finished training time-series model, saving it now.")
        timeseries_model.save(trained_model, config.get("ts_model"))

    print_and_log("Finished trainer session.")
    print_and_log("--------------------------------------------------------------------")


if __name__ == '__main__':
    main()
