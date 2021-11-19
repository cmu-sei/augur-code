import logging

import numpy as np
from sklearn.model_selection import KFold
import tensorflow.keras.callbacks as tfcb

from training import model_utils
from utils.config import Config
from utils import logging
from utils import arguments
from utils.logging import print_and_log
from datasets import dataset
from datasets.model_trainer import ModelTrainer
from datasets.timeseries import TimeSeries
import datasets.timeseries_model as timeseries_model

DEFAULT_CONFIG_FILENAME = "./trainer_config.json"
CONFIG = Config()


# Main code.
def main():
    np.random.seed(555)
    logging.setup_logging("training.log")

    # Parse args and load config.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, "./", DEFAULT_CONFIG_FILENAME)
    CONFIG.load(config_file)

    # Loading models and dataset
    print_and_log("--------------------------------------------------------------------")
    print_and_log("Starting trainer session.")

    main_model_module = dataset.load_model_module(CONFIG.get("model_module"))
    main_trainer = ModelTrainer(main_model_module, CONFIG.get("hyper_parameters"))

    dataset_class = dataset.load_dataset_class(CONFIG.get("dataset_class"))
    dataset_instance = dataset_class()
    dataset_instance.load_from_file(CONFIG.get("dataset"))

    # Run steps depending on config.
    if CONFIG.get("training") == "on":
        trained_model = main_trainer.split_and_train(dataset_instance)
        model_utils.save_model_to_file(trained_model, CONFIG.get("model"))
    if CONFIG.get("cross_validation") == "on":
        main_trainer.cross_validate(dataset_instance)
    if CONFIG.get("evaluation") == "on":
        trained_model = model_utils.load_model_from_file(CONFIG.get("model"))
        default_evaluation_input = dataset_instance.get_model_input()
        default_evaluation_output = dataset_instance.get_output()
        main_trainer.evaluate(trained_model, default_evaluation_input, default_evaluation_output)
    if CONFIG.get("time_series_training") == "on":
        time_interval_params = CONFIG.get("time_interval")
        time_series = TimeSeries()
        if dataset_instance.has_timestamps():
            # If the dataset comes with timestamps, aggregate using them.
            time_series.aggregate_by_timestamp(time_interval_params.get("starting_interval"),
                                               time_interval_params.get("interval_unit"),
                                               dataset_instance,
                                               dataset_instance.get_output())
        else:
            # If the dataset doesn't have timestamps, use the samples_per_interval config to aggregate.
            samples_per_interval = int(CONFIG.get("time_interval").get("samples_per_time_interval"))
            time_series.aggregate_by_number_of_samples(time_interval_params.get("starting_interval"),
                                                       time_interval_params.get("interval_unit"),
                                                       dataset_instance,
                                                       dataset_instance.get_output(),
                                                       samples_per_interval)
        trained_model = timeseries_model.create_fit_model(time_series.get_time_intervals(),
                                                          time_series.get_aggregated(),
                                                          time_interval_params.get("interval_unit"),
                                                          CONFIG.get("ts_hyper_parameters"))
        timeseries_model.save(trained_model, CONFIG.get("ts_model"))

    print_and_log("Finished trainer session.")
    print_and_log("--------------------------------------------------------------------")


if __name__ == '__main__':
    main()
