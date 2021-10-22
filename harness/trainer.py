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
TIMESERIES_MODEL_MODULE = "timeseries_model"
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
    ts_model_module = dataset.load_model_module(TIMESERIES_MODEL_MODULE)
    main_trainer = ModelTrainer(main_model_module, CONFIG.get("hyper_parameters"))
    ts_trainer = ModelTrainer(ts_model_module, CONFIG.get("ts_hyper_parameters"))

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
        time_series = TimeSeries()
        time_series.aggregate_by_number_of_samples(dataset_instance, dataset_instance.get_output(),
                                                   CONFIG.get("ts_hyper_parameters").get("samples_per_time"))
        trained_model, history = ts_trainer.train(timeseries_model.create_model(), training_set)
        model_utils.save_model_to_file(trained_model, CONFIG.get("ts_model"))
    if CONFIG.get("time_series_evaluation") == "on":
        model = model_utils.load_model_from_file(CONFIG.get("ts_model"))
        ts_trainer.evaluate(model)

    print_and_log("Finished trainer session.")
    print_and_log("--------------------------------------------------------------------")


if __name__ == '__main__':
    main()
