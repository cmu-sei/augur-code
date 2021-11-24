from training import model_utils
from utils.config import Config
from utils import arguments
from utils import logging
from utils.logging import print_and_log
from analysis import analyzer
from datasets import dataset
from datasets import ref_dataset

LOG_FILE_NAME = "predictor.log"
DEFAULT_CONFIG_FILENAME = "./predictor_config.json"
METRIC_EXP_CONFIG_FOLDER = "../experiments/predictor"
PACKAGED_FOLDER_BASE = "../output/packaged/"


def load_datasets(input_config):
    """Based on the config, loads the dataset to use, and the reference one if needed."""
    dataset_class = dataset.load_dataset_class(input_config.get("dataset_class"))
    reference_dataset = None
    if "base_dataset" in input_config:
        full_dataset, reference_dataset = ref_dataset.load_full_from_ref_and_base(dataset_class, input_config.get("dataset"), input_config.get("base_dataset"))
    else:
        full_dataset = dataset_class()
        full_dataset.load_from_file(input_config.get("dataset"))

    return full_dataset, reference_dataset


# Main code.
def main():
    logging.setup_logging(LOG_FILE_NAME)

    # Allow selecting configs for experiments, and load it.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, METRIC_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to predict on (and base one if needed).
    full_dataset, reference_dataset = load_datasets(config.get("input"))

    # Load model.
    model = model_utils.load_model_from_file(config.get("input").get("model"))
    model.summary()

    # Predict.
    predictions = analyzer.predict(model, full_dataset.get_model_input(), config.get("threshold"))

    # Analyze and/or save as needed.
    mode = config.get("mode")
    if mode == "analyze":
        # Store the predictions to a file.
        predictions.store_expected_results(full_dataset.get_output())
        analyzer.save_predictions(full_dataset, predictions, config.get("output").get("predictions_output"), reference_dataset)

        # Calculate metrics and store to file.
        metric_results = analyzer.analyze(full_dataset, predictions, config)
        analyzer.save_metrics(metric_results, config.get("output").get("metrics_output"))

        # If requested, package this experiment's results.
        if args.store:
            analyzer.package_results(config, PACKAGED_FOLDER_BASE, LOG_FILE_NAME)
    elif mode == "label":
        analyzer.save_updated_dataset(full_dataset, predictions.get_predictions(), config.get("output").get("labelled_output"))
    else:
        print_and_log("Unsupported mode: " + mode)


if __name__ == '__main__':
    main()
