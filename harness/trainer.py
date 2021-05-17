import sys
import logging
import importlib

import numpy as np
from sklearn.model_selection import KFold
import tensorflow.keras.callbacks as tfcb

from utils import model_utils
from utils.config import Config
from utils import logging
from utils.logging import print_and_log
from datasets import dataset
# from utils import plotter

CONFIG_FILENAME = "./trainer_config.json"
CONFIG = Config()

MODEL_MODULE = None


def get_callbacks(patience=2):
    """Gets helper callbacks to save checkpoints and allow early stopping when needed."""
    file_path = ".model_weights.hdf5"
    es = tfcb.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = tfcb.ModelCheckpoint(file_path, save_best_only=True)
    return [es, msave]


def train(training_set):
    """Train."""
    print_and_log("TRAINING")

    model = MODEL_MODULE.create_model()
    # model.summary()

    epochs = CONFIG.get("hyper_parameters").get("epochs")
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    print_and_log(f'Starting training with hyper parameters: epochs: {epochs}, batch size: {batch_size}')

    validation_data = None
    callbacks = None
    if training_set.has_validation():
        print_and_log("Validation data found")
        validation_data = (training_set.x_validation, training_set.y_validation)
        callbacks = get_callbacks(patience=5)

    history = model.fit(training_set.x_train, training_set.y_train,
                        epochs=epochs,
                        validation_data=validation_data,
                        batch_size=batch_size,
                        callbacks=callbacks)
    print_and_log(f'Final training result ({len(history.history.get("loss"))} epochs): '
                  f'loss: {history.history.get("loss")[-1]}, '
                  f'accuracy: {history.history.get("accuracy")[-1]}')
    if training_set.has_validation():
        print_and_log(f'Validation: val_loss: {history.history.get("val_loss")[-1]}, '
                      f'val_accuracy: {history.history.get("val_accuracy")[-1]}')

    print("Done training!", flush=True)
    # plotter.show_results(history)

    return model, history


def evaluate(model, x_inputs, y_inputs):
    """Does an evaluation."""
    # Load evaluation dataset and model.
    print_and_log("EVALUATION")
    print("Starting evaluation", flush=True)
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    scores = model.evaluate(x_inputs, y_inputs, batch_size=batch_size)
    print(f'Done! Evaluation loss and acc: {scores}')
    return scores


def cross_validate(full_dataset, num_folds=5):
    """k-fold cross-validation to check how model is performing by selecting different sets to train/validate."""

    # Define the K-fold Cross Validator
    print_and_log("CROSS VALIDATION")
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1
    for train_index, test_index in kfold.split(full_dataset.get_single_input(), full_dataset.get_output()):
        # Generate a print
        print('------------------------------------------------------------------------')
        print_and_log(f'Training for fold {fold_no} ...')

        training_set = MODEL_MODULE.get_fold_data(full_dataset, train_index, test_index)

        # Fit data to model
        print_and_log(f'Training fold samples: {training_set.num_train_samples}')
        model, history = train(training_set)

        # Generate generalization metrics
        print_and_log(f'Evaluation fold samples: {training_set.num_validation_samples}')
        scores = evaluate(model, training_set.x_validation, training_set.y_validation)
        print_and_log(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
                      f'{model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print_and_log("Done with cross-validation!")


# Main code.
def main():
    np.random.seed(555)
    logging.setup_logging("training.log")

    # See if we'll use the default or a special config file.
    config_file = CONFIG_FILENAME
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        print('Config file top use: ', config_file)

    # Load config.
    CONFIG.load(config_file)

    # Loading model and dataset
    global MODEL_MODULE
    MODEL_MODULE = dataset.load_model_module(CONFIG.get("model_module"))

    print_and_log("--------------------------------------------------------------------")
    print_and_log("Starting trainer session.")

    dataset_instance = dataset.create_dataset_class(CONFIG.get("dataset_class"))
    dataset_instance.load_from_file(CONFIG.get("dataset"))
    evaluation_input = dataset_instance.get_model_input()
    evaluation_output = dataset_instance.get_output()

    # Run steps depending on config.
    if CONFIG.get("training") == "on":
        training_set = MODEL_MODULE.split_data(dataset_instance, CONFIG.get("hyper_parameters").get("validation_size"))
        print_and_log(f'Dataset samples {dataset_instance.get_number_of_samples()}, '
                      f'training samples: {len(training_set.x_train[0])}, '
                      f'validation samples: {len(training_set.x_validation[0])}')

        model, history = train(training_set)
        model_utils.save_model_to_file(model, CONFIG.get("model"))
        evaluation_input = training_set.x_validation
        evaluation_output = training_set.y_validation
    if CONFIG.get("cross_validation") == "on":
        cross_validate(dataset_instance)
    if CONFIG.get("evaluation") == "on":
        model = model_utils.load_model_from_file(CONFIG.get("model"))
        evaluate(model, evaluation_input, evaluation_output)

    print_and_log("Finished trainer session.")
    print_and_log("--------------------------------------------------------------------")


if __name__ == '__main__':
    main()
