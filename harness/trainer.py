import sys
import logging

import numpy as np
import sklearn.model_selection as skm
from sklearn.model_selection import KFold

import tensorflow.keras.callbacks as tfcb

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config
#from utils import plotter

CONFIG_FILENAME = "./trainer_config.json"
CONFIG = Config()


def print_and_log(message):
    print(message, flush=True)
    logging.info(message)


def get_callbacks(patience=2):
    file_path = ".model_weights.hdf5"
    es = tfcb.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = tfcb.ModelCheckpoint(file_path, save_best_only=True)
    return [es, msave]


# Split training set into train and validation (75% to actually train)
def split_data(x_band, x_angle, y_all):
    x_band_t, x_band_v, x_angle_t, x_angle_v, y_train, y_validation = skm.train_test_split(x_band, x_angle, y_all, random_state=123, train_size=0.75)
    print("Done splitting validation data from train data", flush=True)
    x_train = [x_band_t, x_angle_t]
    x_validation = [x_band_v, x_angle_v]
    return [x_train, y_train, x_validation, y_validation]


def fit(model, x_train, y_train, x_validation=None, y_validation=None):
    epochs = CONFIG.get("hyper_parameters").get("epochs")
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    print_and_log(f'Starting training with hyper parameters: epochs: {epochs}, batch size: {batch_size}')

    validation_data = None
    callbacks = None
    if x_validation is not None and y_validation is not None:
        validation_data = (x_validation, y_validation)
        callbacks = get_callbacks(patience=5)

    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=validation_data,
                        batch_size=batch_size,
                        callbacks=callbacks)
    print_and_log(f'Final training result ({len(history.history.get("loss"))} epochs): loss: {history.history.get("loss")[-1]}, accuracy: {history.history.get("accuracy")[-1]}')
    if validation_data is not None:
        print_and_log(f'Validation: val_loss: {history.history.get("val_loss")[-1]}, val_accuracy: {history.history.get("val_accuracy")[-1]}')
    return history


def train():
    print_and_log("--------------------------------------------------------------------")
    print_and_log("Starting training session.")

    # Load and split data.
    dataset = augur_dataset.DataSet()
    dataset.load_data(CONFIG.get("dataset"))
    [x_train, y_train, x_validation, y_validation] = split_data(dataset.x_combined_bands, dataset.x_angle, dataset.y_output)
    print(x_train)
    print_and_log(f'Dataset samples {dataset.num_samples}, training samples: {len(x_train[0])}, validation samples: {len(x_validation[0])}')

    # Prepare model.
    model = augur_model.create_model()
    model.summary()

    # Train.
    history = fit(model, x_train, y_train, x_validation, y_validation)
    print("Done training!", flush=True)
    #plotter.show_results(history)

    # Save trained model.
    augur_model.save_model_to_file(model, CONFIG.get("model"))

    # Cross-validate.
    cross_validate(dataset)

    print_and_log("Finished training session.")
    print_and_log("--------------------------------------------------------------------")


def evaluate():
    # Load evaluation dataset.
    dataset = augur_dataset.DataSet()
    dataset.load_data(CONFIG.get("test"))

    model = augur_model.load_model_from_file(CONFIG.get("model"))
    print("Evaluate on test data", flush=True)
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    results = model.evaluate([dataset.x_combined_bands, dataset.x_angle], dataset.y_output, batch_size=batch_size)
    print("test loss, test acc:", results, flush=True)


# k-fold cross-validation to check how model is performing by selecting different sets to train/validate.
def cross_validate(dataset, num_folds=5):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    print_and_log("Starting cross validation.")

    # K-fold Cross Validation model evaluation
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1
    for train_index, test_index in kfold.split(dataset.x_combined_bands, dataset.y_output):
        model = augur_model.create_model()

        # Generate a print
        print('------------------------------------------------------------------------')
        print_and_log(f'Training for fold {fold_no} ...')

        # Fit data to model
        inputs_train = [dataset.x_combined_bands[train_index], dataset.x_angle[train_index]]
        output_train = dataset.y_output[train_index]
        history = fit(model, inputs_train, output_train)

        # Generate generalization metrics
        inputs_val = [dataset.x_combined_bands[test_index], dataset.x_angle[test_index]]
        output_val = dataset.y_output[test_index]
        scores = model.evaluate(inputs_val, output_val, verbose=0)
        print_and_log(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1


# Main code.
def main():
    np.random.seed(555)
    logging.basicConfig(filename='training.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

    # See if we'll use the default or a special config file.
    config_file = CONFIG_FILENAME
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        print('Config file top use: ', config_file)

    # Load config.
    CONFIG.load(config_file)

    # Run depending on mode.
    mode = CONFIG.get("mode")
    if mode == "train":
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()
