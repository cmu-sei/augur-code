import sys

import numpy as np
import sklearn.model_selection as skm
from sklearn.model_selection import KFold

import tensorflow.keras.callbacks as tfcb

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config
#from utils import plotter

CONFIG_FILENAME = "./trainer_config.json"


def get_callbacks(filepath, patience=2):
    es = tfcb.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = tfcb.ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


# Split training set into train and validation (75% to actually train)
def split_data(x_band, x_angle, y_all):
    x_band_t, x_band_v, x_angle_t, x_angle_v, y_train, y_validation = skm.train_test_split(x_band, x_angle, y_all, random_state=123, train_size=0.75)
    print("Done splitting validation data from train data", flush=True)
    x_train = [x_band_t, x_angle_t]
    x_validation = [x_band_v, x_angle_v]
    return [x_train, y_train, x_validation, y_validation]


def fit(model, x_train, y_train, x_validation, y_validation):
    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    history = model.fit(x_train, y_train, epochs=20,
                        validation_data=(x_validation, y_validation),
                        batch_size=32,
                        callbacks=callbacks)
    print("Done training!", flush=True)
    return history


def train(config):
    # Load and split data.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("dataset"))
    [x_train, y_train, x_validation, y_validation] = split_data(dataset.x_combined_bands, dataset.x_angle, dataset.y_output)

    # Prepare model.
    model = augur_model.create_model()
    model.summary()

    # Train.
    history = fit(model, x_train, y_train, x_validation, y_validation)
    #plotter.show_results(history)

    # Save trained model.
    augur_model.save_model_to_file(model, config.get("model"))

    # Cross-validate.
    #cross_validate(config, 5, dataset.get_full_input(), dataset.y_output)


def evaluate(config):
    # Load evaluation dataset.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("test"))

    model = augur_model.load_model_from_file(config.get("model"))
    print("Evaluate on test data", flush=True)
    results = model.evaluate([dataset.x_combined_bands, dataset.x_angle], dataset.y_output, batch_size=128)
    print("test loss, test acc:", results, flush=True)


# k-fold cross-validation to check how model is performing by selecting different sets to train/validate.
def cross_validate(config, num_folds, inputs, targets):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        model = augur_model.load_model_from_file(config.get("model"))

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=32,
                            epochs=20)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        #acc_per_fold.append(scores[1] * 100)
        #loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1


# Main code.
def main():
    np.random.seed(555)

    # See if we'll use the default or a special config file.
    config_file = CONFIG_FILENAME
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        print('Config file top use: ', config_file)

    # Load config.
    config = Config()
    config.load(config_file)

    mode = config.get("mode")
    if mode == "train":
        train(config)
    else:
        evaluate(config)


if __name__ == '__main__':
    main()
