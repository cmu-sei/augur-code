import sys

import numpy as np
import sklearn.model_selection as skm

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


def split_data(x_all, x_angle_all, y_all):
    # Split training set into train and validation (75% to actually train)
    x_train, x_valid, x_angle_train, x_angle_valid, y_train, y_valid = skm.train_test_split(x_all, x_angle_all, y_all,
                                                                                        random_state=123,
                                                                                        train_size=0.75)
    print("Done splitting validation data from train data", flush=True)
    return [x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid]


def fit(model, x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid):
    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    history = model.fit([x_train, x_angle_train], y_train, epochs=20,
                        validation_data=([x_valid, x_angle_valid], y_valid),
                        batch_size=32,
                        callbacks=callbacks)
    print("Done training!", flush=True)
    return history


def train(config):
    # Load and split data.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("dataset"))
    [x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid] = split_data(dataset.x_combined_bands,
                                                                                    dataset.x_angle,
                                                                                    dataset.y_results)

    # Prepare model.
    model = augur_model.create_model()
    model.summary()

    # Train.
    history = fit(model, x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid)
    #plotter.show_results(history)

    # Save trained model.
    augur_model.save_model_to_file(model, config.get("model"))


def evaluate(config):
    # Load evaluation dataset.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("test"))

    model = augur_model.load_model_from_file(config.get("model"))
    print("Evaluate on test data", flush=True)
    results = model.evaluate([dataset.x_combined_bands, dataset.x_angle], dataset.y_results, batch_size=128)
    print("test loss, test acc:", results, flush=True)


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
