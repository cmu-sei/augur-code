
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


def train(model, x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid):
    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    history = model.fit([x_train, x_angle_train], y_train, epochs=20,
                        validation_data=([x_valid, x_angle_valid], y_valid),
                        batch_size=32,
                        callbacks=callbacks)
    print("Done training!", flush=True)
    return history


# Main code.
def main():
    np.random.seed(555)

    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

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
    history = train(model, x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid)
    #plotter.show_results(history)

    # Save trained model.
    augur_model.save_model_to_file(model, config.get("output"))


if __name__ == '__main__':
    main()
