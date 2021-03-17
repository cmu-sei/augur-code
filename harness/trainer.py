
import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config

CONFIG_FILENAME = "./trainer_config.json"


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def split_data(x_all, x_angle_all, y_all):
    # Split training set into train and validation (75% to actually train)
    x_train, x_valid, x_angle_train, x_angle_valid, y_train, y_valid = train_test_split(x_all, x_angle_all, y_all,
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


def show_results(history):
    fig = plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('my_figure_1.png')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('my_figure_2.png')


# Main code.
def main():
    np.random.seed(666)

    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

    # Load and split data.
    [x_ids, x_all, x_angle_all, y_all] = augur_dataset.load_data(config.get("dataset"))
    [x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid] = split_data(x_all, x_angle_all, y_all)

    # Prepare model.
    model = augur_model.create_model()
    model.summary()

    # Train.
    history = train(model, x_train, x_angle_train, y_train, x_valid, x_angle_valid, y_valid)

    # Save trained model.
    augur_model.save_model_to_file(model, config.get("output"))
    #show_results(history)


if __name__ == '__main__':
    main()
