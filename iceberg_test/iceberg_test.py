from subprocess import check_output

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def load_data():
    print("Input files: ", flush=True)
    print(check_output(["ls", "./input"]).decode("utf8"), flush=True)

    # Data for each sample has 3 values: band1, band2, inc_angle. band1 and band2 are 2 radar images,
    # each consisting of 75x75 "pixels", floats with a dB unit, obtained by pinging
    # at an angle of inc_angle (band1 and 2 differ on how the response was received).
    print("Loading input files", flush=True)
    train = pd.read_json("./input/train.json")
    test = pd.read_json("./input/test.json")
    print("Done loading data. Train rows: " + str(train.shape[0]) + ", test rows: " + str(test.shape[0]), flush=True)

    print(train.head(1))
    #print(test.head(1))

    # We set the samples with no info on inc_angle to 0 as its value, to simplify.
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    print("Done cleaning up angle", flush=True)

    # Load each image from a one-dimension vector into a 74x75 numpy matrix, for both bands
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

    # Put all training data into X_train (bands 1 and 2), X_angle_train, and y_train.
    # Not sure how colons are useful here, nor why there is a 3rd array which is
    # equal to x_band1 (2*x_band1/2).
    X_train = np.concatenate([
                              x_band1[:, :, :, np.newaxis],
                              x_band2[:, :, :, np.newaxis],
                              ((x_band1+x_band1)/2)[:, :, :, np.newaxis]
                             ],
                             axis=-1)
    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])
    print("Done loading train data into numpy arrays", flush=True)

    # Load each image from a one-dimension vector into a 74x75 numpy matrix, for both bands
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

    # Put all test data into X_test (bands 1 and 2), X_angle_test.
    # Not sure how colons are useful here, nor why there is a 3rd array which is
    # equal to x_band1 (2*x_band1/2).
    X_test = np.concatenate([
                             x_band1[:, :, :, np.newaxis],
                             x_band2[:, :, :, np.newaxis],
                             ((x_band1+x_band1)/2)[:, :, :, np.newaxis]
                            ],
                            axis=-1)
    X_angle_test = np.array(test.inc_angle)
    print("Done loading test data into numpy arrays", flush=True)

    # Split training set into train and validation (75% to actually train)
    X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                        , X_angle_train, y_train, random_state=123, train_size=0.75)
    print("Done splitting validation data from train data", flush=True)

    return [X_train, X_angle_train, y_train, X_valid, X_angle_valid, y_valid, X_test, X_angle_test]


def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation) ((BatchNormalization(momentum=bn_model)) (input_1))
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2, 2)) (img_1)
    img_1 = Dropout(0.2) (img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2, 2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2, 2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D() (img_1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation) ((BatchNormalization(momentum=bn_model)) (input_1))
    img_2 = MaxPooling2D((2, 2)) (img_2)
    img_2 = Dropout(0.2) (img_2)
    img_2 = GlobalMaxPooling2D() (img_2)

    img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))

    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_model) (Dense(256, activation=p_activation)(img_concat)))
    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_model) (Dense(64, activation=p_activation)(dense_layer)))
    output = Dense(1, activation="sigmoid")(dense_layer)

    model = Model([input_1, input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def train(model, X_train, X_angle_train, y_train, X_valid, X_angle_valid, y_valid):
    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    history = model.fit([X_train, X_angle_train], y_train, epochs=20,
                         validation_data=([X_valid, X_angle_valid], y_valid),
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
np.random.seed(666)
[X_train, X_angle_train, y_train, X_valid, X_angle_valid, y_valid, X_test, X_angle_test] = load_data()
model = get_model()
model.summary()
history = train(model, X_train, X_angle_train, y_train, X_valid, X_angle_valid, y_valid)
#show_results(history)
