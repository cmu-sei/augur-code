from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sklearn.model_selection as skm

from training.training_set import TrainingSet


def create_model():
    """Model to be used, obtained from sample solution."""
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


def split_data(dataset, validation_percentage):
    """Split training set into train and validation (75% to actually train)"""
    x_band_t, x_band_v, x_angle_t, x_angle_v, y_train, y_validation = skm.train_test_split(dataset.x_combined_bands,
                                                                                           dataset.x_angle,
                                                                                           dataset.y_output,
                                                                                           random_state=42,
                                                                                           test_size=validation_percentage)
    print("Done splitting validation data from train data", flush=True)
    training_set = TrainingSet()
    training_set.x_train = [x_band_t, x_angle_t]
    training_set.x_validation = [x_band_v, x_angle_v]
    training_set.y_train = y_train
    training_set.y_validation = y_validation
    training_set.num_train_samples = y_train.shape[0]
    training_set.num_validation_samples = y_validation.shape[0]
    return training_set


def get_fold_data(dataset, train_index, test_index):
    """Prepares a training set for the given dataset and indexes"""
    training_set = TrainingSet()
    training_set.x_train = [dataset.x_combined_bands[train_index], dataset.x_angle[train_index]]
    training_set.y_train = dataset.y_output[train_index]
    training_set.x_validation = [dataset.x_combined_bands[test_index], dataset.x_angle[test_index]]
    training_set.y_validation = dataset.y_output[test_index]
    training_set.num_train_samples = training_set.y_train.shape[0]
    training_set.num_validation_samples = training_set.y_validation.shape[0]
    return training_set
