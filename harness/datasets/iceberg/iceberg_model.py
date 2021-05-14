from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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
