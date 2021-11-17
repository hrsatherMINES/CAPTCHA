import emnist
import string
import numpy as np

from tensorflow.keras import layers
from tensorflow import keras

import global_params as gp


def _load_emnist_data():
    X_train, y_train = emnist.extract_training_samples("byclass")
    X_test, y_test = emnist.extract_test_samples("byclass")

    X_train = X_train.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = X_test.astype("float32") / 255
    X_test = np.expand_dims(X_test, -1)

    y_train = keras.utils.to_categorical(y_train, gp.NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, gp.NUM_CLASSES)

    return (X_train, X_test, y_train, y_test)


def _train_model(X_train, y_train, epochs):
    input_shape = (gp.IMG_SIZE, gp.IMG_SIZE, 1)

    model = keras.Sequential([keras.Input(shape=input_shape),
                              layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                              layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                              layers.Flatten(),
                              layers.Dropout(0.5),
                              layers.Dense(gp.NUM_CLASSES, activation="softmax")])
                    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=gp.BATCH_SIZE, epochs=epochs, validation_split=0.1)

    model.save("model.keras")

    return model



def get_model(retrain=False, epochs=15, evaluation_metrics=True):
    X_train, X_test, y_train, y_test = _load_emnist_data()

    if retrain:
        model = _train_model(X_train, y_train, epochs)
    else:
        model = keras.models.load_model("model.keras")

    if evaluation_metrics:
        _, acc = model.evaluate(X_test, y_test, batch_size=128)
        print("Accuracy on EMNIST test data:", acc)

    return model


def decode_prediction(prediction):
    possible_values = [str(x) for x in range(10)] + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    decode_prediction_dict = dict(zip(range(gp.NUM_CLASSES), possible_values))

    value = np.argmax(prediction)
    return decode_prediction_dict[value]