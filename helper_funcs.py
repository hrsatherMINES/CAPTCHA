import string
from difflib import SequenceMatcher

import cv2
import emnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras

import global_params as gp


def _load_emnist_data():
    X_train, y_train = emnist.extract_training_samples("byclass")
    X_test, y_test = emnist.extract_test_samples("byclass")

    X_train = X_train.astype("float32") / 255
    X_train[X_train >= gp.PIXEL_THRESH] = 1
    X_train[X_train < gp.PIXEL_THRESH] = 0
    X_train = np.expand_dims(X_train, -1)

    X_test = X_test.astype("float32") / 255
    X_test[X_test >= gp.PIXEL_THRESH] = 1
    X_test[X_test < gp.PIXEL_THRESH] = 0
    X_test = np.expand_dims(X_test, -1)
    
    y_train = keras.utils.to_categorical(y_train, gp.NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, gp.NUM_CLASSES)

    return (X_train, X_test, y_train, y_test)


def _train_model(X_train, y_train, epochs, visualize):
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

    if visualize:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    model.save("model.keras")

    return model


def get_model(retrain=False, epochs=15, evaluation_metrics=False):
    X_train, X_test, y_train, y_test = _load_emnist_data()

    if retrain:
        model = _train_model(X_train, y_train, epochs, evaluation_metrics)
    else:
        model = keras.models.load_model("model.keras")

    if evaluation_metrics:
        _, acc = model.evaluate(X_test, y_test, batch_size=128)
        print("Accuracy on EMNIST test data:", acc)

    return model


def similar(a, b):
    return 100 * SequenceMatcher(None, a, b).ratio()


def transform_string(input_str):
    input_str = input_str.upper()
    
    def _convert_char(char):
        if char in gp.CHAR_MAPPING:
            return gp.CHAR_MAPPING[char]
        else:
            return char

    output_list = [_convert_char(char) for char in input_str]

    output_str = "".join(output_list)

    return output_str


def _crop_image(img, centroid):
    centroid = [round(x) for x in centroid]
    left = round(centroid[1] - gp.IMG_SIZE/2)
    right = round(centroid[1] + gp.IMG_SIZE/2)
    top = round(centroid[0] - gp.IMG_SIZE/2)
    bottom = round(centroid[0] + gp.IMG_SIZE/2)
    img_slice = img[left:right, top:bottom]

    return img_slice


def _decode_prediction(prediction):
    possible_values = [str(x) for x in range(10)] + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    decode_prediction_dict = dict(zip(range(gp.NUM_CLASSES), possible_values))

    value = np.argmax(prediction)

    return decode_prediction_dict[value]


def _img_to_prediction(cropped_image, model, verbose=False):
    cropped_image_expanded = np.expand_dims(cropped_image, axis=0)

    if cropped_image_expanded.shape != (1, gp.IMG_SIZE, gp.IMG_SIZE):
        return None

    encoded_prediction = model.predict(cropped_image_expanded)
    prediction = _decode_prediction(encoded_prediction)
    
    if verbose:
        print("\nConfidence:", np.max(encoded_prediction))
        print("Predicted value:", prediction)

    return prediction 


def _sort_preds(x_loc_list, prediction_list):
    zipped_lists = zip(x_loc_list, prediction_list)
    sorted_pairs = sorted(zipped_lists)

    sorted_pairs = zip(*sorted_pairs)
    _, output = (list(pair) for pair in sorted_pairs)
    sorted_prediction = "".join(output)

    return sorted_prediction


def _process_captcha(img):
    # Invert color
    img = 255 - img
    # Take the max of each band
    img = np.max(img, axis=2)
    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, 10, 3, 27) # have to denoise it before dividing by 255
    img = 255 - img
    # Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    img = 255 - img
    img = cv2.blur(img, (3,3))
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    return img


def read_captcha(img, model):
    img = _process_captcha(img)

    # Find connected components
    num_components, _, stats, centroids = cv2.connectedComponentsWithStats(img, 1, cv2.CV_8U)
    #This section gets rid of small component areas
    temp_components = 0
    temp_centroids = []
    for i in range(len(stats)):
        if stats[i][4] > gp.MIN_AREA:
            temp_components +=1
            temp_centroids.append(centroids[i])
    centroids = temp_centroids
    num_components = temp_components

    # Normalize
    img = img / 255
    img[img >= gp.PIXEL_THRESH] = 1
    img[img < gp.PIXEL_THRESH] = 0

    x_loc_list = []
    prediction_list = []
    for i in range(1, num_components):
        centroid = centroids[i]

        cropped_image = _crop_image(img, centroid)
       
        prediction = _img_to_prediction(cropped_image, model, verbose=False)
        if prediction is None:  # One of the images is messed up
            continue

        x_loc_list.append(centroid[0])
        prediction_list.append(prediction)

        prediction = _sort_preds(x_loc_list, prediction_list)

    return prediction