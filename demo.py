import argparse
import os

import cv2
import matplotlib.pyplot as plt

import helper_funcs as hf


def predict(image, model):
    predicted_captcha = hf.read_captcha(image, model)
    predicted_captcha = hf.transform_string(predicted_captcha)
    return predicted_captcha


def main():
    # Handle argumants
    parser = argparse.ArgumentParser(description="Read given CAPTCHA image")
    parser.add_argument("-n", "--file_name", type=str, help="Name of the CAPTCHA to be read", required=True)
    args = parser.parse_args()
    
    # Load image
    file_name = args.file_name
    img = cv2.imread(file_name)
    
    # Load model
    print("Loading model...")
    model = hf.get_model(retrain=False)
    print("Model loaded")

    predicted_captcha = predict(img, model)
    print("Prediction:", predicted_captcha)

    # Show image
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
