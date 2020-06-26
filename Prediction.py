"""
Benjamin Raiford — June 2020
Convolutional Neural Network for MNIST
    Predict the class given an image

This project owes a substantial debt to Jason Brownlee's tutorial at:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from os import listdir


# Load and Prepare Image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# Show a 28x28 image that is passed to model.predict()
def plot_image(image):
    img = image.reshape(1, 28, 28)
    plt.figure()
    plt.imshow(img[0])
    plt.show()


# Load (a single) image and predict class
def run_example(ex_image, saved_weights):
    # Load Image
    img = load_image('%s' % ex_image)
    # Load model weights
    model = load_model('%s' % saved_weights)

    # Print probabilities
    digit_probs = model.predict(img)
    print(digit_probs)

    # Print digit with highest probability, and certainty level
    digit = np.argmax(digit_probs, axis=-1)
    percent_certain = digit_probs[-1, digit[0]]
    print(digit[0], "with {:.3%}".format(percent_certain), "certainty\n")


# Classify every image in a given directory
def run_all(directory, saved_weights):
    """
    :param directory: Pass the path of the directory you'd like to read (ex: "PredictionImages")
    :param saved_weights: Pass the weights of the model you are using for classification
    """

    # Load weights
    model = load_model('%s' % saved_weights)

    # For every image x in directory, classify x as 0-9
    file_list = listdir(directory)
    file_list.sort()
    for filename in file_list:
        if not filename.startswith('.'):
            # Show probabilities for each class
            # NOTE: if you use this, comment out the rest of this function (redundant information)
            # run_example("%s/%s" % (directory, filename), saved_weights)

            # Load image
            img = load_image("%s/%s" % (directory, filename))

            # Prediction and certainty level
            digit_probs = model.predict(img)
            digit = np.argmax(digit_probs, axis=-1)
            percent_certain = digit_probs[-1, digit[0]]

            # DEBUG: Show image to check how the .png has translated to the 28x28 array
            # plot_image(img)

            # Print filename, prediction, and certainty level
            print(filename, digit[0], percent_certain)


# Entry Point
# Initialize weights and example images to load
test_directory = "PredictionImages"
weights = "final_model.h5"
run_all(test_directory, weights)

"""
Helpful articles:
https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
https://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
"""
