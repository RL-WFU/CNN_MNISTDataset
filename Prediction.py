"""
Adapted from Jason Brownlee
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


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


# Load image and predict class
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


# Entry Point
# Initialize weights and example images to load
weights = "final_model.h5"
example5 = "PredictionImages/example_5.png"
example7 = "PredictionImages/example_7.png"

# Predict classes for each example image
run_example(example5, weights)
run_example(example7, weights)

"""
Helpful articles:
https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
"""