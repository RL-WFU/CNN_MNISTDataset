"""
Adapted from Jason Brownlee
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


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
    # Predict categorical variable (digit)
    digit = model.predict_classes(img)
    print(digit[0])


# Entry Point
# Initialize weights and example images to load
weights = "final_model.h5"
example5 = "PredictionImages/example_5.png"
example7 = "PredictionImages/example_7.png"

# Predict classes for each example image
run_example(example5, weights)
run_example(example7, weights)
