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
def run_example():
    # Load Image
    img = load_image('PredictionImages/example_5.png')
    # Load model weights
    model = load_model('final_model.h5')
    # Predict categorical variable (digit)
    digit = model.predict_classes(img)
    print(digit[0])


# Entry Point
run_example()
