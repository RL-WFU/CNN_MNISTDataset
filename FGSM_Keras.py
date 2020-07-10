from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Prediction import load_image


# Plot image
def plot_image(image, probs=None):
    img = image.reshape(1, 28, 28)

    # plot figure, if probs are passed, show the prediction and certainty
    plt.figure()
    if probs is not None:
        label, certainty = top_n_predictions(probs, 1)
        plt.title("{} with {:.3%} certainty".format(label[0], certainty[0]))
    plt.imshow(img[0])
    plt.show()


# Return indices and certainty of top n predictions
def top_n_predictions(probs, n=3):
    """
    :param probs: probabilities from model
    :param n: number of top predictions (i.e. n == 3 will return top 3 predictions)
    :return: the indices and certainty of the top n predictions
    """
    probs = probs[0]
    top_indices = np.argpartition(probs, -n)[-n:]
    top_indices_sorted = top_indices[np.argsort(probs[top_indices])]
    top_sorted = probs[top_indices_sorted]

    return top_indices_sorted, top_sorted


# Print top n predictions
def print_top_n_predictions(probs, n=3):
    labels, certainties = top_n_predictions(probs, n)
    for i in range(n - 1, -1, -1):
        print("{} with {:.9%} certainty".format(labels[i], certainties[i]))


# Make noise for the attack
def adversarial_pattern(image, label):
    # Cast image to tensor
    image = tf.cast(image, tf.float32)

    # Watch the image tensor, calculate loss
    with tf.GradientTape() as tape:
        tape.watch(image)
        probs = model(image)
        loss = categorical_crossentropy(label, probs[0])

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient).numpy()

    return signed_grad


# Run an FGSM attack on an image
def fgsm_attack(img_path):
    # FIXME: very confident predictions give zero gradients
    """
      Two possible solutions (not mutually exclusive):
        1) calculate the gradients before softmax; this might require a new model(!)
        2) change the way initial_class is done (not via prediction but via correct label)
    """
    # Process image
    img = load_image(img_path)

    # Get initial predictions
    predictions = model.predict(img)
    initial_class = np.argmax(predictions)

    # Initialize adversarial example with input image
    img_adv = img

    # Original class made one-hot in order to create perturbation
    target = to_categorical(initial_class, 10)

    # Create and show perturbation
    perturbation = adversarial_pattern(img, target)
    plot_image(perturbation)

    # Set epsilons and iterate through
    epsilons = [0, 0.01, 0.02, 0.03, 0.04]
    for epsilon in epsilons:
        print("\n epsilon:", epsilon)

        # Create adversarial image
        img_adv = img_adv + epsilon * perturbation
        # Make prediction from adversarial image
        adv_predictions = model.predict(img_adv)
        # Plot adversarial image
        plot_image(img_adv, adv_predictions)
        # Print top 5 predictions
        print_top_n_predictions(adv_predictions, 5)


# Entry point
if __name__ == "__main__":
    # Set model and image paths
    model = load_model("final_model.h5")
    test_image = "PredictionImages/example_5.png"
    fgsm_attack(test_image)

"""
Helpful articles:
https://github.com/keras-team/keras/issues/5881
https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
Literally whatever you have to read to understand GradientTape()
"""
