from keras.models import load_model

# Load model
model = load_model("final_model.h5")
# Print a summary of the model (layers and num_parameters)
model.summary()

# Print the weights and biases for every layer
num_layers = len(model.layers)
for i in range(num_layers):
    # Set next layer and print descriptive info
    curr_layer = model.layers[i]
    print("\nLayer", i, "—", curr_layer.name)

    # Set weights
    # Observe that get_weights() returns weights in [0] and biases in [1]
    weights = curr_layer.get_weights()
    if len(weights) > 0:
        print("\tWeights", i)
        print(weights[0])
        print("\tBiases", i)
        print(weights[1])
    # Skip layers that have no weights (ex: reshaping layers)
    else:
        print("\tEmpty Layer")
