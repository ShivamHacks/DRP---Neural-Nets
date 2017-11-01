import numpy as np

"""

A simple neural network program with
adjustable parameters

"""

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def evaluate(X, layer):

    if layer < len(network_struct):

        transformations = []

        for neuron in range(0, network_struct[layer]):

            transformation = sigmoid(np.dot(weights[layer - 1][neuron], X))
            transformations.append(transformation)

        next_layer = layer + 1
        evaluate(transformations, next_layer)

    if layer == len(network_struct) - 1:
        return transformations

# Sample Data
# TODO: X and y
# TODO: add dummy variable
X = [

[1, 1, 1, 0, 1],
[0, 1, 1, 1, 0],
[0, 1, 1, 0, 0]

]

y = [[0], [0.5], [0.3]]

# Network

num_train = 10000
num_inputs = 5
num_layers = 2
num_neurons = [3, 1] # number of neurons in each layer
num_iterations = 10000
learning_rate = 0.01

network_struct = [5, 3, 1] # first val is input vect, etc.

# Initialize Weights Array
weights = [] # weights array
for i in range(1, len(network_struct)):

    layer_weights = np.random.rand(network_struct[i], network_struct[i - 1])
    weights.append(layer_weights)


for i in range(0, num_iterations):

    for j in range(0, num_train):

        predicted = evaluate(X[j], 1)
        diff = predicted - y[i]

        for a in range(1, len(network_struct)):
            for b in range(0, len(network_struct[a])):
                delta = diff * sigmoid_prime(predicted) # need to fix
                weights[a][b] += delta

#evaluate(X[0], 1)

#for i in range(0, num_layers):

#    layer_weights = np.random.rand(num_neurons[i], num_inputs) # incorrect, num_inputs not constant
#    weights.append(layer_weights)

# Train Network
"""for i in range(0, num_iterations):

    for j in range(0, num_train):

        for k in range(0, num_layers):

            transformation = np.dot(weights[k], X[j])

            #predicted = sigmoid(transformation)
            predicted = evaluate(X[j], 0)
            actual = y[j]
            diff = predicted - actual
            print diff
            
            delta = diff * sigmoid_prime(transformation)

            for g in range(0, num_neurons[k]):
                weights[k][g] += delta

print weights"""



#error = np.sqrt(diff.dot(diff)) # norm of difference vector --> don't need this
    #delta = 