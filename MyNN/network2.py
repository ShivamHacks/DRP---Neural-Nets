import numpy as np
from copy import copy, deepcopy
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

np.random.seed(6)

# possibly change talk topic to:
# Neural Networks as Polynomial Regressors
# Using Neural Networks as a Universal Approximator

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		self.num_inputs = num_inputs
		self.num_neurons = num_neurons
		self.weights = np.random.randn(num_neurons, num_inputs)
		#print self.weights
		self.biases = np.random.randn(num_neurons, )
		#print self.biases

	def pass_through(self, input):
		"""outputs = []
		for i in range(self.num_neurons):
			dot = np.dot(self.weights[i], input) + self.biases[i]
			outputs.append(activation(dot))
		return outputs"""
		return np.dot(input, self.weights.T) + self.biases

class Network(object):

	def __init__(self, structure):
		"""
		structure[0] = number of inputs
		structure[len(structure) - 1] = number of outputs
		"""
		self.structure = structure
		self.numLayers = len(structure) - 1
		print "Number of layers: " + str(self.numLayers)
		self.layers = []
		print "Network Mapping Layer By Layer: "
		for i in range(1, len(structure)):
			print str(structure[i -1]) + " to " + str(structure[i])
			self.layers.append(Layer(structure[i - 1], structure[i]))

	def evaluate(self, input):
		# RelU activation with linear output
		for i in range(0, self.numLayers - 1):
			input = activation(self.layers[i].pass_through(input))
		return self.layers[-1].pass_through(input) # no activation on last layer

	def evaluate_multiple(self, input):
		output = []
		for i in range(len(input)):
			output.append(self.evaluate(input[i]))
		return output

	def train_one(self, trainX, trainY, learningRate):

		layer_inputs = [ ]
		layer_outputs = [ ]
		input = trainX
		for i in range(0, self.numLayers):
			layer_inputs.append(input)
			output = self.layers[i].pass_through(input)
			layer_outputs.append(output)
			input = activation(output)

		"""print 'layer_ins: ' + str(len(layer_inputs))
		print 'layer_in_0: ' + str(layer_inputs[0].shape)
		print 'layer_in_1: ' + str(layer_inputs[1].shape)

		print 'layer_outs: ' + str(len(layer_outputs))
		print 'layer_out_0: ' + str(layer_outputs[0].shape)
		print 'layer_out_1: ' + str(layer_outputs[1].shape)"""
		#layer_inputs.append(input) # inputs to last layer
		#layer_outputs.append(self.layers[-1].pass_through(input)) # last layer not activated

		error = input - trainY # output error

		for i in reversed(range(0, self.numLayers)):
			current_layer_weights = deepcopy(self.layers[i].weights)

			self.layers[i].weights -= learningRate * np.outer(error, activation(layer_inputs[i]) )
			self.layers[i].biases -= learningRate * error

			error = np.multiply(np.dot(current_layer_weights.T , error), activation_derivative(layer_outputs[i-1]))

		#self.layers[0].weights -= learningRate * np.outer(error, activation(layer_inputs[0]) )
		#self.layers[0].biases -= learningRate * error



		"""
		error = input - trainY # output error
		output_layer_weights = deepcopy(self.layers[-1].weights)
		self.layers[-1].weights -= learningRate * np.outer(error, activation(layer_outputs[-1]) )
		self.layers[-1].biases -= learningRate * error

		error = np.multiply(np.dot(output_layer_weights.T , error), activation_derivative(layer_inputs[-2]))

		current_layer_weights = deepcopy(output_layer_weights)
		for i in reversed(range(0, self.numLayers - 1)):
			#print 'i ran!'
			error = np.multiply(np.dot(current_layer_weights.T , error), activation_derivative(layer_inputs[i - 1]))
			current_layer_weights = deepcopy(self.layers[i].weights)
			self.layers[i].weights -= learningRate * np.outer(error, layer_outputs[i - 1])
			self.layers[i].biases -= learningRate * error
			#error = np.multiply(np.dot(current_layer_weights.T , error), activation_derivative(layer_inputs[i]))
		"""

	def train(self, X, y, learning_rate, epochs):
		X, y = shuffle(X, y)
		for i in range(epochs):
			for j in range(len(X)):
				self.train_one(X[j], y[j], learning_rate)
			print 'Completed Epoch: ' + str(i+1) + ' Error: ' + str(mean_squared_error(self.evaluate_multiple(X), y))


def activation(x):
	#return x * (x > 0) #ReLu
	#return 1.0/(1.0+np.exp(-x)) # Sigmoid
	return np.tanh(x) # tanh

def activation_derivative(x):
	# consider Leaky ReLu
	#return (x > 0) * 1 #ReLu
	#return activation(x)*(1-activation(x)) # Sigmoid
	return 1.0 - np.tanh(x)**2 # tanh

# Test Network

myNN = Network([1, 5, 1])
def func_to_predict(x): return x ** 2
"""
Good tests:

x ** 2
np.exp(x)
0.5 * x ** 3 + 0.5 * x ** 2
0.5 * np.sin(x) + 0.5 * np.cos(x)

"""

print "output: " + str(myNN.evaluate(0.5))

dataRange = [-1.0, 1]
step = 0.05
numDataPoints = int( (dataRange[1] - dataRange[0]) / step) + 1
learning_rate = 0.1
numEpochs = 100

x = np.arange(dataRange[0], dataRange[1] + step, step)
y = func_to_predict(x)

train_x = np.array(x).reshape(numDataPoints, 1)
train_y = np.array(y).reshape(numDataPoints, 1)

myNN.train(train_x, train_y, learning_rate, numEpochs)

y_test = np.array(myNN.evaluate_multiple(x)).reshape(1, numDataPoints)

# PLOT

import matplotlib.pyplot as plt

plt.scatter(train_x, train_y, color='b')
plt.scatter(train_x, y_test, color='r')

x_graph = np.linspace(dataRange[0], dataRange[1], 1000)
plt.plot(x_graph, func_to_predict(x_graph), color='y')

plt.show()

"""

Issues: data range can only be between -1 and 1 --> both x and y
^ If i do exponential, doesn't work when x > 0 b/c y > 1
when numepochs increases too much, error goes up (say over 500 epochs)
program works better when last layer is activated

"""
