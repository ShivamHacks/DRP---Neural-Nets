import numpy as np
from copy import copy, deepcopy
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

np.random.seed(10)

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
		structure[0]  = number of inputs
		structure[-1] = number of outputs
		"""
		self.structure = structure # CHECK HERE
		self.num_layers = len(structure) - 1
		self.weights = [ np.random.randn(n_neurons, n_weights) for n_weights, n_neurons in zip(structure[:-1], structure[1:]) ]
		self.biases = [ np.random.randn(n_neurons, ) for n_neurons in structure[1:] ]

	def pass_through(self, input, layer):
		return np.dot(self.weights[layer], input) + self.biases[layer]

	def predict(self, input):
		for i in range(0, self.num_layers - 1):
			input = activation(self.pass_through(input, i))
		return self.pass_through(input, -1) # no activation on last layer

	def cost(self, output, actual):
		return 0.5 * np.linalg.norm(output - actual)

	def train_one(self, train_x, train_y, learning_rate, eta=0.1):

		weight_gradients = []
		bias_gradients = []

		for i in range(self.num_layers): # iterate over layers

			layer_weight_gradients = np.zeros(self.weights[i].shape)
			layer_bias_gradients = np.zeros(self.biases[i].shape)

			# CHECK HERE
			for j in range(self.structure[i + 1]): # iterate over neuron
				
				# calculate bias gradients for this neuron
				self.biases[i][j] += eta
				cost_adjusted_bias_more = self.cost(self.predict(train_x), train_y)
				self.biases[i][j] -= 2 * eta
				cost_adjusted_bias_less = self.cost(self.predict(train_x), train_y)
				self.biases[i][j] += eta
				bias_gradient = (cost_adjusted_bias_more - cost_adjusted_bias_less) / (2 * eta)
				
				layer_bias_gradients[j] = bias_gradient

				#neuron_weight_gradients = []

				for k in range(len(self.weights[i][j])):

					self.weights[i][j][k] += eta
					cost_adjusted_weight_more = self.cost(self.predict(train_x), train_y)
					self.weights[i][j][k] -= 2 * eta
					cost_adjusted_weight_less = self.cost(self.predict(train_x), train_y)
					self.weights[i][j][k] += eta
					weight_gradient = (cost_adjusted_weight_more - cost_adjusted_weight_less) / (2 * eta)
					
					layer_weight_gradients[j][k] = weight_gradient

				#layer_weight_gradients.append(neuron_weight_gradients)

			weight_gradients.append(layer_weight_gradients)
			bias_gradients.append(layer_bias_gradients)

		#print [weights_gradient.shape for weights, weights_gradient in zip(self.weights, weight_gradients)]
		self.weights = [weights - learning_rate * weights_gradient for weights, weights_gradient in zip(self.weights, weight_gradients)]
		self.biases = [biases - learning_rate * bias_gradient for biases, bias_gradient in zip(self.biases, bias_gradients)]

		#self.weights -= learning_rate * weight_gradients
		#self.biases -= learning_rate * bias_gradients

		"""for i in range(len(weight_gradients)):
			for j in range(len(weight_gradients[i])):
				for k in range(len(weight_gradients[i][j])):
					self.layers[i].weights[j][k] -= learning_rate * weight_gradients[i][j][k]
		
		for i in range(len(bias_gradients)):
			for j in range(len(bias_gradients[i])):
				self.layers[i].biases[j] -= learning_rate * bias_gradients[i][j]"""

	def train(self, X, y, learning_rate, epochs):
		X, y = shuffle(X, y)
		for i in range(epochs):
			for j in range(len(X)):
				self.train_one(X[j], y[j], learning_rate)
			print 'Completed Epoch: ' + str(i+1) + ' Error: ' + str(mean_squared_error(self.evaluate_multiple(X), y))

	def evaluate_multiple(self, input):
		output = []
		for i in range(len(input)):
			output.append(self.predict(input[i]))
		return output


def activation(x):
	return np.tanh(x)

def activation_derivative(x):
	return 1.0 - np.tanh(x)**2

# Test Network

myNN = Network([1, 3, 3, 1])

def func_to_predict(x): return x ** 3 + (x - 1) ** 2

dataRange = [-1.0, 1.0]
step = 0.05
numDataPoints = int( (dataRange[1] - dataRange[0]) / step) + 1
learning_rate = 0.1
num_epochs = 100

x = np.arange(dataRange[0], dataRange[1] + step, step)
y = func_to_predict(x)

train_x = np.array(x).reshape(numDataPoints, 1)
train_y = np.array(y).reshape(numDataPoints, 1)

myNN.train(train_x, train_y, learning_rate, num_epochs)

y_test = []
for i in range(len(train_x)):
	y_test.append(myNN.predict(train_x[i]))
y_test = np.array(y_test).reshape(numDataPoints, 1)

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
