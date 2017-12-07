import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

np.random.seed(10)

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		self.num_inputs = num_inputs
		self.num_neurons = num_neurons
		self.weights = np.random.randn(num_neurons, num_inputs)
		self.biases = np.random.randn(num_neurons, )

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

	def train_one(self, train_x, train_y, learning_rate, eta=0.01):

		weight_gradients = []
		bias_gradients = []

		for i in range(self.num_layers): # iterate over layers

			layer_weight_gradients = np.zeros(self.weights[i].shape)
			layer_bias_gradients = np.zeros(self.biases[i].shape)

			# CHECK HERE
			for j in range(self.structure[i + 1]): # iterate over neurons
				
				# calculate bias gradients for this neuron
				self.biases[i][j] += eta
				cost_adjusted_bias_more = self.cost(self.predict(train_x), train_y)
				self.biases[i][j] -= 2 * eta
				cost_adjusted_bias_less = self.cost(self.predict(train_x), train_y)
				self.biases[i][j] += eta
				bias_gradient = (cost_adjusted_bias_more - cost_adjusted_bias_less) / (2 * eta)
				
				layer_bias_gradients[j] = bias_gradient

				for k in range(len(self.weights[i][j])):

					self.weights[i][j][k] += eta
					cost_adjusted_weight_more = self.cost(self.predict(train_x), train_y)
					self.weights[i][j][k] -= 2 * eta
					cost_adjusted_weight_less = self.cost(self.predict(train_x), train_y)
					self.weights[i][j][k] += eta
					weight_gradient = (cost_adjusted_weight_more - cost_adjusted_weight_less) / (2 * eta)
					
					layer_weight_gradients[j][k] = weight_gradient

			weight_gradients.append(layer_weight_gradients)
			bias_gradients.append(layer_bias_gradients)

		self.weights = [weights - learning_rate * weights_gradient for weights, weights_gradient in zip(self.weights, weight_gradients)]
		self.biases = [biases - learning_rate * bias_gradient for biases, bias_gradient in zip(self.biases, bias_gradients)]

	def train(self, X, y, learning_rate, epochs):
		X, y = shuffle(X, y)
		for i in range(epochs):
			for j in range(len(X)):
				self.train_one(X[j], y[j], learning_rate)
			print 'Completed Epoch: ' + str(i+1) + ' Error: ' + str(mean_squared_error(self.predict_multiple(X), y))

	def predict_multiple(self, input):
		output = []
		for i in range(len(input)):
			output.append(self.predict(input[i]))
		return output


def activation(x):
	#return np.tanh(x)
	return x * (x > 0) #ReLu

def activation_derivative(x):
	#return 1.0 - np.tanh(x)**2
	return (x > 0) * 1 #ReLu

def generate_data(data_range, step, data_split, function_to_predict):
	num_data_points = int( (data_range[1] - data_range[0]) / step) + 1
	x = np.arange(data_range[0], data_range[1] + step, step).reshape(num_data_points, 1)
	y = function_to_predict(x)
	x, y = shuffle(x, y)
	train_x = x[:int(num_data_points * data_split)]
	train_y = y[:int(num_data_points * data_split)]
	test_x = x[int(num_data_points * data_split):]
	test_y = y[int(num_data_points * data_split):]
	continuous_x = np.linspace(data_range[0], data_range[1], 1000)
	return train_x, train_y, test_x, test_y, continuous_x

# Build Network

myNN = Network([1, 3, 1])

def func_to_predict(x): return x ** 2 #50 * x ** 2 #x ** 2 * np.sin(1 / x)

train_x, train_y, test_x, test_y, continuous_x = generate_data([-1.0, 1.0], 0.005, 0.8, func_to_predict)

# Train Network
learning_rate = 0.01
num_epochs = 50
myNN.train(train_x, train_y, learning_rate, num_epochs)

# Plot results

import matplotlib.pyplot as plt

plt.scatter(test_x, test_y, color='b')
plt.scatter(test_x, myNN.predict_multiple(test_x), color='r')

# Graph continuous function
plt.plot(continuous_x, func_to_predict(continuous_x), color='y')

plt.show()

"""

Issues: data range can only be between -1 and 1 --> both x and y
^ If i do exponential, doesn't work when x > 0 b/c y > 1
when numepochs increases too much, error goes up (say over 500 epochs)
program works better when last layer is activated

"""
