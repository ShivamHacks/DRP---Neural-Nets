import numpy as np
from copy import copy, deepcopy
from sklearn import preprocessing
from sklearn.utils import shuffle

np.random.seed(6)

# possibly change talk topic to:
# Neural Networks as Polynomial Regressors
# Using Neural Networks as a Universal Approximator

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		#self.weights = np.zeros((num_neurons, num_inputs))
		#self.biases = np.zeros((num_neurons, ))
		#print self.weights
		#self.biases = np.zeros((num_neurons, ))
		#self.weights = np.random.randint(10, 15, [num_neurons, num_inputs]) # make randn
		#self.weights = np.random.randn(num_neurons, num_inputs)
		self.weights = np.random.randn(num_neurons, num_inputs)
		self.biases = np.random.randn(num_neurons, )

	def pass_through(self, input):
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
		return self.layers[self.numLayers - 1].pass_through(input)

	def evaluate_multiple(self, input):
		output = []
		for i in range(len(input)):
			output.append(self.evaluate(input[i]))
		return output

	def train_one(self, trainX, trainY, learningRate):

		#print "Evaluation for: X" + str(trainX)
		#print "\tPredicted: " + str(self.evaluate(trainX))
		#print "\tActual: " + str(trainY)
		#print "\tDifference: " + str(self.evaluate(trainX) - trainY)

		layer_inputs = []
		layer_outputs = []
		input = trainX
		for i in range(0, self.numLayers): # don't need last output
			layer_inputs.append(input)
			input = activation(self.layers[i].pass_through(input))
			layer_outputs.append(input)

		error = self.evaluate(trainX) - trainY
		for i in reversed(range(0, self.numLayers)):
			#print "SHAPE: "+ str(error.shape)
			current_layer_weights = deepcopy(self.layers[i].weights)
			#print "CURRENT: " + str(current_layer_weights)
			#print "prev outputs: " + str(layer_outputs[i - 1])
			#print "ADJUSTMENT: " + str(learningRate * np.outer(error, layer_outputs[i - 1]))
			self.layers[i].weights -= learningRate * np.outer(error, layer_outputs[i - 1]) # output error linearly related to output layer input
			#print "ADJUSTED: " + str(self.layers[i].weights)
			self.layers[i].biases -= learningRate * error
			error = np.multiply(np.dot(current_layer_weights.T , error), activation_derivative(layer_inputs[i]))
			#error = error.reshape(error.shape[0], 1)

	def train(self, X, y, learning_rate, epochs):
		for i in range(epochs):
			for j in range(len(X)):
				self.train_one(X[j], y[j], learning_rate)
			print 'Completed Epoch: ' + str(i+1) + ' Error: ' + str(np.linalg.norm(self.evaluate_multiple(X) - y))


def activation(x):
	return x * (x > 0) #ReLu
	#return 1.0/(1.0+np.exp(-x)) # Sigmoid

def activation_derivative(x, epsilon=0.1):
	# consider Leaky ReLu
	return (x > 0) * 1 #ReLu
	#return activation(x)*(1-activation(x)) # Sigmoid

# Test Network

myNN = Network([1, 3, 1])
def func_to_predict(x): return x ** 2

dataRange = [-1.0, 1.0]
step = 0.05
numDataPoints = int( (dataRange[1] - dataRange[0]) / step) + 1
learning_rate = 0.05
numEpochs = 100

x = np.arange(dataRange[0], dataRange[1] + step, step)
y = func_to_predict(x)
x, y = shuffle(x, y)

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